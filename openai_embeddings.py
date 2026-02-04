import numpy as np
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple
import asyncio

# Импорты для OpenAI
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Импорт для косинусного сходства
from sklearn.metrics.pairwise import cosine_similarity


class OpenAIEmbedder:
    """Класс для создания эмбеддингов через OpenAI API и работы с ними"""

    def __init__(self, api_key: str = None, model: str = "text-embedding-3-small"):
        """
        Инициализация эмбеддера

        Args:
            api_key: Ключ OpenAI API. Если None, берется из .env
            model: Модель для создания эмбеддингов
        """
        load_dotenv()
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.client = None

        if not self.api_key:
            raise ValueError("Не указан OPENAI_API_KEY в .env или при инициализации")

    async def _get_client(self):
        """Создание асинхронного клиента OpenAI"""
        if self.client is None:
            self.client = AsyncOpenAI(api_key=self.api_key)
        return self.client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_embedding(self, text: str) -> List[float]:
        """Получение эмбеддинга для одного текста"""
        client = await self._get_client()

        try:
            response = await client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Ошибка при получении эмбеддинга: {e}")
            raise

    async def embed_query(self, query: str) -> Dict[str, Any]:
        """
        Векторизация запроса пользователя

        Args:
            query: Текст запроса пользователя

        Returns:
            Dict: Словарь с текстом запроса и его вектором
        """
        try:
            embedding = await self.get_embedding(query)
            return {
                'query_text': query,
                'query_embedding': embedding,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Ошибка при векторизации запроса: {e}")
            return {
                'query_text': query,
                'query_embedding': None,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def embed_query_sync(self, query: str) -> Dict[str, Any]:
        """
        Синхронная версия векторизации запроса

        Args:
            query: Текст запроса пользователя

        Returns:
            Dict: Словарь с текстом запроса и его вектором
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self.embed_query(query))
        except RuntimeError:
            pass

        return asyncio.run(self.embed_query(query))

    def calculate_cosine_similarity(self,
                                    query_embedding: List[float],
                                    post_embeddings: List[List[float]]) -> List[float]:
        """
        Расчет косинусного сходства между вектором запроса и векторами сообщений

        Args:
            query_embedding: Вектор запроса пользователя
            post_embeddings: Список векторов сообщений

        Returns:
            List[float]: Список значений косинусного сходства (в процентах)
        """
        if not query_embedding or not post_embeddings:
            return []

        # Преобразуем в numpy массивы
        query_array = np.array(query_embedding).reshape(1, -1)
        posts_array = np.array(post_embeddings)

        # Вычисляем косинусное сходство
        similarities = cosine_similarity(query_array, posts_array)[0]

        # Преобразуем в проценты
        similarities_percent = similarities * 100

        return similarities_percent.tolist()

    def filter_posts_by_similarity(self,
                                   query: str,
                                   posts: List[Dict[str, Any]],
                                   threshold: float = 85.0,
                                   embedding_field: str = 'embedding',
                                   similarity_field: str = 'similarity_score') -> Tuple[
        List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Фильтрация сообщений по косинусному сходству с запросом

        Args:
            query: Запрос пользователя
            posts: Список сообщений с эмбеддингами
            threshold: Порог сходства в процентах (по умолчанию 85%)
            embedding_field: Название поля с эмбеддингом
            similarity_field: Название поля для сохранения score

        Returns:
            Tuple: (все сообщения с score, отфильтрованные сообщения)
        """
        if not posts:
            return [], []

        # Векторизуем запрос
        query_data = self.embed_query_sync(query)
        query_embedding = query_data['query_embedding']

        if not query_embedding:
            print("❌ Не удалось получить эмбеддинг для запроса")
            return [], []

        # Собираем эмбеддинги сообщений
        post_embeddings = []
        valid_posts = []

        for post in posts:
            if embedding_field in post and post[embedding_field] is not None:
                post_embeddings.append(post[embedding_field])
                valid_posts.append(post)

        if not post_embeddings:
            print("⚠️ Нет валидных эмбеддингов в сообщениях")
            return [], []

        # Вычисляем косинусное сходство
        similarities = self.calculate_cosine_similarity(query_embedding, post_embeddings)

        # Добавляем score к сообщениям
        all_posts_with_scores = []
        filtered_posts = []

        for i, (post, similarity) in enumerate(zip(valid_posts, similarities)):
            # Создаем копию поста, чтобы не менять оригинал
            post_with_score = post.copy()
            post_with_score[similarity_field] = similarity
            post_with_score['query'] = query  # Добавляем запрос для контекста

            all_posts_with_scores.append(post_with_score)

            # Проверяем порог
            if similarity >= threshold:
                filtered_posts.append(post_with_score)

        # Сортируем по убыванию сходства
        all_posts_with_scores.sort(key=lambda x: x[similarity_field], reverse=True)
        filtered_posts.sort(key=lambda x: x[similarity_field], reverse=True)

        print(f"📊 Всего сообщений: {len(all_posts_with_scores)}")
        print(f"✅ Сообщений выше порога {threshold}%: {len(filtered_posts)}")

        if filtered_posts:
            print(f"📈 Максимальное сходство: {filtered_posts[0][similarity_field]:.2f}%")
            print(f"📉 Минимальное сходство среди отфильтрованных: {filtered_posts[-1][similarity_field]:.2f}%")

        return all_posts_with_scores, filtered_posts

    def save_similarity_results(self,
                                all_posts: List[Dict[str, Any]],
                                filtered_posts: List[Dict[str, Any]],
                                query: str,
                                base_filename: str = "similarity_results",
                                output_dir: str = "results") -> Tuple[str, str]:
        """
        Сохранение результатов расчета сходства в файлы

        Args:
            all_posts: Все сообщения с score
            filtered_posts: Отфильтрованные сообщения
            query: Исходный запрос
            base_filename: Базовое имя файла
            output_dir: Директория для сохранения

        Returns:
            Tuple: Пути к сохраненным файлам
        """
        # Создаем директорию, если ее нет
        os.makedirs(output_dir, exist_ok=True)

        # Создаем timestamp для уникальности имен файлов
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Подготавливаем данные для сохранения
        query_safe = query.replace(" ", "_").replace("/", "_")[:50]

        # 1. Файл со всеми сообщениями и score
        all_posts_filename = f"{base_filename}_all_{timestamp}_{query_safe}.json"
        all_posts_path = os.path.join(output_dir, all_posts_filename)

        all_results_data = {
            'query': query,
            'total_posts': len(all_posts),
            'timestamp': datetime.now().isoformat(),
            'posts': all_posts
        }

        with open(all_posts_path, 'w', encoding='utf-8') as f:
            json.dump(all_results_data, f, ensure_ascii=False, indent=2)

        print(f"💾 Сохранены все сообщения с score: {all_posts_path}")

        # 2. Файл с отфильтрованными сообщениями
        filtered_posts_filename = f"{base_filename}_filtered_{timestamp}_{query_safe}.json"
        filtered_posts_path = os.path.join(output_dir, filtered_posts_filename)

        filtered_results_data = {
            'query': query,
            'total_posts': len(filtered_posts),
            'timestamp': datetime.now().isoformat(),
            'posts': filtered_posts
        }

        with open(filtered_posts_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_results_data, f, ensure_ascii=False, indent=2)

        print(f"💾 Сохранены отфильтрованные сообщения: {filtered_posts_path}")

        # 3. Также сохраняем в Excel (опционально)
        try:
            import pandas as pd

            # Сохраняем все сообщения в Excel
            all_df = pd.DataFrame(all_posts)
            all_excel_path = os.path.join(output_dir, f"{base_filename}_all_{timestamp}_{query_safe}.xlsx")
            all_df.to_excel(all_excel_path, index=False)
            print(f"📊 Excel со всеми сообщениями: {all_excel_path}")

            # Сохраняем отфильтрованные сообщения в Excel
            if filtered_posts:
                filtered_df = pd.DataFrame(filtered_posts)
                filtered_excel_path = os.path.join(output_dir,
                                                   f"{base_filename}_filtered_{timestamp}_{query_safe}.xlsx")
                filtered_df.to_excel(filtered_excel_path, index=False)
                print(f"📊 Excel с отфильтрованными сообщениями: {filtered_excel_path}")

        except ImportError:
            print("⚠️ Pandas не установлен, Excel файлы не созданы")

        return all_posts_path, filtered_posts_path

    def search_similar_posts(self,
                             query: str,
                             posts: List[Dict[str, Any]],
                             threshold: float = 85.0,
                             save_results: bool = True) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Полный процесс поиска похожих сообщений

        Args:
            query: Запрос пользователя
            posts: Список сообщений с эмбеддингами
            threshold: Порог сходства в процентах
            save_results: Сохранять ли результаты в файлы

        Returns:
            Tuple: (все сообщения с score, отфильтрованные сообщения)
        """
        print(f"\n🔍 Поиск похожих сообщений для запроса: '{query}'")
        print(f"📊 Порог сходства: {threshold}%")

        # Фильтруем сообщения по сходству
        all_posts_with_scores, filtered_posts = self.filter_posts_by_similarity(
            query=query,
            posts=posts,
            threshold=threshold
        )

        # Сохраняем результаты
        if save_results and all_posts_with_scores:
            self.save_similarity_results(all_posts_with_scores, filtered_posts, query)

        return all_posts_with_scores, filtered_posts

    async def add_embeddings_to_posts(self, posts: List[Dict[str, Any]],
                                      text_field: str = 'text',
                                      embedding_field: str = 'embedding',
                                      batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        Добавление эмбеддингов к постам

        Args:
            posts: Список словарей с постами
            text_field: Название поля с текстом
            embedding_field: Название поля для эмбеддинга
            batch_size: Размер батча для асинхронной обработки

        Returns:
            List[Dict]: Посты с добавленными эмбеддингами
        """
        if not posts:
            return posts

        async def process_batch(batch_posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Обработка батча постов"""
            tasks = []
            valid_posts = []

            # Собираем задачи для валидных постов
            for post in batch_posts:
                if text_field in post and post[text_field] and len(post[text_field].strip()) > 0:
                    tasks.append(self.get_embedding(post[text_field]))
                    valid_posts.append(post)
                else:
                    post[embedding_field] = None

            if not tasks:
                return batch_posts

            # Запускаем все задачи параллельно
            embeddings = await asyncio.gather(*tasks, return_exceptions=True)

            # Добавляем эмбеддинги к валидным постам
            for i, (post, embedding) in enumerate(zip(valid_posts, embeddings)):
                if isinstance(embedding, Exception):
                    print(f"⚠️ Ошибка для поста {i}: {embedding}")
                    post[embedding_field] = None
                else:
                    post[embedding_field] = embedding

            return batch_posts

        # Обрабатываем посты батчами
        result_posts = []
        total_batches = (len(posts) + batch_size - 1) // batch_size

        for i in range(0, len(posts), batch_size):
            batch = posts[i:i + batch_size]
            batch_num = i // batch_size + 1
            print(f"🔄 Обработка батча {batch_num}/{total_batches}")

            processed_batch = await process_batch(batch)
            result_posts.extend(processed_batch)

            # Небольшая задержка между батчами, чтобы не превысить лимиты API
            if i + batch_size < len(posts):
                await asyncio.sleep(1)

        print(f"✅ Обработано {len(result_posts)} постов")
        return result_posts

    def add_embeddings_sync(self, posts: List[Dict[str, Any]],
                            text_field: str = 'text',
                            embedding_field: str = 'embedding',
                            batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        Синхронная версия добавления эмбеддингов

        Args:
            posts: Список словарей с постами
            text_field: Название поля с текстом
            embedding_field: Название поля для эмбеддинга
            batch_size: Размер батча для асинхронной обработки

        Returns:
            List[Dict]: Посты с добавленными эмбеддингами
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Если уже есть запущенный loop, создаем новую задачу
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(
                    self.add_embeddings_to_posts(posts, text_field, embedding_field, batch_size)
                )
        except RuntimeError:
            pass

        # Если нет event loop или он уже запущен, создаем новый
        return asyncio.run(
            self.add_embeddings_to_posts(posts, text_field, embedding_field, batch_size)
        )