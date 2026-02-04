import asyncio
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import stop_after_attempt, wait_exponential, retry

load_dotenv()

class OpenAIEmbedder:
    """Класс для создания эмбеддингов через OpenAI API"""

    def __init__(self, api_key: str = None, model: str = "text-embedding-3-small"):
        """
        Инициализация эмбеддера

        Args:
            api_key: Ключ OpenAI API. Если None, берется из .env
            model: Модель для создания эмбеддингов
        """
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