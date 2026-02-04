import json

from openai_embeddings import OpenAIEmbedder
from telegram_parser import TelegramParser


def main():
    """
    Короткий запуск парсера Telegram с созданием эмбеддингов и поиском
    """
    try:
        # 1. Парсим Telegram
        parser = TelegramParser()
        parser.config.search_limit = 999_999
        parser.config.batch_size = 5

        # Задаем параметры
        channels = [
            'tb_invest_official',
            'bmw_abnn',
            'meduzalive',
        ]

        date_from = '2026-02-01'  # Дата должна быть в прошлом

        print(f"🚀 Начинаем парсинг каналов: {channels}")
        print(f"📅 Период: с {date_from}")

        # Парсим
        messages = parser.parse(channels, date_from)

        # Сохраняем сырой парсинг
        parser.to_json(messages, "telegram_raw.json")
        parser.to_excel(messages, "telegram_raw.xlsx")

        # 2. Создаем эмбеддинги
        embedder = OpenAIEmbedder()
        messages_with_embeddings = embedder.add_embeddings_sync(
            posts=messages,
            text_field='text',
            embedding_field='embedding',
            batch_size=10
        )

        # Сохраняем парсинг + эмбеддинги
        parser.to_json(messages_with_embeddings, "telegram_with_embeddings.json")
        parser.to_excel(messages_with_embeddings, "telegram_with_embeddings.xlsx")
        print(f"✅ Обработано {len(messages_with_embeddings)} сообщений с эмбеддингами")

        # with open('telegram_with_embeddings.json', 'r', encoding='utf-8') as file:
        #     messages_with_embeddings = json.load(file)


        # 3. Пример поиска похожих сообщений
        print("\n" + "=" * 50)
        print("🔍 ПРИМЕРЫ ПОИСКА ПОХОЖИХ СООБЩЕНИЙ")
        print("=" * 50)

        # Примеры запросов
        test_queries = [
            "Сбер дивиденды",
            "Новости о переговорах США, Украины и России",
            "На сколько Астра увеличила отгрузку по итогам 2025 года?",
        ]

        for query in test_queries:
            # Поиск похожих сообщений
            all_results, filtered_results = embedder.search_similar_posts(
                query=query,
                posts=messages_with_embeddings,
                threshold=50.0,
                save_results=True
            )

            # Выводим топ-3 результата
            print(f"\n📋 Топ-3 результата для запроса '{query}':")
            for i, post in enumerate(filtered_results[:3]):
                score = post.get('similarity_score', 0)
                text_preview = post.get('text', '')[:100] + "..." if len(post.get('text', '')) > 100 else post.get(
                    'text', '')
                print(f"  {i + 1}. Сходство: {score:.2f}%")
                print(f"     Текст: {text_preview}")
                print()

            if not filtered_results:
                print(f"  ❌ Нет сообщений с сходством выше 85%")

    except Exception as e:
        print(f"❌ Ошибка в main: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()