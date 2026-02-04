from openai_embeddings import OpenAIEmbedder
from telegram_parser import TelegramParser


def main():
    """
    Короткий запуск парсера Telegram с созданием эмбеддингов
    """
    try:
        # 1. Парсим Telegram
        parser = TelegramParser()

        # Задаем параметры
        channels = [
                    'tb_invest_official',
                    # 'pravdadirty',
                    # 'bmw_abnn',
                    # 'meduzalive',
                    ]

        date_from = '2026-01-01'  # Дата должна быть в прошлом

        print(f"🚀 Начинаем парсинг каналов: {channels}")
        print(f"📅 Период: с {date_from}")

        # Парсим
        messages = parser.parse(channels, date_from)

        # Сохраняем сырой парсинг
        parser.to_json(messages, "telegram_raw.json")
        parser.to_excel(messages, "telegram_raw.xlsx")

        # 2. Создаем эмбеддинги (опционально)
        embedder = OpenAIEmbedder()
        messages_with_embeddings = embedder.add_embeddings_sync(
            posts=messages,
            text_field='text',
            embedding_field='embedding',
            batch_size=5
        )

        # Сохраняем парсинг + эмбединги
        parser.to_json(messages_with_embeddings, "telegram_with_embeddings.json")
        parser.to_excel(messages_with_embeddings, "telegram_with_embeddings.xlsx")
        print(f"✅ Обработано {len(messages_with_embeddings)} сообщений с эмбеддингами")

    except Exception as e:
        print(f"❌ Ошибка в main: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()