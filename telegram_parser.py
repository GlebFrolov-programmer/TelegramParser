import os
import asyncio
import json
from datetime import datetime, timezone
from typing import List, Dict, Any
import pandas as pd
from telethon import TelegramClient, errors
from telethon.errors import ChannelPrivateError
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential


load_dotenv()

class TelegramConfig:
    """Конфигурация для парсера Telegram"""

    def __init__(self):
        # Обязательные параметры
        self.api_id = os.getenv('TELEGRAM_API_ID')
        self.api_hash = os.getenv('TELEGRAM_API_HASH')
        self.phone = os.getenv('TELEGRAM_PHONE')

        # Дополнительные параметры
        self.session_name = os.getenv('TELEGRAM_SESSION', 'telegram_session')
        self.search_limit = int(os.getenv('TELEGRAM_SEARCH_LIMIT', '10000'))
        self.template_url = os.getenv('TELEGRAM_TEMPLATE_URL', 'https://t.me/s/{CHANNEL_NAME}/{ID_MESSAGE}')
        self.max_retries = int(os.getenv('TELEGRAM_MAX_RETRIES', '3'))
        self.batch_size = int(os.getenv('TELEGRAM_BATCH_SIZE', '3'))

    def update(self, **kwargs):
        """Обновление параметров конфигурации"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class TelegramParser:
    def __init__(self, config: TelegramConfig = None):
        """
        Инициализация парсера Telegram

        Args:
            config: Конфигурация парсера. Если None, создается дефолтная из .env
        """
        self.config = config or TelegramConfig()
        self.client = None
        self._session_file = f"{self.config.session_name}.session"

        # Проверка обязательных параметров
        if not all([self.config.api_id, self.config.api_hash, self.config.phone]):
            raise ValueError("Не указаны обязательные параметры в .env файле или конфиге")

    async def _get_client(self) -> TelegramClient:
        """Создание или подключение к существующему клиенту Telegram"""
        if self.client is None:
            # Создаем клиента
            self.client = TelegramClient(
                self.config.session_name,
                int(self.config.api_id),
                self.config.api_hash,
            )

        # Подключаем клиента, если не подключен
        if not self.client.is_connected():
            await self.client.connect()

        # Проверяем авторизацию
        if not await self.client.is_user_authorized():
            print("🔑 Сессия не найдена или устарела. Начинаем авторизацию...")
            try:
                await self.client.start(phone=self.config.phone)
                print("✅ Авторизация успешна! Сессия сохранена.")
            except errors.PhoneCodeInvalidError:
                print("❌ Неверный код. Попробуйте еще раз.")
                raise
            except errors.SessionPasswordNeededError:
                print("🔐 Требуется двухфакторная аутентификация.")
                password = input("Введите пароль двухфакторной аутентификации: ")
                await self.client.start(phone=self.config.phone, password=password)
                print("✅ Авторизация успешна! Сессия сохранена.")
            except Exception as e:
                print(f"❌ Ошибка при авторизации: {e}")
                raise
        else:
            print("✅ Используется сохраненная сессия.")

        return self.client

    async def _parse_channel(self, client, channel_name: str, date_from: datetime) -> List[Dict[str, Any]]:
        """Парсинг одного канала"""
        try:
            # Получаем информацию о канале
            try:
                channel = await client.get_entity(channel_name)
            except ValueError:
                # Если канал не найден, пытаемся найти по username
                channel = await client.get_input_entity(channel_name)

            messages = []
            count = 0

            # Парсим сообщения
            async for message in client.iter_messages(
                    channel,
                    # offset_date=date_from,
                    limit=self.config.search_limit
            ):
                if not message:
                    continue

                if message.date > date_from:
                    count += 1

                    if message.text:
                        # Базовые данные
                        msg_data = {
                            'channel': channel_name,
                            'url': f"https://t.me/s/{channel_name}/{message.id}",
                            'text': message.text,
                            'date': message.date.isoformat() if message.date else None,
                            'message_id': message.id,
                        }
                        messages.append(msg_data)
                else:
                    break

            return messages

        except ChannelPrivateError:
            print(f"⚠️ Канал {channel_name} приватный или недоступен")
            return []
        except Exception as e:
            print(f"❌ Ошибка при парсинге канала {channel_name}: {type(e).__name__}: {e}")
            return []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=5, max=10)
    )
    async def parse_async(self, channels: List[str], date_from: datetime, delay_between_batches: float = 1.0) -> List[Dict[str, Any]]:
        """Асинхронный парсинг каналов с разбиением на партии"""
        try:
            client = await self._get_client()
            all_messages = []

            # Разбиваем каналы на партии
            batches = []
            for i in range(0, len(channels), self.config.batch_size):
                batch = channels[i:i + self.config.batch_size]
                batches.append(batch)

            print(f"📦 Разбито на {len(batches)} партий по {self.config.batch_size} каналов в каждой")

            # Обрабатываем каждую партию
            for batch_num, batch in enumerate(batches, 1):
                print(f"\n🔄 Партия {batch_num}/{len(batches)}: {batch}")

                # Создаем задачи для текущей партии
                tasks = []
                for channel in batch:
                    print(f"   📡 Создана задача для: {channel}")
                    task = self._parse_channel(client, channel, date_from)
                    tasks.append(task)

                # Запускаем ВСЕ задачи текущей партии параллельно
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Собираем результаты текущей партии
                batch_messages = []
                for channel, result in zip(batch, results):
                    if isinstance(result, Exception):
                        print(f"   ❌ Ошибка в канале {channel}: {result}")
                    else:
                        print(f"   ✅ Канал {channel}: {len(result)} сообщений")
                        batch_messages.extend(result)

                all_messages.extend(batch_messages)
                print(f"   📊 Партия {batch_num} обработана: {len(batch_messages)} сообщений")

                # Пауза между партиями (если не последняя)
                if batch_num < len(batches):
                    print(f"   ⏸️ Пауза {delay_between_batches} сек...")
                    await asyncio.sleep(delay_between_batches)
            print(f"\n🎉 Все партии обработаны. Всего сообщений: {len(all_messages)}")
            return all_messages

        except Exception as e:
            print(f"❌ Критическая ошибка при парсинге: {e}")
            return []
        finally:
            # Закрываем соединение после завершения работы
            if self.client and self.client.is_connected():
                await self.client.disconnect()
                self.client = None

    def parse(self, channels: List[str], date_from: str) -> List[Dict[str, Any]]:
        """
        Основной метод парсинга

        Args:
            channels: Список названий каналов (например, ['channel1', 'channel2'])
            date_from: Дата начала парсинга в формате 'YYYY-MM-DD'

        Returns:
            List[Dict]: Список сообщений
        """
        try:
            # Преобразуем строку в datetime
            date_from_dt = datetime.strptime(date_from, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )

            # Запускаем асинхронный парсинг
            if hasattr(asyncio, 'get_running_loop'):
                try:
                    loop = asyncio.get_running_loop()
                    # Если уже есть запущенный loop, создаем новую задачу
                    import nest_asyncio
                    nest_asyncio.apply()
                    return loop.run_until_complete(
                        self.parse_async(channels, date_from_dt)
                    )
                except RuntimeError:
                    # Если нет running loop, создаем новый
                    return asyncio.run(self.parse_async(channels, date_from_dt))
            else:
                # Для старых версий Python
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(
                    self.parse_async(channels, date_from_dt)
                )
        except Exception as e:
            print(f"❌ Ошибка в основном методе parse: {e}")
            return []

    def to_excel(self, data: List[Dict[str, Any]], filename: str = "telegram_data.xlsx"):
        """Экспорт в Excel"""
        if not data:
            print("⚠️ Нет данных для экспорта")
            return

        df = pd.DataFrame(data)

        # Создаем Excel writer для настройки
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Telegram Data')


        print(f"✅ Данные сохранены в {filename}")

    def to_json(self, data: List[Dict[str, Any]], filename: str = "telegram_data.json"):
        """Экспорт в JSON"""
        if not data:
            print("⚠️ Нет данных для экспорта")
            return

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ Данные сохранены в {filename}")
