#!/usr/bin/sh

# Скрипт для запуска обучения на всем датасете с перебором гиперпараметров
# Использование: ./train_full_with_tuning.sh

set -e  # Остановка при ошибке

echo "=========================================="
echo "Data Efficiency Training Script"
echo "=========================================="

# 1. Инициализация окружения
echo ""
echo "Шаг 1: Инициализация окружения..."
if ! command -v uv &> /dev/null; then
    echo "Ошибка: uv не установлен. Установите uv: https://github.com/astral-sh/uv"
    exit 1
fi

uv sync
echo "✓ Окружение инициализировано"

# 2. Скачивание датасета
echo ""
echo "Шаг 2: Скачивание датасета..."
if [ -d "./data/train" ] && [ -d "./data/validation" ] && [ -d "./data/test" ]; then
    echo "✓ Датасет уже скачан (пропускаем)"
else
    uv run download_dataset
    echo "✓ Датасет скачан"
fi

# 3. Загрузка переменных окружения
echo ""
echo "Шаг 3: Загрузка переменных окружения..."
if [ -f ".env" ]; then
    set -a  # Автоматически экспортировать все переменные
    source .env
    set +a
    echo "✓ Переменные окружения загружены из .env"
else
    echo "⚠ Файл .env не найден. Продолжаем без него."
    echo "  Для использования ClearML создайте .env с CLEARML_API_ACCESS_KEY и CLEARML_API_SECRET_KEY"
fi

# 4. Запуск обучения с перебором гиперпараметров
echo ""
echo "Шаг 4: Запуск обучения на всем датасете с перебором гиперпараметров..."
echo "=========================================="

# Используем конфиг с включенным перебором гиперпараметров
CONFIG_FILE="configs/train_config_with_tuning.json"

# Проверяем существование конфига
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Ошибка: Конфиг $CONFIG_FILE не найден"
    exit 1
fi

# Запускаем обучение
uv run run --config "$CONFIG_FILE"

echo ""
echo "=========================================="
echo "Обучение завершено!"
echo "=========================================="
