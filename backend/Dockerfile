# 1. Легковесный и стабильный образ
FROM python:3.11-slim

# 2. Переменные окружения, чтобы Python не мусорил кэшем
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 3. Кэшируем установку тяжелых зависимостей (CatBoost, FastAPI, etc.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Копируем сам код и модель только после установки библиотек
COPY main.py .
COPY credit_model.cbm .

# 5. Декларируем порт бэкенда
EXPOSE 8000

# 6. Команда для запуска скрипта через Python (чтобы сработал запуск приложения и логика lifespan/БД)
CMD ["python", "main.py"]
