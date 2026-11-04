import sys
from unittest.mock import MagicMock

# 1. Изолируем внешние системы (создаем пустые макеты-заглушки)
sys.modules['mlflow'] = MagicMock()
sys.modules['psycopg2'] = MagicMock()

# 2. Импортируем инструменты тестирования и твое FastAPI приложение
from fastapi.testclient import TestClient
from main import app

# Создаем тестового клиента
client = TestClient(app)

def test_read_root():
    # Отправляем проверочный GET-запрос на корневой эндпоинт
    response = client.get("/")
    
    # Проверяем, что статус ответа равен 200 (ОК)
    assert response.status_code == 200
    
    # Проверяем текст ответа
    assert response.json() == {'message': 'Сервер кредитного скоринга запущен и готов к работе'}

def test_predict_default():
    # 1. Готовим тестовые входные данные для модели
    test_data = {
        "RevolvingUtilizationOfUnsecuredLines": 0.3,
        "NumberOfTimes90DaysLate": 0,
        "age": 45,
        "NumberOfTime30_59DaysPastDueNotWorse": 1,
        "DebtRatio": 0.2,
        "NumberOfOpenCreditLinesAndLoans": 5
    }
    
    # 2. Отправляем POST-запрос с JSON-телом на эндпоинт /predict
    response = client.post("/predict", json=test_data)
    
    # 3. Разбираем работу pytest assert:
    # Проверяем, что сервер успешно обработал запрос (код 200)
    assert response.status_code == 200
    
    json_data = response.json()
    
    # Проверяем наличие ключа 'probability'
    assert 'probability' in json_data
    
    # Проверяем, что вероятность находится в границах от 0.0 до 1.0
    assert 0.0 <= json_data['probability'] <= 1.0
    
    # Проверяем корректность категории риска
    assert json_data['risk_category'] in ['Низкий', 'Высокий']
    
    # Проверяем, что сервер вернул SHAP-объяснения
    assert 'shap_features' in json_data
    assert 'age' in json_data['shap_features']

def test_global_shap():
    # 1. Отправляем GET-запрос на эндпоинт глобальной важности признаков
    response = client.get("/global_shap")
    
    # 2. Проверяем, что эндпоинт доступен (код 200)
    assert response.status_code == 200
    
    json_data = response.json()
    
    # 3. Проверяем, что в ответе вернулись веса для ключевых признаков модели
    assert "age" in json_data
    assert "DebtRatio" in json_data
    
    # Проверяем, что значение веса — это число (тип float или int), а не пустая строка
    assert isinstance(json_data["age"], (int, float))
