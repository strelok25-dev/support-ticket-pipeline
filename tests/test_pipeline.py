import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_endpoint():
    payload = {
        "ticket_text": "где мой заказ 12345, уже 15 дней нет",
        "customer_tier": "Regular",
        "order_value": 5000,
        "days_since_order": 15
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "category" in data
    assert "priority" in data
    assert "draft" in data
    assert "response_text" in data["draft"]
    assert "tone" in data["draft"]
    assert "key_points" in data["draft"]
    assert "next_steps" in data["draft"]
    
    # Проверка значений
    assert data["category"] in ["Где мой заказ", "Поврежденный товар", "Ошибка адреса", "Запрос на возврат"]
    assert data["priority"] in ["High", "Medium", "Low"]
    assert data["draft"]["tone"] in ["empathetic", "neutral", "apologetic"]

def test_invalid_input():
    payload = {
        "ticket_text": "",  # слишком короткий (min_length=10 в схеме)
        "customer_tier": "Regular",
        "order_value": 5000,
        "days_since_order": 15
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # validation error

# Примечание: эти тесты — интеграционные. Для их работы нужны:
# 1. Файлы моделей в models/
# 2. Запущенная Ollama с моделью qwen2.5:7b
# Если хочешь unit-тесты — нужно мокать run_pipeline