import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    # Проверяем точное совпадение с тем, что возвращает наш продакшн-код
    assert response.json() == {"status": "ok", "service": "ticket_pipeline"}

def test_invalid_input():
    payload = {
        "ticket_text": "",  
        "customer_tier": "Regular",
        "order_value": 5000,
        "days_since_order": 15
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

@patch("app.main.run_pipeline")
def test_predict_with_mock(mock_pipeline):
    # Настраиваем фейковый ответ, который вернет "замоканная" функция
    mock_pipeline.return_value = {
        "category": "Где мой заказ",
        "priority": "High",
        "draft": {
            "response_text": "Здравствуйте! Ваш заказ в пути.",
            "tone": "empathetic",
            "key_points": ["order_status"],
            "next_steps": ["track_order"]
        }
    }
    
    payload = {
        "ticket_text": "где мой заказ 12345, уже 15 дней нет",
        "customer_tier": "Regular",
        "order_value": 5000,
        "days_since_order": 15
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["category"] == "Где мой заказ"
    
    # Проверяем, что функция была вызвана 1 раз с нужными аргументами
    mock_pipeline.assert_called_once()