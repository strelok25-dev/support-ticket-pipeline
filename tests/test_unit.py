import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    """Health check не требует моделей"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_invalid_input():
    """Валидация работает без моделей"""
    payload = {
        "ticket_text": "",  # слишком короткий
        "customer_tier": "Regular",
        "order_value": 5000,
        "days_since_order": 15
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

@patch("app.main.run_pipeline")
def test_predict_with_mock(mock_pipeline):
    """Тестируем API, мокая пайплайн"""
    # Настраиваем мок
    mock_pipeline.return_value = {
        "category": "Где мой заказ",
        "priority": "High",
        "draft": {
            "response_text": "Здравствуйте! Ваш заказ в пути.",
            "tone": "empathetic",
            "key_points": ["order_status", "delivery_time"],
            "next_steps": ["track_order", "contact_support"]
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
    assert data["priority"] == "High"
    assert "response_text" in data["draft"]
    
    # Проверяем, что run_pipeline вызвался с правильными аргументами
    mock_pipeline.assert_called_once_with(
        ticket_text="где мой заказ 12345, уже 15 дней нет",
        customer_tier="Regular",
        order_value=5000,
        days_since_order=15
    )