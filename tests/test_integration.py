import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.mark.integration
def test_predict_endpoint():
    """Интеграционный тест: требует моделей и Ollama"""
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
    assert data["category"] in ["Где мой заказ", "Поврежденный товар", "Ошибка адреса", "Запрос на возврат"]
    assert data["priority"] in ["High", "Medium", "Low"]