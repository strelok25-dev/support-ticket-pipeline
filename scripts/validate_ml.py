import joblib
import json
import pandas as pd

def validate_models():
    print("Валидация ML-моделей...\n")
    
    # Загрузка моделей
    category_pipeline = joblib.load("models/category_pipeline.joblib")
    priority_pipeline = joblib.load("models/priority_pipeline.joblib")
    
    with open("models/category_metadata.json", "r", encoding="utf-8") as f:
        category_meta = json.load(f)
    
    with open("models/priority_metadata.json", "r", encoding="utf-8") as f:
        priority_meta = json.load(f)
    
    print(f"✅ Модель категории загружена")
    print(f"   Классы: {category_meta['classes']}")
    print(f"   Accuracy: {category_meta['accuracy']:.4f}, F1: {category_meta['f1_macro']:.4f}")
    
    print(f"\n✅ Модель приоритета загружена")
    print(f"   Классы: {priority_meta['classes']}")
    print(f"   Accuracy: {priority_meta['accuracy']:.4f}, F1: {priority_meta['f1_macro']:.4f}")
    
    # Тестовые примеры
    test_tickets = [
        "где мой заказ 12345, уже 15 дней нет",
        "товар пришел разбитый, хочу замену",
        "хочу вернуть деньги",
    ]
    
    print("\nТестовые предсказания:")
    for i, text in enumerate(test_tickets, 1):
        category = category_pipeline.predict([text])[0]
        
        # Для приоритета нужны метаданные (имитируем)
        test_row = pd.DataFrame([{
            "customer_tier": "Regular",
            "order_value": 5000,
            "days_since_order": 10,
            "predicted_category": category
        }])
        
        priority = priority_pipeline.predict(test_row)[0]
        
        print(f"\n{i}. Текст: '{text}'")
        print(f"   Категория: {category}")
        print(f"   Приоритет: {priority}")

if __name__ == "__main__":
    validate_models()