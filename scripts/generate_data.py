import pandas as pd
import numpy as np
import os

# Шаблоны текстов для каждой категории
CATEGORY_TEMPLATES = {
    "Где мой заказ": [
        "где мой заказ {order_id}",
        "когда будет доставка заказа {order_id}",
        "заказ {order_id} не приходит уже {days} дней",
        "трек-номер не работает, где посылка",
        "курьер не звонит, заказ {order_id}",
        "прошло {days} дней, а заказа нет",
        "где моя посылка",
        "статус заказа {order_id} не меняется",
    ],
    "Поврежденный товар": [
        "получил товар, он разбит",
        "пришла разбитая упаковка, товар поврежден",
        "товар пришел в ужасном состоянии",
        "коробка помята, содержимое сломано",
        "получил бракованный товар",
        "товар с дефектом, хочу замену",
        "пришло не то, что заказывал, плюс повреждено",
    ],
    "Ошибка адреса": [
        "я указал неверный адрес, можно изменить",
        "ошибка в адресе доставки, помогите",
        "нужно поменять адрес для заказа {order_id}",
        "я ошибся с адресом, как исправить",
        "доставка не по тому адресу",
        "адрес неверный, можно перенаправить",
    ],
    "Запрос на возврат": [
        "хочу вернуть товар",
        "как оформить возврат",
        "верните деньги за заказ {order_id}",
        "товар не подошел, хочу вернуть",
        "оформите возврат пожалуйста",
        "хочу отменить заказ и получить возврат",
        "возврат средств, заказ {order_id}",
    ],
}

def generate_ticket_text(category: str, rng: np.random.Generator) -> str:
    """Генерирует текст тикета с шумом"""
    templates = CATEGORY_TEMPLATES[category]
    text = rng.choice(templates)
    
    # Подстановка плейсхолдеров
    text = text.format(
        order_id=rng.integers(10000, 99999),
        days=rng.integers(5, 30)
    )
    
    # Шум: случайные слова (10% шанс)
    if rng.random() < 0.1:
        noise_words = ["срочно", "быстро", "помогите", "пожалуйста", "!!!"]
        text += " " + rng.choice(noise_words)
    
    # Шум: разная длина (5% шанс добавить длинное описание)
    if rng.random() < 0.05:
        text += ". Очень жду ответа, ситуация неприятная, прошу разобраться как можно скорее"
    
    return text

def generate_priority(row: pd.Series, rng: np.random.Generator) -> str:
    """Нелинейные правила для приоритета (чтобы модель училась)"""
    # High: VIP + дорогой заказ + давно ждёт ИЛИ поврежденный товар
    if row["customer_tier"] == "VIP" and row["order_value"] > 10000 and row["days_since_order"] > 7:
        return "High"
    if row["category"] == "Поврежденный товар" and row["days_since_order"] > 5:
        return "High"
    
    # Medium: средний срок + обычная категория ИЛИ VIP
    if row["days_since_order"] > 10 and row["category"] in ["Где мой заказ", "Ошибка адреса"]:
        return "Medium"
    if row["customer_tier"] == "VIP":
        return "Medium"
    
    # Low: остальное
    return "Low"

def generate_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    
    categories = list(CATEGORY_TEMPLATES.keys())
    data = []
    
    for _ in range(n_samples):
        category = rng.choice(categories)
        ticket_text = generate_ticket_text(category, rng)
        
        customer_tier = rng.choice(["Regular", "VIP"], p=[0.8, 0.2])
        order_value = rng.lognormal(mean=8.5, sigma=1.0)
        order_value = round(np.clip(order_value, 500, 50000), 2)
        days_since_order = rng.integers(1, 30)
        
        row = pd.Series({
            "ticket_text": ticket_text,
            "customer_tier": customer_tier,
            "order_value": order_value,
            "days_since_order": days_since_order,
            "category": category,
        })
        
        priority = generate_priority(row, rng)
        row["priority"] = priority
        data.append(row)
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    print("Генерация синтетических данных...")
    df = generate_dataset(n_samples=5000)
    
    os.makedirs("data/raw", exist_ok=True)
    filepath = "data/raw/tickets.csv"
    df.to_csv(filepath, index=False)
    
    print(f"✅ Данные сохранены в {filepath}")
    print(f"Размер: {len(df)} строк")
    print("\nРаспределение категорий:")
    print(df["category"].value_counts())
    print("\nРаспределение приоритетов:")
    print(df["priority"].value_counts())
    print("\nПример:")
    print(df.head(3).to_string())