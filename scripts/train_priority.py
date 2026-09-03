import pandas as pd
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

def train_priority_model(
    data_path: str = "data/raw/tickets.csv",
    category_model_path: str = "models/category_pipeline.joblib"
):
    print("Загрузка данных...")
    df = pd.read_csv(data_path)
    
    # Получаем предсказания модели категории (как фичу)
    print("Загрузка модели категории...")
    category_pipeline = joblib.load(category_model_path)
    df["predicted_category"] = category_pipeline.predict(df["ticket_text"])
    
    # Фичи для модели приоритета
    feature_cols = ["customer_tier", "order_value", "days_since_order", "predicted_category"]
    X = df[feature_cols]
    y = df["priority"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Обучение модели приоритета...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["customer_tier", "predicted_category"]),
            ("num", StandardScaler(), ["order_value", "days_since_order"])
        ]
    )
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    
    print(f"\nМетрики модели приоритета:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (macro): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Сохранение
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/priority_pipeline.joblib")
    
    metadata = {
        "model_type": "priority",
        "classes": list(pipeline.classes_),
        "feature_columns": feature_cols,
        "accuracy": acc,
        "f1_macro": f1,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }
    
    return metadata

if __name__ == "__main__":
    metadata = train_priority_model()
    
    with open("models/priority_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Модель сохранена в models/priority_pipeline.joblib")
    print(f"✅ Метаданные сохранены в models/priority_metadata.json")