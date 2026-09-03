import pandas as pd
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

def train_category_model(data_path: str = "data/raw/tickets.csv"):
    print("Загрузка данных...")
    df = pd.read_csv(data_path)
    
    X = df["ticket_text"]
    y = df["category"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Обучение модели категории...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    
    print(f"\nМетрики модели категории:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (macro): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Сохранение
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/category_pipeline.joblib")
    
    metadata = {
        "model_type": "category",
        "classes": list(pipeline.classes_),
        "accuracy": acc,
        "f1_macro": f1,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }
    
    return metadata

if __name__ == "__main__":
    metadata = train_category_model()
    
    with open("models/category_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Модель сохранена в models/category_pipeline.joblib")
    print(f"✅ Метаданные сохранены в models/category_metadata.json")