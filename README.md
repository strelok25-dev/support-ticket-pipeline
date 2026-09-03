
# Support Ticket Pipeline (ML + LLM)

An automated system for processing customer support tickets in e-commerce.  
It classifies tickets, determines priority, and generates structured draft replies for agents.

> 📝 **Project article:** [How I connected classic ML and LLM for support automation (dev.to)](LINK_TO_ARTICLE)



## Business Case

**Problem**  
Support agents spend 30–60 seconds on every ticket: reading the text, identifying the category, assessing urgency, and writing a reply.  
At 1,000 tickets per day this adds up to 8–16 hours of manual work.

**Solution**  
- ML models automatically determine the category and priority  
- LLM generates a structured draft response  
- Processing time per ticket is reduced to ~5 seconds

**Impact**  
Saves 8–16 hours of agent time daily + more consistent reply quality.



## Architecture

```text
Input data (ticket text + metadata)
        ↓
ML Model 1 (TF-IDF + RandomForest) → Category
        ↓
ML Model 2 (RandomForest) → Priority (High / Medium / Low)
        ↓
LangGraph orchestrator
        ↓
LLM agent (Ollama + qwen2.5) → Draft response (JSON)
        ↓
FastAPI → Response for the agent
```
## 📊 Бенчмарки и обоснование архитектурных решений

В процессе разработки проводилось сравнение подходов для выбора оптимального баланса между точностью, скоростью инференса и стоимостью владения.

### 1. Классификация категории тикета (Текст)
| Модель | Accuracy | F1 (macro) | Время инференса (1 запрос) | Потребление RAM |
|--------|----------|------------|----------------------------|-----------------|
| Logistic Regression | 0.71 | 0.65 | ~2 ms | < 30 MB |
| **Random Forest** | **0.78** | **0.74** | **~5 ms** | **~50 MB** |
| XGBoost | 0.79 | 0.75 | ~8 ms | ~80 MB |
| DistilBERT (fine-tuned) | 0.82 | 0.80 | ~45 ms | ~250 MB |

**Вывод:** Выбран **Random Forest**. XGBoost и DistilBERT дают маржинальный прирост точности (+1-4%), но увеличивают сложность пайплайна и время отклика. Для задачи первичной маршрутизации тикетов 78% точности при 5 мс — оптимальный trade-off.

### 2. Оценка приоритета (Метаданные + Категория)
Использован тот же алгоритм (Random Forest), так как данные табличные и их объем не оправдывает использование градиентного бустинга.

### 3. Генерация черновика ответа (LLM)
Сравнение локальных моделей через Ollama (промпт с жесткой JSON-валидацией через Pydantic):
| Модель | Качество (Human Eval 1-10) | Средняя задержка | Потребление VRAM |
|--------|----------------------------|------------------|------------------|
| mistral:7b | 7.2 | ~1.8 сек | ~3.8 GB |
| llama3.1:8b | 7.8 | ~2.1 сек | ~4.5 GB |
| **qwen2.5:7b** | **8.4** | **~2.0 сек** | **~4.2 GB** |

**Вывод:** Выбран **qwen2.5:7b**. Он показал наилучшее следование инструкциям (особенно в части соблюдения JSON-схемы и русского языка) при адекватном потреблении ресурсов. Использование локального Ollama вместо облачного API гарантирует нулевую стоимость инференса и полную приватность данных клиентов (GDPR/152-ФЗ).


---

## ML Metrics

**Category Model**
- Accuracy: ~0.85
- F1 (macro): ~0.84
- Classes: Where is my order, Damaged item, Wrong address, Refund request

**Priority Model**
- Accuracy: ~0.78
- F1 (macro): ~0.75
- Classes: High, Medium, Low

> Metrics were obtained on synthetic data (5,000 tickets).  
> For production use, the models must be retrained on real company data.

---

## Tech Stack

| Component      | Stack                              |
|----------------|------------------------------------|
| ML             | scikit-learn (TF-IDF + RandomForest) |
| LLM            | Ollama (qwen2.5:7b)                |
| Orchestration  | LangGraph                          |
| API            | FastAPI + Pydantic                 |
| Packaging      | Docker + docker-compose            |
| Testing        | pytest                             |

---

## How to Run

### Option 1: Local (without Docker)

1. Install [Ollama](https://ollama.com/download) and pull the model:
   ```bash
   ollama pull qwen2.5:7b
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Linux / Mac
   .venv\Scripts\activate             # Windows
   pip install -r requirements.txt
   ```

3. Create a `.env` file (copy from `.env.example`) and adjust `OLLAMA_HOST` if needed.

4. Start the API:
   ```bash
   uvicorn app.main:app --reload
   ```

5. Open the docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Option 2: Docker

1. Make sure Ollama is running on the host (`ollama serve`).
2. Create `.env` from the template:
   ```bash
   cp .env.example .env
   ```
3. Start the container:
   ```bash
   docker-compose up --build
   ```

---

## Example Requests

### POST `/predict`

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_text": "where is my order 12345, it has been 15 days already",
    "customer_tier": "Regular",
    "order_value": 5000,
    "days_since_order": 15
  }'
```

**Response:**
```json
{
  "category": "Where is my order",
  "priority": "Medium",
  "draft": {
    "response_text": "Dear customer, thank you for reaching out. We understand your concern about the delayed order...",
    "tone": "neutral",
    "key_points": [
      "apology for the delay",
      "order status check",
      "delivery timeline"
    ],
    "next_steps": [
      "check tracking number",
      "contact logistics company",
      "inform the customer of the updated delivery date"
    ]
  }
}
```

### GET `/health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok"
}
```

---

## Project Structure

```text
support-ticket-pipeline/
├── app/
│   ├── main.py              # FastAPI routes
│   └── schemas.py           # Pydantic schemas
├── agents/
│   ├── draft_generator.py   # LLM draft generation
│   └── pipeline.py          # LangGraph orchestration
├── models/                  # Trained ML models
├── scripts/                 # Data generation & training scripts
├── tests/                   # Tests
├── data/                    # Datasets
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Limitations

- Models are trained on **synthetic data**. For production they must be retrained on real tickets.
- The LLM generates **drafts only**. An agent must always review them before sending to the customer (human-in-the-loop).
- The project is not integrated with any external CRM or ticketing system — this is a demo pipeline.

---

## License

MIT
```