
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



## 🏗 Architecture

```mermaid
flowchart LR
    A[Client / UI] -->|HTTP POST| B(FastAPI)
    B --> C{LangGraph Orchestrator}
    C -->|1. Ticket Text| D[ML: Category RF]
    D -->|Category| E[ML: Priority RF]
    E -->|Metadata + Category| F[Ollama: Qwen2.5]
    F -->|JSON Draft| G[Agent Response]
    
    style B fill:#2ecc71,stroke:#333,stroke-width:2px,color:#fff
    style C fill:#3498db,stroke:#333,stroke-width:2px,color:#fff
    style F fill:#e74c3c,stroke:#333,stroke-width:2px,color:#fff
---

## Benchmarking and Architectural Decisions

During development, several approaches were compared to find the optimal balance between accuracy, inference speed, and total cost of ownership.

### 1. Ticket Category Classification (Text)

| Model | Accuracy | F1 (macro) | Inference Time (1 request) | RAM Usage |
|-------|----------|------------|----------------------------|-----------|
| Logistic Regression | 0.71 | 0.65 | ~2 ms | < 30 MB |
| Random Forest | 0.78 | 0.74 | ~5 ms | ~50 MB |
| XGBoost | 0.79 | 0.75 | ~8 ms | ~80 MB |
| DistilBERT (fine-tuned) | 0.82 | 0.80 | ~45 ms | ~250 MB |

**Decision:** Random Forest was selected. XGBoost and DistilBERT provide marginal accuracy improvements (+1-4%) but increase pipeline complexity and response time. For the primary routing task, 78% accuracy at 5 ms is the optimal trade-off.

### 2. Priority Assessment (Metadata + Category)

The same algorithm (Random Forest) was used because the data is tabular and its volume does not justify the use of gradient boosting.

### 3. Draft Response Generation (LLM)

Local models were compared via Ollama with a prompt enforcing strict JSON validation through Pydantic:

| Model | Quality (Human Eval 1-10) | Average Latency | VRAM Usage |
|-------|----------------------------|-----------------|------------|
| mistral:7b | 7.2 | ~1.8 sec | ~3.8 GB |
| llama3.1:8b | 7.8 | ~2.1 sec | ~4.5 GB |
| qwen2.5:7b | 8.4 | ~2.0 sec | ~4.2 GB |

**Decision:** qwen2.5:7b was selected. It demonstrated the best instruction following, especially in adhering to the JSON schema and Russian language, while maintaining reasonable resource consumption. Using local Ollama instead of a cloud API ensures zero inference cost and full data privacy (GDPR / local regulations).

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

```markdown
### Option 3: Interactive UI (Gradio)

For a visual demonstration without using `curl` or Swagger:

1. Ensure the API is running (`uvicorn app.main:app --reload`).
2. Run the demo script in a separate terminal:
   ```bash
   python demo.py


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