import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import TicketInput, PredictResponse
from agents.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Support Ticket Pipeline", version="1.0.0")

# CORS для Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # потом можно ограничить
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=PredictResponse)
async def predict(ticket: TicketInput):
    """Основной эндпоинт: классификация + генерация ответа"""
    try:
        result = run_pipeline(
            ticket_text=ticket.ticket_text,
            customer_tier=ticket.customer_tier,
            order_value=ticket.order_value,
            days_since_order=ticket.days_since_order
        )
        return PredictResponse(**result)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

@app.get("/health")
async def health():
    """Проверка здоровья сервиса"""
    return {"status": "ok"}