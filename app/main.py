import logging
import time
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import TicketInput, PredictResponse
from agents.pipeline import run_pipeline

# 1. ПРОДАКШН-ЛОГИРОВАНИЕ
# Добавляем время, уровень и четкий формат. 
# В будущем это легко перенаправить в JSON для сбора в Grafana/Prometheus.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ticket_pipeline")

app = FastAPI(title="Support Ticket Pipeline", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Перед релизом заменить на конкретный домен Streamlit
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. MIDDLEWARE ДЛЯ ГЛОБАЛЬНОГО ТРЕКИНГА
# Перехватывает каждый запрос, вешает на него уникальный ID и замеряет общее время
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]  # Короткий ID для читаемости в консоли
    start_time = time.perf_counter()    # Более точный таймер, чем time.time()
    
    # Сохраняем ID в объекте запроса, чтобы достать его внутри эндпоинта
    request.state.request_id = request_id
    
    logger.info(f"[{request_id}] START {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        process_time = (time.perf_counter() - start_time) * 1000
        logger.info(f"[{request_id}] COMPLETE {request.method} {request.url.path} | Status: {response.status_code} | Latency: {process_time:.2f}ms")
        return response
    except Exception as e:
        process_time = (time.perf_counter() - start_time) * 1000
        # exc_info=True добавит полный traceback в логи. Это твое спасение при дебаге.
        logger.error(f"[{request_id}] FAILED {request.method} {request.url.path} | Latency: {process_time:.2f}ms | Error: {str(e)}", exc_info=True)
        raise

# 3. ОСНОВНОЙ ЭНДПОИНТ
@app.post("/predict", response_model=PredictResponse)
async def predict(ticket: TicketInput, request: Request):
    # Достаем ID, созданный middleware
    request_id = request.state.request_id
    
    # Логируем входные данные, но ОБРЕЗАЕМ текст, чтобы не спамить в консоль километрами текста
    text_preview = ticket.ticket_text[:50].replace('\n', ' ') + "..." if len(ticket.ticket_text) > 50 else ticket.ticket_text
    logger.info(f"[{request_id}] Processing: '{text_preview}' | Tier: {ticket.customer_tier}")
    
    try:
        start_time = time.perf_counter()
        
        # Вызов твоего пайплайна
        result = run_pipeline(
            ticket_text=ticket.ticket_text,
            customer_tier=ticket.customer_tier,
            order_value=ticket.order_value,
            days_since_order=ticket.days_since_order
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Успешное выполнение с ключевыми бизнес-метриками
        logger.info(
            f"[{request_id}] SUCCESS | "
            f"Category: {result.get('category')} | "
            f"Priority: {result.get('priority')} | "
            f"Latency: {latency_ms:.2f}ms"
        )
        
        return PredictResponse(**result)
        
    except ValueError as e:
        # Ожидаемые ошибки (например, невалидные данные от клиента)
        logger.warning(f"[{request_id}] VALIDATION ERROR: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        # Критические ошибки (упал Ollama, ошибка в ML-модели и т.д.)
        logger.error(f"[{request_id}] CRITICAL PIPELINE ERROR: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка обработки тикета")

@app.get("/health")
async def health():
    """Проверка здоровья сервиса"""
    return {"status": "ok", "service": "ticket_pipeline"}