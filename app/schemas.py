from pydantic import BaseModel, Field
from typing import List, Literal

class TicketInput(BaseModel):
    ticket_text: str = Field(..., min_length=10, description="Текст обращения клиента")
    customer_tier: Literal["Regular", "VIP"] = Field(..., description="Статус клиента")
    order_value: float = Field(..., gt=0, description="Сумма заказа")
    days_since_order: int = Field(..., ge=1, description="Дней с момента заказа")

class DraftResponse(BaseModel):
    response_text: str = Field(..., description="Текст ответа клиенту")
    tone: Literal["empathetic", "neutral", "apologetic"] = Field(..., description="Тон ответа")
    key_points: List[str] = Field(..., description="Ключевые моменты ответа")
    next_steps: List[str] = Field(..., description="Следующие шаги для оператора")

class PredictResponse(BaseModel):
    category: str = Field(..., description="Категория проблемы")
    priority: str = Field(..., description="Приоритет (High/Medium/Low)")
    draft: DraftResponse = Field(..., description="Черновик ответа LLM")