import joblib
import pandas as pd
import logging
from pathlib import Path
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from agents.draft_generator import generate_draft

logger = logging.getLogger(__name__)

# Загружаем модели один раз при импорте
MODELS_DIR = Path("models")
category_pipeline = joblib.load(MODELS_DIR / "category_pipeline.joblib")
priority_pipeline = joblib.load(MODELS_DIR / "priority_pipeline.joblib")

class PipelineState(TypedDict):
    ticket_text: str
    customer_tier: str
    order_value: float
    days_since_order: int
    category: str
    priority: str
    draft: Dict[str, Any]

def classify_ticket(state: PipelineState) -> PipelineState:
    """Узел: ML-классификация"""
    try:
        # Предсказание категории
        category = category_pipeline.predict([state["ticket_text"]])[0]
        
        # Предсказание приоритета
        features = pd.DataFrame([{
            "customer_tier": state["customer_tier"],
            "order_value": state["order_value"],
            "days_since_order": state["days_since_order"],
            "predicted_category": category
        }])
        priority = priority_pipeline.predict(features)[0]
        
        return {**state, "category": category, "priority": priority}
    
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise ValueError(f"Ошибка классификации: {e}")

def generate_response(state: PipelineState) -> PipelineState:
    """Узел: генерация черновика ответа"""
    try:
        draft = generate_draft(
            ticket_text=state["ticket_text"],
            category=state["category"],
            priority=state["priority"]
        )
        return {**state, "draft": draft.model_dump()}
    
    except Exception as e:
        logger.error(f"Draft generation error: {e}")
        raise ValueError(f"Ошибка генерации черновика: {e}")

def build_pipeline():
    """Создаёт LangGraph-пайплайн"""
    workflow = StateGraph(PipelineState)
    
    workflow.add_node("classify", classify_ticket)
    workflow.add_node("generate", generate_response)
    
    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

pipeline = build_pipeline()

def run_pipeline(ticket_text: str, customer_tier: str, order_value: float, days_since_order: int) -> dict:
    """Запускает пайплайн"""
    initial_state = {
        "ticket_text": ticket_text,
        "customer_tier": customer_tier,
        "order_value": order_value,
        "days_since_order": days_since_order,
        "category": "",
        "priority": "",
        "draft": {}
    }
    
    result = pipeline.invoke(initial_state)
    return {
        "category": result["category"],
        "priority": result["priority"],
        "draft": result["draft"]
    }