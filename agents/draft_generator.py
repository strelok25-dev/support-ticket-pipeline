import ollama
import json
import os
import logging
from app.schemas import DraftResponse

logger = logging.getLogger(__name__)

# Читаем хост Ollama из переменной окружения (по умолчанию для локального запуска)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
ollama_client = ollama.Client(host=OLLAMA_HOST)

PROMPT_TEMPLATE = """Сгенерируй ответ клиенту на основе:

Текст обращения: {ticket_text}
Категория проблемы: {category}
Приоритет: {priority}

Требования к ответу:
- Если приоритет High: тон empathetic/apologetic, предложи компенсацию или срочное решение
- Если приоритет Medium: тон neutral, вежливый, стандартное решение
- Если приоритет Low: тон neutral, краткий, ссылка на FAQ или трек-номер

Верни JSON в формате:
{{
  "response_text": "текст ответа клиенту",
  "tone": "empathetic|neutral|apologetic",
  "key_points": ["ключевой момент 1", "момент 2"],
  "next_steps": ["шаг 1 для оператора", "шаг 2"]
}}

Ответ должен быть строго в указанном JSON-формате. Никакого текста до или после JSON."""

def generate_draft(ticket_text: str, category: str, priority: str, model: str = None) -> DraftResponse:
    """Генерирует черновик ответа через Ollama"""
    # Модель из переменной окружения или дефолт
    if model is None:
        model = os.getenv("LLM_MODEL", "qwen2.5:7b")
    
    prompt = PROMPT_TEMPLATE.format(
        ticket_text=ticket_text,
        category=category,
        priority=priority
    )
    
    try:
        response = ollama_client.chat(
            model=model,
            messages=[
                {"role": "system", "content": "Ты — опытный оператор поддержки интернет-магазина. Отвечай только валидным JSON."},
                {"role": "user", "content": prompt}
            ],
            format="json",
            options={"temperature": 0.3}
        )
        
        content = response["message"]["content"]
        data = json.loads(content)
        return DraftResponse(**data)
    
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        raise ValueError(f"Не удалось распарсить ответ LLM: {e}")
    
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        raise ValueError(f"Ошибка генерации ответа: {e}")