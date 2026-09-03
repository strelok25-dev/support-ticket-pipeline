import gradio as gr
import requests

# Адрес твоего запущенного FastAPI приложения
API_URL = "http://127.0.0.1:8000/predict"

def process_ticket(ticket_text, customer_tier, order_value, days_since_order):
    """Отправляет запрос в FastAPI и возвращает результат"""
    try:
        payload = {
            "ticket_text": ticket_text,
            "customer_tier": customer_tier,
            "order_value": float(order_value),
            "days_since_order": int(days_since_order)
        }
        
        # Делаем запрос к твоему API с таймаутом 30 секунд
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        category = data.get("category", "Не определено")
        priority = data.get("priority", "Не определено")
        draft = data.get("draft", {}).get("response_text", "Черновик не сгенерирован")
        
        return category, priority, draft
        
    except requests.exceptions.ConnectionError:
        return "Ошибка", "Ошибка", f"API недоступен по адресу {API_URL}. Убедись, что запустил uvicorn app.main:app"
    except Exception as e:
        return "Ошибка", "Ошибка", str(e)

# Настройка интерфейса Gradio
demo = gr.Interface(
    fn=process_ticket,
    inputs=[
        gr.Textbox(label="Текст тикета", lines=3, placeholder="Например: Где мой заказ 12345, уже 15 дней нет"),
        gr.Radio(["Regular", "VIP"], label="Тип клиента", value="Regular"),
        gr.Number(label="Сумма заказа (₽)", value=5000),
        gr.Number(label="Дней с момента заказа", value=15)
    ],
    outputs=[
        gr.Textbox(label="Категория", interactive=False),
        gr.Textbox(label="Приоритет", interactive=False),
        gr.Textbox(label="Черновик ответа от AI", lines=6, interactive=False)
    ],
    title=" AI Support Assistant",
    description="Автоматическая классификация тикетов и генерация черновиков ответов (ML + LLM)",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()