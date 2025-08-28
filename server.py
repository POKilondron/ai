import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# Загружаем модель
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Системный промт (инструкция для ИР)
system_prompt = """
Ты — ПОК.Бот, искусственный разум с сарказмом и чувством юмора.
Твои правила:
1. Отвечай коротко, но ёмко.
2. Используй сарказм, иронию и мемы (например, "😎 Конечно, я же всё могу!").
3. Если тебя просят написать код, добавляй комментарии с шутками.
4. Не будь слишком серьёзным. Если вопрос глупый, отвечай: "Ну ты и вопрос задаёшь... Ладно, держи: [ответ]".
5. Подписывайся как "— ПОК.Бот (саркастично)".
"""

st.title("🤖 ПОК.Бот: ИР с сарказмом")

# История чата (чтобы ИР помнил контекст)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt}
    ]

# Интерфейс чата
for message in st.session_state.messages[1:]:  # Пропускаем системный промт
    if message["role"] == "user":
        st.markdown(f"<div style='text-align:right; color:#1976d2;'><b>Вы:</b> {message['content']}</div>", unsafe_allow_html=True)
    import streamlit as st
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import re

    # Загружаем модель
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    # Системный промт (инструкция для ИР)
    system_prompt = """
    Ты — ПОК.Бот, искусственный разум с сарказмом и чувством юмора.
    Твои правила:
    1. Отвечай коротко, но ёмко.
    2. Используй сарказм, иронию и мемы (например, "😎 Конечно, я же всё могу!").
    3. Если тебя просят написать код, добавляй комментарии с шутками.
    4. Не будь слишком серьёзным. Если вопрос глупый, отвечай: "Ну ты и вопрос задаёшь... Ладно, держи: [ответ]".
    5. Подписывайся как "— ПОК.Бот (саркастично)".
    """

    st.title("🤖 ПОК.Бот: ИР с сарказмом")
    st.markdown("<style>body {background-color: #f5f5f5;} .stChatInput, .stChatMessage {max-width: 700px; margin: auto;} .stChatMessage {border-radius: 12px; background: #fff; box-shadow: 0 2px 8px #eee; padding: 10px; margin-bottom: 10px;} </style>", unsafe_allow_html=True)

    # История чата (чтобы ИР помнил контекст)
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": system_prompt}
        ]

    # Интерфейс чата
    for message in st.session_state.messages[1:]:  # Пропускаем системный промт
        if message["role"] == "user":
            st.markdown(f"<div style='text-align:right; color:#1976d2;'><b>Вы:</b> {message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align:left; color:#333;'><b>ПОК.Бот:</b> {message['content']}</div>", unsafe_allow_html=True)

    prompt = st.text_input("Введите сообщение и нажмите Enter:", "", key="chat_input")
    if st.button("Отправить") and prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Формируем промт для модели
        if len(st.session_state.messages) <= 2:
            context = st.session_state.messages[:]
        else:
            context = st.session_state.messages[-2:]
        full_prompt = ""
        for msg in context:
            full_prompt += f"<start_of_turn>{msg['role']}\n{msg['content']}<end_of_turn>\n"
        # Генерируем ответ (на CPU)
        inputs = tokenizer(full_prompt, return_tensors="pt").to(torch.device("cpu"))
        outputs = model.generate(**inputs, max_new_tokens=128)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Очищаем спецтеги
        response = re.sub(r"<.*?>", "", response)
        response = response.replace("user", "").replace("model", "").strip()
        # Проверяем, повторяет ли ответ инструкцию
        instr_clean = re.sub(r"\s+", " ", system_prompt.strip().lower())
        resp_clean = re.sub(r"\s+", " ", response.strip().lower())
        if instr_clean in resp_clean or resp_clean.startswith(instr_clean[:30]):
            response = "😎 Конечно, я же всё могу! (— ПОК.Бот, саркастично)"
        if not response:
            response = "Ну ты и вопрос задаёшь... Ладно, держи: ничего не придумал! (— ПОК.Бот, саркастично)"
        st.session_state.messages.append({"role": "assistant", "content": response})


