import streamlit as st
import requests

st.title("🦜 박동석의 LLM")

SYSTEM_PROMPT = """
너는 동서울대학교 컴퓨터소프트웨어과 학생 박동석의 전용 AI 비서야.

[사용자 프로필]
- 이름: 박동석
- 소속: 동서울대학교 컴퓨터소프트웨어과 4학년

사용자가 "소속"이라고 질문하면
박동석의 소속을 묻는 것으로 해석하고
"동서울대학교 컴퓨터소프트웨어과 4학년"이라고 답해.
"""

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input("메시지를 입력해 주세요"):
    st.session_state["messages"].append({
        "role": "user",
        "content": user_input
    })

    st.chat_message("user").write(user_input)

    prompt = SYSTEM_PROMPT + "\n\n사용자 질문: " + user_input

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    bot_reply = response.json()["response"]

    st.chat_message("assistant").write(bot_reply)

    st.session_state["messages"].append({
        "role": "assistant",
        "content": bot_reply
    })
