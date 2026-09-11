import streamlit as st
import requests

st.set_page_config(
    page_title="나만의 맞춤형 LLM 챗봇",
    page_icon="🦜"
)

st.title("🦜 박동석의 LLM")

SYSTEM_PROMPT = """
너는 동서울대학교 컴퓨터소프트웨어과 학생 박동석의 전용 AI 비서야.

반드시 모든 답변을 자연스럽고 친절한 한국어로만 작성해.
영어를 최대한 사용하지 마.

[사용자 프로필]
- 이름: 박동석
- 소속: 동서울대학교 컴퓨터소프트웨어과 4학년

[답변 원칙]
- 프로필에 있는 정보를 기준으로 답변해.
- 없는 정보는 지어내지 마.
- 사용자가 "소속"이라고 입력하면
  박동석의 소속을 묻는 것으로 이해하고
  "박동석님의 소속은 동서울대학교 컴퓨터소프트웨어과 4학년입니다."
  라고 답변해.
- AI 회사나 Ollama, Llama의 소속을 답하지 마.
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

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    for msg in st.session_state["messages"]:
        messages.append(msg)

    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):

            try:
                response = requests.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": "llama3",
                        "messages": messages,
                        "stream": False
                    },
                    timeout=120
                )

                response.raise_for_status()

                result = response.json()

                bot_reply = result["message"]["content"]

                st.write(bot_reply)

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": bot_reply
                })

            except Exception as err:
                st.error(f"Ollama 연결 오류: {err}")
