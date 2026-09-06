import streamlit as st
from groq import Groq

st.set_page_config(page_title="나만의 맞춤형 LLM 챗봇", page_icon="🦜")
st.title("🦜 Chat Custom LLM")

# Groq 클라이언트 초기화
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# 개인정보 및 한국어 출력 규칙 프롬프트
SYSTEM_PROMPT = """
너는 동서울대학교 컴퓨터소프트웨어과 학생 박동석의 전용 AI 비서야.
반드시 모든 답변을 자연스럽고 친절한 한국어로만 작성해. 영어를 섞지 마.

[사용자 프로필]
- 이름: 박동석
- 소속: 동서울대학교 컴퓨터소프트웨어과 4학년

[가족 정보]
- 구성원: 총 4명 (아버지 박태일, 어머니 박정숙, 형 박동기, 본인 박동석)
- 주의: 동생은 없음 (본인이 막내)
- 반려견: 박노랑 (1살)

[답변 원칙]
프로필과 가족 정보에 있는 사실에 기반해서만 답변하고, 없는 정보는 지어내지 말고 정정해 줘.
"""

# 세션 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 이전 대화 내용 출력
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# 사용자 입력 처리
if user_input := st.chat_input("메시지를 입력해 주세요"):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Groq API 전달용 메시지 리스트
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in st.session_state["messages"]:
        api_messages.append({"role": m["role"], "content": m["content"]})

    # 최신 Llama 3.3 70B 모델 호출
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            chat_completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=api_messages
            )
            bot_reply = chat_completion.choices[0].message.content
            st.write(bot_reply)

    st.session_state["messages"].append({"role": "assistant", "content": bot_reply})
