import streamlit as st
from groq import Groq

st.set_page_config(page_title="나만의 맞춤형 LLM 챗봇", page_icon="🦜")
st.title("🦜 Chat Custom LLM")

# Groq 클라이언트 초기화
if "GROQ_API_KEY" not in st.secrets:
    st.error("Streamlit Secrets에 GROQ_API_KEY가 설정되지 않았습니다.")
    st.stop()

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# 개인정보 및 한국어 출력 규칙 프롬프트
SYSTEM_PROMPT = """
너는 동서울대학교 컴퓨터소프트웨어과 학생 박동석의 전용 AI 비서야.
반드시 모든 답변을 자연스럽고 친절한 한국어로만 작성해. 영어를 절대 섞지 마.

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

    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in st.session_state["messages"]:
        api_messages.append({"role": m["role"], "content": m["content"]})

    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            try:
                # 1순위: llama-3.3-70b-versatile, 2순위: llama-3.1-8b-instant, 3순위: gemma2-9b-it
                models_to_try = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"]
                bot_reply = None
                last_error = None

                for target_model in models_to_try:
                    try:
                        chat_completion = client.chat.completions.create(
                            model=target_model,
                            messages=api_messages,
                            temperature=0.2,
                        )
                        bot_reply = chat_completion.choices[0].message.content
                        break
                    except Exception as e:
                        last_error = e
                        continue

                if bot_reply:
                    st.write(bot_reply)
                    st.session_state["messages"].append({"role": "assistant", "content": bot_reply})
                else:
                    st.error(f"모든 모델 호출 실패: {last_error}")

            except Exception as err:
                st.error(f"전체 오류 내용: {err}")
