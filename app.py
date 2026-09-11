import streamlit as st


import requests



st.set_page_config(page_title="나만의 맞춤형 LLM 챗봇", page_icon="🦜")
st.title("🦜 박동석의 LLM")

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
- 전화번호 : 010-5686-6633
- 학번 : 2670052
- 학교 정보 : 장현초등학교,광동중학교,광동고등학교,동서울대학교

[가족 정보]
- 구성원: 총 4명 (아버지 박태일, 어머니 박정숙, 형 박동기, 본인 박동석)
- 가족 전화번호 : 아버지 전화번호 : 010-6213-0626, 어머니 전화번호:010-5686-6633
- 주의: 동생은 없음 (본인이 막내)
- 반려견(강아지): 박노랑 (1살)
- 구성원의 나이: 아버지는 55세, 어머니는 57세, 형은 31살
- 사는 곳 : 경기도 남양주 진접읍 장현리
- 아빠와 엄마는 각각 아버지와 어머니를 뜻해
[답변 원칙]
프로필과 가족 정보에 명시된 사실에 기반해서만 답변하고, 없는 정보는 지어내지 말고 정정해 줘.
예시) 우리 아빠 이름이 뭐야 -> 박동석님의 아빠이름은 박태일입니다처럼 나오게 해줘 하나의 예시일뿐이야
나의 엄마 이름이 뭐야 -> 박동석님의 엄마이름은 박정숙입니다처럼 나오게 해줘 하나의 예시일뿐이야
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
                # Groq 계정에서 현재 사용 가능한 실제 모델 목록 조회
                available_models = [m.id for m in client.models.list().data if "whisper" not in m.id]
                
                # llama 8b 계열 우선 선택, 없으면 사용 가능한 첫 번째 텍스트 모델 선택
                target_model = None
                for candidate in ["llama-3.1-8b-instant", "llama3-70b-8192", "mixtral-8x7b-32768"]:
                    if candidate in available_models:
                        target_model = candidate
                        break
                if not target_model and available_models:
                    target_model = available_models[0]

                chat_completion = client.chat.completions.create(
                    model=target_model,
                    messages=api_messages,
                    temperature=0.2,
                )
                bot_reply = chat_completion.choices[0].message.content
                st.write(bot_reply)
                st.session_state["messages"].append({"role": "assistant", "content": bot_reply})

            except Exception as err:
                st.error(f"오류 발생: {err}")
