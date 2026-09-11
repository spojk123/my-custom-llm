import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(
    page_title="나만의 맞춤형 LLM 챗봇",
    page_icon="🦜"
)

st.title("🦜 박동석의 LLM")

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"


# 모델은 최초 한 번만 로드
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    model.eval()

    return tokenizer, model


with st.spinner("LLM 모델을 불러오는 중입니다..."):
    tokenizer, model = load_model()


SYSTEM_PROMPT = """
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
[답변 원칙] 프로필과 가족 정보에 명시된 사실에 기반해서만 답변하고, 없는 정보는 지어내지 말고 정정해 줘. 
예시) 우리 아빠 이름이 뭐야 -> 박동석님의 아빠이름은 박태일입니다처럼 나오게 해줘 하나의 예시일뿐이야
나의 엄마 이름이 뭐야 -> 박동석님의 엄마이름은 박정숙입니다처럼 나오게 해줘 하나의 예시일뿐이야
"""


# 확실하게 보여줘야 하는 개인정보 질문은 직접 처리
def profile_answer(question):
    q = question.strip().replace(" ", "").replace("?", "")

    # 이름 관련 질문
    if "이름" in q:
        return "박동석님의 이름은 박동석입니다."

    # 소속 관련 질문
    if "소속" in q:
        return "박동석님의 소속은 동서울대학교 컴퓨터소프트웨어과 4학년입니다."

    # 학교 관련 질문
    if "학교" in q or "대학교" in q:
        return "박동석님은 동서울대학교 컴퓨터소프트웨어과 4학년입니다."

    # 학과 관련 질문
    if "학과" in q or "전공" in q:
        return "박동석님의 학과는 동서울대학교 컴퓨터소프트웨어과입니다."

    return None

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


if prompt := st.chat_input("메시지를 입력해 주세요"):

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.write(prompt)


    # 프로필 관련 질문인지 먼저 확인
    fixed_answer = profile_answer(prompt)


    if fixed_answer:

        answer = fixed_answer

    else:

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        # 최근 대화만 전달
        for msg in st.session_state.messages[-4:]:
            messages.append(msg)


        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


        inputs = tokenizer(
            text,
            return_tensors="pt"
        )


        with st.spinner("답변 생성 중..."):

            with torch.no_grad():

                output = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.3,
                    repetition_penalty=1.1
                )


        generated = output[0][inputs["input_ids"].shape[1]:]

        answer = tokenizer.decode(
            generated,
            skip_special_tokens=True
        ).strip()


    with st.chat_message("assistant"):
        st.write(answer)


    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
