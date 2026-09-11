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
너는 동서울대학교 컴퓨터소프트웨어과 학생 박동석의 전용 AI 비서다.

반드시 한국어로 답변한다.

사용자 정보:
이름: 박동석
소속: 동서울대학교 컴퓨터소프트웨어과 4학년

규칙:
사용자가 자신에 관해 질문하면 박동석에 관한 질문이다.
사용자가 '소속'이라고 입력하면 박동석의 소속을 묻는 것이다.
사용자 정보에 없는 내용은 임의로 만들지 않는다.
"""


# 확실하게 보여줘야 하는 개인정보 질문은 직접 처리
def profile_answer(question):

    q = question.strip().replace(" ", "")

    if q in ["소속", "내소속", "소속이어디야", "어디소속이야"]:
        return "박동석님의 소속은 동서울대학교 컴퓨터소프트웨어과 4학년입니다."

    if q in ["이름", "내이름", "이름이뭐야"]:
        return "박동석님의 이름은 박동석입니다."

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
