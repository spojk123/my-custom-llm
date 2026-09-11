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
    q = question.strip().replace(" ", "").replace("?", "")

    # =========================
    # 가족 정보 - 구체적인 질문부터 먼저 검사
    # =========================

    # 아버지
    if ("아버지" in q or "아빠" in q) and "이름" in q:
        return "박동석님의 아버지 이름은 박태일입니다."

    # 어머니
    if ("어머니" in q or "엄마" in q) and "이름" in q:
        return "박동석님의 어머니 이름은 박정숙입니다."

    # 형
    if "형" in q and "이름" in q:
        return "박동석님의 형 이름은 박동기입니다."

    # 반려견
    if ("강아지" in q or "반려견" in q) and "이름" in q:
        return "박동석님의 반려견 이름은 박노랑입니다."

    # 가족 구성원
    if "가족" in q and ("구성" in q or "몇명" in q):
        return "박동석님의 가족은 아버지 박태일, 어머니 박정숙, 형 박동기, 본인 박동석으로 총 4명입니다."

    # =========================
    # 나이
    # =========================

    if ("아버지" in q or "아빠" in q) and "나이" in q:
        return "박동석님의 아버지는 55세입니다."

    if ("어머니" in q or "엄마" in q) and "나이" in q:
        return "박동석님의 어머니는 57세입니다."

    if "형" in q and "나이" in q:
        return "박동석님의 형은 31살입니다."

    # =========================
    # 소속 / 학교
    # =========================

    if "소속" in q:
        return "박동석님의 소속은 동서울대학교 컴퓨터소프트웨어과 4학년입니다."

    if "학과" in q or "전공" in q:
        return "박동석님의 학과는 동서울대학교 컴퓨터소프트웨어과입니다."

    if "학교" in q or "대학교" in q:
        return "박동석님은 동서울대학교 컴퓨터소프트웨어과 4학년입니다."

    # =========================
    # 마지막에 본인 이름 검사
    # =========================

    if "이름" in q:
        return "박동석님의 이름은 박동석입니다."

    return None
