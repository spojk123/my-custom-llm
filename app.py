import streamlit as st
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# Streamlit 설정
# =========================================================

st.set_page_config(
    page_title="나만의 맞춤형 LLM 챗봇",
    page_icon="🦜"
)

st.title("🦜 박동석의 LLM")


# =========================================================
# 오픈소스 LLM 설정
# =========================================================

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"


@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    model.eval()

    return tokenizer, model


with st.spinner("LLM 모델을 불러오는 중입니다..."):
    tokenizer, model = load_model()


# =========================================================
# knowledge.txt 불러오기
# =========================================================

@st.cache_resource
def load_knowledge():

    with open(
        "knowledge.txt",
        "r",
        encoding="utf-8"
    ) as f:
        text = f.read()

    # 한 줄을 하나의 지식 데이터로 사용
    chunks = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    return chunks


knowledge_chunks = load_knowledge()


# =========================================================
# TF-IDF Retriever 생성
# =========================================================

@st.cache_resource
def create_retriever(chunks):

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4)
    )

    vectors = vectorizer.fit_transform(
        chunks
    )

    return vectorizer, vectors


vectorizer, knowledge_vectors = create_retriever(
    knowledge_chunks
)


# =========================================================
# 질문 정규화
# =========================================================

def normalize_question(question):

    q = question.strip()

    # 가족 표현 통일
    q = q.replace("엄마", "어머니")
    q = q.replace("아빠", "아버지")
    q = q.replace("강아지", "반려견")

    # 자기 자신을 나타내는 표현 통일
    q = q.replace("나의", "박동석의")
    q = q.replace("내가", "박동석이")
    q = q.replace("나는", "박동석은")
    q = q.replace("내", "박동석의")

    # 가족 질문의 '우리'
    q = q.replace("우리 어머니", "박동석의 어머니")
    q = q.replace("우리 아버지", "박동석의 아버지")
    q = q.replace("우리 형", "박동석의 형")
    q = q.replace("우리 반려견", "박동석의 반려견")
    q = q.replace("우리 가족", "박동석의 가족")

    return q


# =========================================================
# knowledge.txt 검색
# =========================================================

def retrieve_documents(question, top_k=1):

    normalized_question = normalize_question(
        question
    )

    question_vector = vectorizer.transform(
        [normalized_question]
    )

    similarities = cosine_similarity(
        question_vector,
        knowledge_vectors
    )[0]

    indexes = similarities.argsort()[::-1][:top_k]

    results = []

    for index in indexes:

        score = float(
            similarities[index]
        )

        # 너무 관련 없는 자료가 검색되는 것을 방지
        if score >= 0.15:

            results.append({
                "text": knowledge_chunks[index],
                "score": score
            })

    return results


# =========================================================
# 시스템 프롬프트
# =========================================================

RAG_SYSTEM_PROMPT = """
너는 박동석의 맞춤형 AI 비서다.

반드시 한국어로 답변한다.

제공된 참고 자료에 질문의 답이 있다면
참고 자료의 내용을 최우선으로 사용한다.

참고 자료의 사실을 변경하지 않는다.

참고 자료에 없는 내용을 임의로 만들지 않는다.

답변은 짧고 자연스럽게 작성한다.
"""


GENERAL_SYSTEM_PROMPT = """
너는 친절한 한국어 AI 비서다.

사용자의 일반적인 질문에 답변한다.

반드시 한국어로 답변한다.

확실하지 않은 내용은 임의로 만들어내지 않는다.

가능하면 짧고 이해하기 쉽게 답변한다.
"""


# =========================================================
# LLM 실행 함수
# =========================================================

def generate_llm_answer(
    system_prompt,
    user_prompt,
    max_new_tokens=100
):

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15
        )

    generated = output[0][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    return answer


# =========================================================
# 한국어 답변 품질 검사
# =========================================================

def is_good_korean_answer(answer):

    if not answer:
        return False

    korean_count = 0
    english_count = 0

    for ch in answer:

        if "가" <= ch <= "힣":
            korean_count += 1

        elif "a" <= ch.lower() <= "z":
            english_count += 1

    # 한국어가 거의 없는 경우
    if korean_count < 3:
        return False

    # 영어가 지나치게 많은 경우
    if english_count > korean_count * 2:
        return False

    # 너무 긴 이상한 답변 방지
    if len(answer) > 500:
        return False

    return True


# =========================================================
# 검색 문장 자연스럽게 변환
# =========================================================

def clean_document_answer(document):

    answer = document.strip()

    if answer.endswith("이다."):
        answer = answer[:-3] + "입니다."

    elif answer.endswith("한다."):
        answer = answer[:-3] + "합니다."

    elif answer.endswith("있다."):
        answer = answer[:-3] + "있습니다."

    elif answer.endswith("없다."):
        answer = answer[:-3] + "없습니다."

    return answer


# =========================================================
# 세션 대화 기록
# =========================================================

if "messages" not in st.session_state:

    st.session_state["messages"] = []


# =========================================================
# 이전 대화 출력
# =========================================================

for message in st.session_state["messages"]:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# =========================================================
# 사용자 질문 입력
# =========================================================

if question := st.chat_input(
    "메시지를 입력해 주세요"
):

    # 사용자 질문 저장
    st.session_state["messages"].append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.write(question)


    # =====================================================
    # 1. RAG 검색
    # =====================================================

    retrieved = retrieve_documents(
        question,
        top_k=1
    )


    # =====================================================
    # RAG 자료가 검색된 경우
    # =====================================================

    if retrieved:

    best_document = retrieved[0]["text"]

    rag_prompt = f"""
정보: {best_document}
질문: {question}

정보를 바탕으로 질문에 대한 답만 한국어 한 문장으로 작성하세요.
정보, 질문, 참고 자료 등의 내용을 답변에 다시 출력하지 마세요.
"""

    with st.spinner("답변 생성 중..."):
        llm_answer = generate_llm_answer(
            RAG_SYSTEM_PROMPT,
            rag_prompt,
            max_new_tokens=40
        )

    if is_good_korean_answer(llm_answer):
        bad_words = [
            "참고 자료",
            "[참고",
            "[질문]",
            "정보:",
            "질문:",
            "규칙:"
        ]

        if any(word in llm_answer for word in bad_words):
            answer = clean_document_answer(best_document)
        else:
            answer = llm_answer

    else:
        answer = clean_document_answer(best_document)

    # =====================================================
    # 2. knowledge.txt에 없는 일반 질문
    # =====================================================

    else:

        general_prompt = f"""
사용자의 다음 질문에 답변하세요.

질문:
{question}

조건:

1. 반드시 한국어로 답변하세요.
2. 가능한 한 정확하게 답변하세요.
3. 모르는 내용은 만들어내지 마세요.
4. 짧고 이해하기 쉽게 답변하세요.
"""


        with st.spinner(
            "LLM 자체 지식으로 답변 생성 중..."
        ):

            llm_answer = generate_llm_answer(
                GENERAL_SYSTEM_PROMPT,
                general_prompt,
                max_new_tokens=100
            )


        # 정상적인 한국어 답변이면 출력
        if is_good_korean_answer(
            llm_answer
        ):

            answer = llm_answer

        else:

            # SmolLM2가 영어 헛소리를 한 경우
            answer = (
                "이 질문은 등록된 지식 자료에 없으며, "
                "현재 사용 중인 소형 LLM이 "
                "안정적인 한국어 답변을 생성하지 못했습니다."
            )


    # =====================================================
    # 답변 출력
    # =====================================================

    with st.chat_message(
        "assistant"
    ):

        st.write(answer)


    # =====================================================
    # 답변 저장
    # =====================================================

    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer
    })
