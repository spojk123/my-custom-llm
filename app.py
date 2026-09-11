import streamlit as st
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# Streamlit 기본 설정
# =========================================================

st.set_page_config(
    page_title="나만의 맞춤형 LLM 챗봇",
    page_icon="🦜"
)

st.title("🦜 박동석의 LLM")


# =========================================================
# 오픈소스 LLM
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

    # 한 줄 = 하나의 지식 데이터
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
# 질문 문장 정리
# =========================================================

def normalize_question(question):
    q = question.strip()

    # 가족 호칭 통일
    q = q.replace("엄마", "어머니")
    q = q.replace("아빠", "아버지")
    q = q.replace("강아지", "반려견")

    # 사용자 자신을 박동석으로 통일
    q = q.replace("나의", "박동석의")
    q = q.replace("내가", "박동석이")
    q = q.replace("나는", "박동석은")
    q = q.replace("내", "박동석의")
    q = q.replace("나", "박동석")

    return q


# =========================================================
# 관련 문서 검색
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

        # 관련도가 너무 낮으면 사용하지 않음
        if score >= 0.10:

            results.append({
                "text": knowledge_chunks[index],
                "score": score
            })

    return results


# =========================================================
# 시스템 프롬프트
# =========================================================

SYSTEM_PROMPT = """
너는 동서울대학교 컴퓨터소프트웨어과 학생 박동석의 전용 AI 비서다.

반드시 한국어로만 답변한다.

사용자의 질문에 답할 때 제공된 참고 자료를 최우선으로 사용한다.

참고 자료에 답이 존재하면 그 내용을 사실 그대로 사용한다.

참고 자료에 없는 사실은 절대 만들어내지 않는다.

사용자가 '나', '내', '우리'라고 말하면 박동석을 의미한다.

영어로 답변하지 않는다.

답변은 한 문장 또는 두 문장으로 짧고 자연스럽게 작성한다.
"""


# =========================================================
# LLM 답변 품질 검사
# =========================================================

def is_good_korean_answer(answer):

    if not answer:
        return False

    korean_count = 0
    english_count = 0

    for ch in answer:

        if "가" <= ch <= "힣":
            korean_count += 1

        elif (
            "a" <= ch.lower() <= "z"
        ):
            english_count += 1

    # 한국어가 거의 없으면 실패
    if korean_count < 3:
        return False

    # 영어 비율이 너무 높으면 실패
    if english_count > korean_count:
        return False

    # 너무 긴 헛소리 방지
    if len(answer) > 300:
        return False

    return True


# =========================================================
# 검색된 문장을 자연스럽게 변환
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
# 대화 기록 초기화
# =========================================================

if "messages" not in st.session_state:
    st.session_state["messages"] = []


# =========================================================
# 기존 대화 출력
# =========================================================

for message in st.session_state["messages"]:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# =========================================================
# 사용자 입력
# =========================================================

if question := st.chat_input(
    "메시지를 입력해 주세요"
):

    # 사용자 메시지 저장
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
    # 검색 결과 표시
    # =====================================================

    with st.expander(
        "🔎 검색된 참고 자료"
    ):

        if retrieved:

            for i, item in enumerate(
                retrieved,
                start=1
            ):

                st.write(
                    f"{i}. {item['text']}"
                )

                st.caption(
                    f"관련도: "
                    f"{item['score']:.3f}"
                )

        else:

            st.write(
                "관련된 자료를 찾지 못했습니다."
            )


    # =====================================================
    # 2. 검색 결과가 없는 경우
    # =====================================================

    if not retrieved:

        answer = (
            "등록된 지식 자료에서 "
            "해당 질문에 대한 정보를 찾지 못했습니다."
        )


    # =====================================================
    # 3. 검색 결과가 있는 경우
    # =====================================================

    else:

        best_document = retrieved[0]["text"]

        context = best_document


        user_prompt = f"""
다음 참고 자료만 이용해서 질문에 답하세요.

[참고 자료]

{context}

[질문]

{question}

규칙:
1. 반드시 한국어로 답변하세요.
2. 참고 자료의 사실을 변경하지 마세요.
3. 한 문장 또는 두 문장으로 간단하게 답변하세요.
4. 참고 자료에 없는 정보를 만들지 마세요.
"""


        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
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


        # =================================================
        # LLM 답변 생성
        # =================================================

        with st.spinner(
            "관련 자료를 검색하고 답변 생성 중..."
        ):

            with torch.no_grad():

                output = model.generate(
                    **inputs,
                    max_new_tokens=60,
                    do_sample=False,
                    repetition_penalty=1.2
                )


        generated = output[0][
            inputs["input_ids"].shape[1]:
        ]


        llm_answer = tokenizer.decode(
            generated,
            skip_special_tokens=True
        ).strip()


        # =================================================
        # 영어 헛소리 방지
        # =================================================

        if is_good_korean_answer(
            llm_answer
        ):

            answer = llm_answer

        else:

            # LLM이 이상하면
            # 검색 결과를 자연스럽게 출력
            answer = clean_document_answer(
                best_document
            )


    # =====================================================
    # 답변 출력
    # =====================================================

    with st.chat_message("assistant"):

        st.write(answer)


    # =====================================================
    # 답변 저장
    # =====================================================

    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer
    })


# =========================================================
# 사이드바
# =========================================================

with st.sidebar:

    st.header(
        "📚 박동석 맞춤형 LLM"
    )

    st.write(
        "외부 LLM API를 사용하지 않습니다."
    )

    st.write(
        "Streamlit Cloud에서 "
        "오픈소스 LLM을 직접 실행합니다."
    )

    st.write(
        "질문과 관련된 자료를 "
        "knowledge.txt에서 검색한 뒤 "
        "LLM에 참고 자료로 전달합니다."
    )

    st.write(
        f"등록된 지식 데이터: "
        f"{len(knowledge_chunks)}개"
    )

    st.divider()

    if st.button(
        "🗑️ 대화 내용 초기화"
    ):

        st.session_state["messages"] = []

        st.rerun()
