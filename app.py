import streamlit as st
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


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


# =========================================================
# SmolLM2 모델 로딩
# =========================================================

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

@st.cache_data
def load_knowledge():

    with open(
        "knowledge.txt",
        "r",
        encoding="utf-8"
    ) as f:
        text = f.read()

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

    # 가족 질문부터 먼저 처리
    q = q.replace("우리 어머니", "박동석의 어머니")
    q = q.replace("우리 아버지", "박동석의 아버지")
    q = q.replace("우리 형", "박동석의 형")
    q = q.replace("우리 반려견", "박동석의 반려견")
    q = q.replace("우리 가족", "박동석의 가족")

    # 자기 자신 표현
    q = q.replace("나의", "박동석의")
    q = q.replace("내가", "박동석이")
    q = q.replace("나는", "박동석은")
    q = q.replace("내", "박동석의")

    return q


# =========================================================
# 정확 키워드 검색
# =========================================================

def exact_keyword_search(question):

    q = normalize_question(
        question
    )

    keyword_map = [
        ("소속", "소속은"),
        ("학과", "학과는"),
        ("아버지", "아버지"),
        ("어머니", "어머니"),
        ("반려견", "반려견"),
        ("가족", "가족"),
        ("이름", "이름은"),
        ("형", "형")
    ]

    for keyword, target in keyword_map:

        if keyword in q:

            for text in knowledge_chunks:

                if target in text:

                    # 이름 질문일 경우
                    # 아버지 이름, 어머니 이름 등이
                    # 잘못 걸리는 것을 방지
                    if keyword == "이름":

                        if "아버지" in q:
                            if "아버지 이름" in text:
                                return text

                        elif "어머니" in q:
                            if "어머니 이름" in text:
                                return text

                        elif "형" in q:
                            if "형 이름" in text:
                                return text

                        elif "반려견" in q:
                            if "반려견 이름" in text:
                                return text

                        else:
                            if text.startswith("박동석의 이름"):
                                return text

                    else:
                        return text

    return None


# =========================================================
# TF-IDF 검색
# =========================================================

def retrieve_documents(
    question,
    top_k=2
):

    # -----------------------------------------------------
    # 1. 정확 키워드 검색 먼저
    # -----------------------------------------------------

    exact_result = exact_keyword_search(
        question
    )

    if exact_result:

        return [
            {
                "text": exact_result,
                "score": 1.0
            }
        ]

    # -----------------------------------------------------
    # 2. TF-IDF 검색
    # -----------------------------------------------------

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

    indexes = (
        similarities
        .argsort()[::-1][:top_k]
    )

    results = []

    for index in indexes:

        score = float(
            similarities[index]
        )

        # 너무 관련 없는 결과 제외
        if score >= 0.20:

            results.append({
                "text":
                    knowledge_chunks[index],

                "score":
                    score
            })

    return results


# =========================================================
# format_docs
# =========================================================
#
# 교수님 PPT의 format_docs 역할
# =========================================================

def format_docs(documents):

    return "\n\n".join(
        document["text"]
        for document in documents
    )


# =========================================================
# SmolLM2 실행 함수
# =========================================================

def generate_with_smol(prompt):

    messages = [
        {
            "role": "system",
            "content": (
                "너는 친절한 한국어 AI 비서다. "
                "반드시 한국어로 답변한다. "
                "제공된 Context가 있으면 "
                "그 내용을 가장 우선한다. "
                "답변은 짧게 작성한다."
            )
        },
        {
            "role": "user",
            "content": prompt
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
            max_new_tokens=50,
            do_sample=False,
            repetition_penalty=1.15
        )

    generated_tokens = output[0][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return answer


# =========================================================
# LangChain LLM Runnable
# =========================================================

def llm_function(prompt_value):

    if hasattr(
        prompt_value,
        "to_string"
    ):

        prompt_text = (
            prompt_value.to_string()
        )

    else:

        prompt_text = str(
            prompt_value
        )

    return generate_with_smol(
        prompt_text
    )


llm = RunnableLambda(
    llm_function
)


# =========================================================
# RAG Prompt
# =========================================================

rag_prompt = PromptTemplate.from_template(
    """
다음 Context를 이용하여 질문에 답하세요.

Context:
{context}

Question:
{question}

규칙:
1. 반드시 한국어로 답변하세요.
2. Context의 내용을 가장 우선해서 사용하세요.
3. Context의 사실을 바꾸지 마세요.
4. Context에 없는 내용을 만들지 마세요.
5. 질문에 대한 답만 작성하세요.
6. Context, Question, 규칙을 다시 출력하지 마세요.
7. 한 문장으로 짧게 답변하세요.

Answer:
"""
)


# =========================================================
# RAG Chain
# =========================================================
#
# Prompt
#   |
# LLM
#   |
# StrOutputParser
#
# Retriever / format_docs 는
# 앞에서 경량 방식으로 직접 실행
# =========================================================

rag_chain = (

    rag_prompt

    |

    llm

    |

    StrOutputParser()

)


# =========================================================
# 일반 질문 Prompt
# =========================================================

general_prompt = PromptTemplate.from_template(
    """
다음 질문에 답변하세요.

Question:
{question}

규칙:
1. 반드시 한국어로 답변하세요.
2. 가능한 한 정확하게 답변하세요.
3. 확실하지 않은 내용은 만들지 마세요.
4. 짧고 이해하기 쉽게 답변하세요.

Answer:
"""
)


# =========================================================
# 일반 질문 Chain
# =========================================================

general_chain = (

    general_prompt

    |

    llm

    |

    StrOutputParser()

)


# =========================================================
# 답변 품질 검사
# =========================================================

def is_good_korean_answer(
    answer,
    document=None
):

    if not answer:
        return False

    # 글자 깨짐
    if "�" in answer:
        return False

    bad_words = [
        "Context:",
        "Question:",
        "Answer:",
        "규칙:",
        "참고 자료",
        "[질문]",
        "[참고",
        "정보:",
        "질문:",
        "Ryotan",
        "Shinzoku",
        "\"정보\"",
        "\"질문\""
    ]

    for word in bad_words:

        if word in answer:
            return False

    korean_count = sum(
        1
        for ch in answer
        if "가" <= ch <= "힣"
    )

    english_count = sum(
        1
        for ch in answer
        if "a" <= ch.lower() <= "z"
    )

    # 한국어가 너무 적음
    if korean_count < 5:
        return False

    # 영어 비율이 지나치게 높음
    if english_count > korean_count:
        return False

    # 이상하게 긴 답변
    if len(answer) > 250:
        return False

    # -----------------------------------------------------
    # 검색된 문서와 전혀 관련 없는 답변 방지
    # -----------------------------------------------------

    if document:

        important_words = [
            word.strip(
                ".,!?()[]{}\"'"
            )
            for word in document.split()
            if len(
                word.strip(
                    ".,!?()[]{}\"'"
                )
            ) >= 2
        ]

        match_count = sum(
            1
            for word in important_words
            if word in answer
        )

        if match_count == 0:
            return False

    return True


# =========================================================
# 검색 문장을 자연스럽게 변환
# =========================================================

def clean_document_answer(document):

    answer = document.strip()

    if answer.endswith("이다."):

        answer = (
            answer[:-3]
            + "입니다."
        )

    elif answer.endswith("한다."):

        answer = (
            answer[:-3]
            + "합니다."
        )

    elif answer.endswith("있다."):

        answer = (
            answer[:-3]
            + "있습니다."
        )

    elif answer.endswith("없다."):

        answer = (
            answer[:-3]
            + "없습니다."
        )

    return answer


# =========================================================
# 세션 대화 기록
# =========================================================

if "messages" not in st.session_state:

    st.session_state[
        "messages"
    ] = []


# =========================================================
# 이전 대화 출력
# =========================================================

for message in st.session_state[
    "messages"
]:

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
    st.session_state[
        "messages"
    ].append({
        "role": "user",
        "content": question
    })

    with st.chat_message(
        "user"
    ):

        st.write(
            question
        )


    # =====================================================
    # 1. Retriever 실행
    # =====================================================

    retrieved = retrieve_documents(
        question,
        top_k=2
    )


    # =====================================================
    # 2. 검색 자료가 있는 경우
    # =====================================================

    if retrieved:

        context = format_docs(
            retrieved
        )

        best_document = (
            retrieved[0]["text"]
        )

        with st.spinner(
            "RAG를 이용하여 답변 생성 중..."
        ):

            llm_answer = (
                rag_chain.invoke({
                    "context":
                        context,

                    "question":
                        question
                })
            )


        # =================================================
        # 정상적인 LLM 답변
        # =================================================

        if is_good_korean_answer(
            llm_answer,
            best_document
        ):

            answer = llm_answer


        # =================================================
        # SmolLM2 답변이 이상한 경우
        # =================================================

        else:

            answer = (
                clean_document_answer(
                    best_document
                )
            )


    # =====================================================
    # 3. knowledge.txt에 관련 정보가 없는 경우
    # =====================================================

    else:

        with st.spinner(
            "LLM 자체 지식으로 답변 생성 중..."
        ):

            llm_answer = (
                general_chain.invoke({
                    "question":
                        question
                })
            )


        if is_good_korean_answer(
            llm_answer
        ):

            answer = llm_answer


        else:

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

        st.write(
            answer
        )


    # =====================================================
    # 답변 저장
    # =====================================================

    st.session_state[
        "messages"
    ].append({
        "role": "assistant",
        "content": answer
    })
