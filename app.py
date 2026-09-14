import streamlit as st
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda
)

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


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
# Embedding 모델 생성
# =========================================================

@st.cache_resource
def load_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        ),
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings


with st.spinner("Embedding 모델을 불러오는 중입니다..."):
    embeddings = load_embeddings()


# =========================================================
# FAISS Vector Store 생성
# =========================================================

@st.cache_resource
def create_vectorstore(chunks):

    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    return vectorstore


vectorstore = create_vectorstore(
    knowledge_chunks
)


# =========================================================
# Retriever 생성
# =========================================================

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 2
    }
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

    # 자기 자신 표현
    q = q.replace("나의", "박동석의")
    q = q.replace("내가", "박동석이")
    q = q.replace("나는", "박동석은")
    q = q.replace("내", "박동석의")

    # 가족 질문
    q = q.replace(
        "우리 어머니",
        "박동석의 어머니"
    )

    q = q.replace(
        "우리 아버지",
        "박동석의 아버지"
    )

    q = q.replace(
        "우리 형",
        "박동석의 형"
    )

    q = q.replace(
        "우리 반려견",
        "박동석의 반려견"
    )

    q = q.replace(
        "우리 가족",
        "박동석의 가족"
    )

    return q


# =========================================================
# normalize Runnable
# =========================================================

normalize_runnable = RunnableLambda(
    normalize_question
)


# =========================================================
# format_docs
# =========================================================

def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
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
                "Context에 답이 있으면 "
                "Context의 내용을 우선 사용한다. "
                "질문에 대한 답만 짧게 작성한다."
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
            max_new_tokens=60,
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
        prompt_text = prompt_value.to_string()

    else:
        prompt_text = str(prompt_value)

    return generate_with_smol(
        prompt_text
    )


llm = RunnableLambda(
    llm_function
)


# =========================================================
# RAG Prompt
# =========================================================

prompt = PromptTemplate.from_template(
    """
다음 Context를 이용하여 질문에 답하세요.

Context:
{context}

Question:
{question}

규칙:
1. 반드시 한국어로 답변하세요.
2. Context에 있는 내용을 가장 우선해서 사용하세요.
3. Context의 사실을 변경하지 마세요.
4. Context에 없는 내용을 임의로 만들지 마세요.
5. 질문에 대한 답만 작성하세요.
6. Context, Question, 규칙을 다시 출력하지 마세요.
7. 한 문장으로 짧게 답변하세요.

Answer:
"""
)


# =========================================================
# 교수님 PPT 형태의 RAG Chain
# =========================================================

rag_chain = (

    {
        "context":
            normalize_runnable
            |
            retriever
            |
            RunnableLambda(format_docs),

        "question":
            RunnablePassthrough()
    }

    |

    prompt

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
다음 사용자의 질문에 답변하세요.

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
# 한국어 답변 품질 검사
# =========================================================

def is_good_korean_answer(
    answer,
    docs=None
):

    if not answer:
        return False

    # 깨진 문자
    if "�" in answer:
        return False

    # 프롬프트 유출 / 이상한 출력
    bad_words = [
        "Context:",
        "Question:",
        "Answer:",
        "규칙:",
        "참고 자료",
        "[질문]",
        "[참고",
        "정보를 바탕으로",
        "Ryotan",
        "Shinzoku",
        "\"정보\"",
        "\"질문\""
    ]

    for word in bad_words:

        if word in answer:
            return False

    korean_count = 0
    english_count = 0

    for ch in answer:

        if "가" <= ch <= "힣":
            korean_count += 1

        elif "a" <= ch.lower() <= "z":
            english_count += 1

    # 한국어가 너무 적음
    if korean_count < 5:
        return False

    # 영어가 한국어보다 너무 많음
    if english_count > korean_count:
        return False

    # 너무 긴 이상한 답변
    if len(answer) > 250:
        return False

    # =====================================================
    # 검색된 문서와 답변 관련성 검사
    # =====================================================

    if docs:

        context = " ".join(
            doc.page_content
            for doc in docs
        )

        context_words = [
            word.strip(
                ".,!?()[]{}\"'"
            )
            for word in context.split()
            if len(
                word.strip(
                    ".,!?()[]{}\"'"
                )
            ) >= 2
        ]

        match_count = sum(
            1
            for word in context_words
            if word in answer
        )

        # 검색된 문서 단어가 하나도 없으면
        # hallucination 가능성이 높음
        if match_count == 0:
            return False

    return True


# =========================================================
# 검색 문장 자연스럽게 변환
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
# 검색 결과의 실제 관련도 확인
# =========================================================

def get_relevant_documents(question):

    normalized_question = normalize_question(
        question
    )

    results = (
        vectorstore
        .similarity_search_with_relevance_scores(
            normalized_question,
            k=2
        )
    )

    relevant_docs = []

    for document, score in results:

        # 너무 관련 없는 문서 방지
        if score >= 0.28:

            relevant_docs.append(
                document
            )

    return relevant_docs


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
    # 관련 문서 검색
    # =====================================================

    with st.spinner(
        "관련 자료를 검색하는 중..."
    ):

        relevant_docs = (
            get_relevant_documents(
                question
            )
        )


    # =====================================================
    # 1. 관련 자료가 있는 경우
    # → 교수님 PPT 형태 RAG Chain 실행
    # =====================================================

    if relevant_docs:

        with st.spinner(
            "RAG를 이용하여 답변 생성 중..."
        ):

            llm_answer = (
                rag_chain.invoke(
                    question
                )
            )

        # 정상적인 한국어 답변이면 사용
        if is_good_korean_answer(
            llm_answer,
            relevant_docs
        ):

            answer = llm_answer

        # SmolLM2가 이상한 답변 생성 시
        # 가장 관련 높은 검색 문장으로 대체
        else:

            answer = (
                clean_document_answer(
                    relevant_docs[
                        0
                    ].page_content
                )
            )


    # =====================================================
    # 2. knowledge.txt에 없는 일반 질문
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
