import streamlit as st
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==================================================
# Streamlit 기본 설정
# ==================================================

st.set_page_config(
    page_title="나만의 맞춤형 LLM 챗봇",
    page_icon="🦜"
)

st.title("🦜 박동석의 LLM")


# ==================================================
# LLM 모델
# ==================================================

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


# ==================================================
# 지식 문서 불러오기
# ==================================================

@st.cache_resource
def load_knowledge():

    with open(
        "knowledge.txt",
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read()

    # 빈 줄을 기준으로 지식을 여러 조각으로 분리
    chunks = [
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]

    return chunks


knowledge_chunks = load_knowledge()


# ==================================================
# TF-IDF 검색기
# ==================================================

@st.cache_resource
def create_retriever(chunks):

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4)
    )

    vectors = vectorizer.fit_transform(chunks)

    return vectorizer, vectors


vectorizer, knowledge_vectors = create_retriever(
    knowledge_chunks
)


# ==================================================
# 관련 문서 검색
# ==================================================

def retrieve_documents(question, top_k=3):

    question_vector = vectorizer.transform(
        [question]
    )

    similarities = cosine_similarity(
        question_vector,
        knowledge_vectors
    )[0]

    indexes = similarities.argsort()[::-1][:top_k]

    documents = []

    for index in indexes:

        # 너무 관련 없는 문서는 제외
        if similarities[index] > 0.03:

            documents.append(
                knowledge_chunks[index]
            )

    return documents


# ==================================================
# 시스템 프롬프트
# ==================================================

SYSTEM_PROMPT = """
너는 동서울대학교 컴퓨터소프트웨어과 학생
박동석을 위한 맞춤형 AI 비서다.

반드시 한국어로 답변한다.

사용자의 질문에 답할 때
제공된 참고 자료를 가장 우선적으로 사용한다.

참고 자료에 답이 있다면
반드시 참고 자료의 내용을 기준으로 답한다.

참고 자료에 없는 사실을
임의로 만들어내지 않는다.

답변은 짧고 자연스러운 한국어 문장으로 작성한다.

사용자가 '나', '내', '우리 가족'과 같이 질문하면
사용자는 박동석을 의미한다.
"""


# ==================================================
# 대화 기록
# ==================================================

if "messages" not in st.session_state:

    st.session_state["messages"] = []


for message in st.session_state["messages"]:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# ==================================================
# 사용자 입력
# ==================================================

if question := st.chat_input(
    "메시지를 입력해 주세요"
):

    st.session_state["messages"].append(
        {
            "role": "user",
            "content": question
        }
    )


    with st.chat_message("user"):

        st.write(question)


    # ==============================================
    # RAG 검색
    # ==============================================

    documents = retrieve_documents(
        question,
        top_k=3
    )


    if documents:

        context = "\n\n".join(
            documents
        )

    else:

        context = (
            "질문과 관련된 정보가 "
            "지식 문서에 없습니다."
        )


    # ==============================================
    # LLM에 전달할 프롬프트
    # ==============================================

    user_prompt = f"""
[참고 자료]

{context}

[사용자 질문]

{question}

참고 자료에 있는 내용만 이용해서
한국어로 정확하고 간단하게 답변하세요.
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


    # ==============================================
    # LLM 답변 생성
    # ==============================================

    with st.chat_message("assistant"):

        with st.spinner(
            "관련 자료를 검색하고 답변 생성 중..."
        ):

            with torch.no_grad():

                output = model.generate(
                    **inputs,
                    max_new_tokens=120,
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


        if not answer:

            answer = (
                "관련된 정보를 찾았지만 "
                "답변을 생성하지 못했습니다."
            )


        st.write(answer)


    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer
        }
    )


# ==================================================
# 검색된 자료 확인 기능
# ==================================================

with st.sidebar:

    st.header("📚 RAG 정보")

    st.write(
        f"등록된 지식 조각: "
        f"{len(knowledge_chunks)}개"
    )

    st.write(
        "질문과 관련된 자료를 검색한 후 "
        "오픈소스 LLM이 답변합니다."
    )
