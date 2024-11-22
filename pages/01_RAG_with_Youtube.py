import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import YoutubeLoader
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

st.set_page_config(
    page_title="RAG Chat with YouTube", page_icon=":video_camera:"
)

st.title("YouTube 영상 기반 RAG 분석")

st.markdown(
    """
    유튜브 영상의 한글 자막을 바탕으로 채팅이 가능합니다.
    """
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "last_url" not in st.session_state:
    st.session_state["last_url"] = None


def extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be" in url:
        return url.split("/")[-1]
    else:
        return None


url = st.text_input(
    "채팅에 활용할 YouTube 링크를 입력하세요",
    "",
    help="채팅에 사용할 유튜브 링크를 입력하세요 (한글 자막이 있는 경우만 사용 가능합니다)",
)


@st.cache_data(show_spinner="유튜브 자막을 불러오고 있습니다...", max_entries=1)
def load_from_youtube(input_url):
    try:
        video_id = extract_video_id(input_url)
        if not video_id:
            raise ValueError("올바르지 않은 유튜브 URL입니다.")
        loader = YoutubeLoader(video_id, language="ko")
        docs = loader.load()
        return docs
    except Exception as e:
        st.error(f"유튜브 자막을 가져오지 못했습니다: {e}")
        return None


@st.cache_resource(show_spinner="임베딩을 생성하고 있습니다...")
def run_embedding(_docs, url):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    )
    splitted_docs = text_splitter.split_documents(_docs)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(splitted_docs, embeddings)
    return vectorstore.as_retriever()


def load_previous_chat():
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["message"])


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def create_chain(retriever):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
            당신은 유튜브 자막 데이터를 기반으로 사용자의 질문에 답변하는 RAG(정보 검색 증강) 챗봇입니다. 당신의 역할은 사용자가 입력한 질문을 분석하고, 질문의 의도를 파악한 후 유튜브 자막 데이터에서 관련 있는 정보를 검색하고 필요한 내용을 요약하는 것입니다. 검색된 정보가 충분하지 않을 경우, 일반적인 지식이나 문맥을 바탕으로 보충하여 응답해야 합니다. 답변은 정확하고 간결하며, 사용자의 질문 의도에 맞도록 작성되어야 합니다.

            답변 작성 시 다음 규칙을 준수하십시오:
            1. 답변은 항상 자막 데이터에서 파생된 정보를 우선적으로 사용합니다.
            2. 사용자가 제공한 질문에 맞는 자막 내 특정 문장이나 섹션을 명시적으로 인용할 수 있습니다.
            3. 자막 데이터에 관련 정보가 없을 경우, "해당 질문에 대한 정보는 제공된 자막 데이터에 없습니다."라고 답변합니다.
            4. 필요하면 추가 설명이나 맥락을 제공하되, 과도한 추측은 하지 않습니다.
            5. 질문에 대한 응답은 자연스럽고 이해하기 쉬운 문장으로 작성합니다.
            
            예를 들어, 사용자가 "이 영상에서 주요 키워드는 무엇인가요?"라고 묻는다면, 제공된 자막 데이터에서 관련 키워드를 분석하여 "제공된 자막 데이터에서 주요 키워드로 '데이터 분석', '머신 러닝', '시각화 기술'이 언급되었습니다."와 같이 응답합니다. 사용자가 "이 강의에서 설명한 데이터 분석 과정은 무엇인가요?"라고 묻는다면, 자막 데이터를 기반으로 "자막에 따르면 데이터 분석 과정은 1) 데이터 수집, 2) 데이터 정제, 3) 데이터 분석, 4) 시각화 및 보고 단계로 나뉩니다."와 같이 응답합니다.
            """),
            ("human", "{question}"),
        ]
    )

    llm = ChatOpenAI(
        openai_api_key=api_key,
        model="gpt-4o-mini",
        temperature=0.7,
        streaming=True,
    )

    return RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=False
    )


if url:
    if st.session_state["last_url"] != url:
        st.session_state["messages"] = []
        st.session_state["last_url"] = url

    docs = load_from_youtube(url)
    if docs:
        retriever = run_embedding(docs, url)
        chain = create_chain(retriever)

        send_message("유튜브 자막이 처리되었습니다. 질문을 입력해주세요.", "ai", save=False)

        # 유튜브 영상을 접었다 펼 수 있는 Expander 추가
        with st.expander("YouTube 영상 보기"):
            video_id = extract_video_id(url)
            if video_id:
                st.video(f"https://www.youtube.com/watch?v={video_id}")

        load_previous_chat()

        question = st.chat_input("질문을 입력하세요.")
        if question:
            send_message(question, "human")

            response = chain.run({"query": question})
            send_message(response, "ai", save=True)

else:
    st.info("유튜브 링크가 입력되면 채팅이 시작됩니다.")
