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

st.set_page_config(page_title="RAG Chat with YouTube", page_icon=":video_camera:")

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

@st.cache_data(show_spinner="유튜브 자막을 불러오고 있습니다...", max_entries=1, ttl=600)
def load_from_youtube(input_url):
    try:
        video_id = extract_video_id(input_url)
        if not video_id:
            raise ValueError("올바르지 않은 유튜브 URL입니다.")
        loader = YoutubeLoader(video_id, language="ko")
        docs = loader.load()
        if not docs:
            raise ValueError("자막을 찾을 수 없습니다. 한글 자막이 있는지 확인하세요.")
        return docs
    except Exception as e:
        st.error(f"유튜브 자막을 가져오지 못했습니다: {e}")
        return None

@st.cache_resource(show_spinner="임베딩을 생성하고 있습니다...", ttl=600)
def run_embedding(_docs, url):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splitted_docs = text_splitter.split_documents(_docs)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(splitted_docs, embeddings)
    return vectorstore.as_retriever()

def load_previous_chat():
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["message"])

def save_message(message, role, max_messages=10):
    # 새로운 메시지를 추가
    st.session_state["messages"].append({"message": message, "role": role})
    
    # 메시지가 max_messages보다 많으면 오래된 메시지를 삭제
    if len(st.session_state["messages"]) > max_messages:
        # 가장 오래된 메시지를 삭제
        st.session_state["messages"].pop(0)

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def create_chain(retriever):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "유튜브 자막 내용을 바탕으로 유저의 질문에 도움이 될 수 있도록 답변해줘"),
            ("human", "{question}"),
        ]
    )

    llm = ChatOpenAI(
        openai_api_key=api_key,
        model="gpt-4o-mini",
        temperature=0.7,
        streaming=False,
    )

    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)

if url:

    if st.session_state["last_url"] != url:
        st.session_state["messages"] = []
        st.session_state["last_url"] = url

    docs = load_from_youtube(url)

    # API 키 로드 확인
st.write("API Key Loaded:", bool(api_key))

# URL 입력 검증
st.write("Input URL:", url)
st.write("Extracted Video ID:", extract_video_id(url))

# 자막 로딩 결과 확인
docs = load_from_youtube(url)
st.write("Loaded Documents:", docs)

# 임베딩 생성 확인
if docs:
    retriever = run_embedding(docs, url)
    st.write("Retriever Created:", bool(retriever))
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
