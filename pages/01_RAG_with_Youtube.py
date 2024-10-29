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
            ("system", "유튜브 자막 내용을 바탕으로 유저의 질문에 도움이 될 수 있도록 답변해줘"),
            ("human", "{question}"),
        ]
    )

    llm = ChatOpenAI(
        openai_api_key=api_key,
        model="gpt-4o-mini",
        temperature=0.7,
        streaming=True,
    )

    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)

if url:

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
