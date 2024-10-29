import streamlit as st

st.set_page_config(page_title="Home", page_icon=":house:")

st.title("RAG with Youtube Video")

st.markdown(
    """
    이번 실습에선 RAG를 통해 **유튜브 영상**에 대한 답변을 하는 챗봇을 구현하는 과정을 실습해보도록 하겠습니다.  
    
    유튜브에 업로드된 영상에는 자막을 설정할 수 있습니다. `LangChain` 라이브러리의 `YoutubeLoader` 기능을 활용하면 영상에서 자막 스크립트를 추출할 수 있습니다.
    
    해당 자막을 임베딩하여 데이터베이스화 한다면 답변에 활용하는 챗봇을 구현할 수 있습니다.
    """
)