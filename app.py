from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
import os
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

api_key = st.sidebar.text_input("OPENAI_API_KEY",placeholder="Enter you OPENAI_API_KEY",type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    st.success("API key has been set successfully.")
    llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
)

@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)#파일을 가져옴
    docs = loader.load_and_split(text_splitter=splitter)#가져온 파일을 분리함
    embeddings = OpenAIEmbeddings()#파일을 AI가 사용하는 벡터로 변환함
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)#돈을 절약하기위해 캐시에 저장함
    vectorstore = FAISS.from_documents(docs, cached_embeddings)#저장된 것을 사용함
    retriever = vectorstore.as_retriever()
    return retriever


def send_message(message, role, save=True):#AI와 대화하는 함수
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


def paint_history():#채팅기록이 날아가지 않게 다시 그리는 함수
    for message in st.session_state["messages"]:#세션 스테이트에 저장된것을 전부 꺼내온다.
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.

Enter your OPENAI_API_KEY in sidebar.
"""
)

with st.sidebar:#안에 있는 내용은 전부 st.sidebar
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )
    st.header("https://github.com/Whaileinthesky/FullStack_GPT")
    st.subheader("Function Part")
    st.code("""
@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)#파일을 가져옴
    docs = loader.load_and_split(text_splitter=splitter)#가져온 파일을 분리함
    embeddings = OpenAIEmbeddings()#파일을 AI가 사용하는 벡터로 변환함
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)#돈을 절약하기위해 캐시에 저장함
    vectorstore = FAISS.from_documents(docs, cached_embeddings)#저장된 것을 사용함
    retriever = vectorstore.as_retriever()
    return retriever


def send_message(message, role, save=True):#AI와 대화하는 함수
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


def paint_history():#채팅기록이 날아가지 않게 다시 그리는 함수
    for message in st.session_state["messages"]:#세션 스테이트에 저장된것을 전부 꺼내온다.
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)
""")
    st.subheader("Homepage")
    st.code("""
st.title("DocumentGPT")

st.markdown(
    "
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.

Enter your OPENAI_API_KEY in sidebar.
"
)

""")
    st.subheader("Sidebar")
    st.code("""
with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )
    st.header("https://github.com/Whaileinthesky/FullStack_GPT")
""")
    st.subheader("Execute Part")
    st.code("""
if file:#파일을 업로드하면
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        response = chain.invoke(message)
        send_message(response.content, "ai")

else:#없으면 세션스테이트 초기화
    st.session_state["messages"] = []
""")

    

if file:#파일을 업로드하면
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        response = chain.invoke(message)
        send_message(response.content, "ai")

else:#없으면 세션스테이트 초기화
    st.session_state["messages"] = []