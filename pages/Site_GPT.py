from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from bs4 import BeautifulSoup
import streamlit as st


answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)

def get_answer(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answer_chain = answers_prompt|llm 
    return [
       {
           "question":question,
           "answer": answer_chain.invoke(
            {"question": question, "context": doc.page_content}
        ).content,
        "source":doc.metadata["source"],
        "date":doc.metadata["lastmod"],         
       } for doc in docs
    ]

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.
            Use the answers that have the highest score (more helpful) and favor the most recent ones.
            Cite sources and return the sources of the answers as they are, do not change them.
            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

def choose_answer():
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt|llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )

@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(url,
                           parsing_function=parse_page,
                           filter_urls=[
            r"^(./ai-gateway/).",
            r"^(./vectorize/).",
            r"^(./workers-ai/).",
        ],
                           )    
    loader.requests_per_second = 1
    docs = loader.load_and_split(text_splitter=splitter)
    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vectorstore.as_retriever()

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
            .replace("\n", " ")
            .replace("\xa0", " ")
            .replace("CloseSearch Submit Blog", "")
            )

def send_message(message, role, save=True):#AI와 대화하는 함수이다. 
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


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)
api_key = st.sidebar.text_input("OPENAI_API_KEY",placeholder="Enter you OPENAI_API_KEY",type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    st.success("API key has been set successfully.")
    llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
)

st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )
    st.subheader("Github Repository")
    st.write("https://github.com/Whaileinthesky/siteGPT")


if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        retriever = load_website(url)
        send_message("I'm ready! Ask away!", "ai", save=False)
        paint_history()
        query = st.chat_input("Ask a question to the website.")
        if query:
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answer)
                | RunnableLambda(choose_answer)
            )
            result = chain.invoke(query)
            send_message(response.content, "ai")
            st.markdown(result.content.replace("$", "\$"))
        else:#없으면 세션스테이트 초기화
            st.session_state["messages"] = []