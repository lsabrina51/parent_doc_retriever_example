import streamlit as st
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
import os


# Streamlit UI
st.set_page_config(page_title="Basic UMich Student Org Chatbot", page_icon="🎓")
st.title("🎓 UMich Student Org Chatbot")
st.caption("Ask about student organizations, clubs, and related resources at the University of Michigan.")

# Load environment variables
load_dotenv()

endpoint = os.environ['AZURE_OPENAI_ENDPOINT']

# Initialize LLM
llm = AzureChatOpenAI(
    deployment_name=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
    openai_api_version=os.environ['AZURE_OPENAI_API_VERSION'],
    openai_api_key=os.environ['AZURE_OPENAI_API_KEY'],
    azure_endpoint=endpoint,
    openai_organization=os.environ['AZURE_OPENAI_ORGANIZATION'],
    temperature=0.25
)

# Load documents and embed if necessary
@st.cache_resource
def load_vectorstore():
    loader = PyPDFLoader("umich-example.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=endpoint, 
        openai_api_version=os.environ['AZURE_OPENAI_API_VERSION'],  
        openai_api_key=os.environ['AZURE_OPENAI_API_KEY'],   
        openai_organization=os.environ['AZURE_OPENAI_ORGANIZATION'],
        model="text-embedding-3-large" 
    )

    try:
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory="umich_chroma_store"
        )
        if vectorstore._collection.count() == 0:
            raise ValueError("Empty vectorstore")
    except:
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory="umich_chroma_store"
        )
        vectorstore.persist()
    return vectorstore

vectorstore = load_vectorstore()

# Prompt templates
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_system_prompt = """You are an expert in student organizations at the University of Michigan. 
Suggest groups, clubs, and resources. Include supporting email addresses and web links.

{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# RAG Chain
retriever =vectorstore.as_retriever(search_kwargs={"k": 4})

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# Session-level chat history
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


# Session input
if "session_id" not in st.session_state:
    st.session_state.session_id = "user-session"

# Chat form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Your question:")
    submitted = st.form_submit_button("Ask")

if submitted and user_input:
    with st.spinner("Thinking..."):
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": st.session_state.session_id}},
        )
    st.markdown("**Answer:**")
    st.write(response["answer"])

# Show history
if st.toggle("Show chat history"):
    history = store.get(st.session_state.session_id)
    if history:
        for msg in history.messages:
            role = "You" if msg.type == "human" else "Bot"
            st.markdown(f"**{role}:** {msg.content}")