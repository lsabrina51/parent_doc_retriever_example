import streamlit as st
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from dotenv import load_dotenv
import os
import logging 
from langchain_core.runnables import RunnableLambda

# Set up logging to file (append mode)
logging.basicConfig(filename="umich_pdr_log.txt", 
                    filemode='a', 
                    format='%(asctime)s - %(message)s', 
                    level=logging.INFO)

# ---------------------- Streamlit Setup ---------------------- #
st.set_page_config(page_title="PDR Umich Enrollment", page_icon="🎓")
st.title("PDR Umich Enrollment")
st.caption("Ask about student organizations, clubs, and related resources at the University of Michigan.")

load_dotenv()
endpoint = os.environ['AZURE_OPENAI_ENDPOINT']

# ---------------------- LLM Setup ---------------------- #
llm = AzureChatOpenAI(
    deployment_name=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
    openai_api_version=os.environ['AZURE_OPENAI_API_VERSION'],
    openai_api_key=os.environ['AZURE_OPENAI_API_KEY'],
    azure_endpoint=endpoint,
    openai_organization=os.environ['AZURE_OPENAI_ORGANIZATION'],
    temperature=0.25
)

# ---------------------- Vectorstore + Parent Retriever ---------------------- #
@st.cache_resource
def load_split_parent_vectorstore():
    loader = PyPDFLoader("umich-example.pdf")
    docs = loader.load()
    logging.info(f"[LOAD] Loaded {len(docs)} PDF documents")

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=endpoint,
        openai_api_version=os.environ['AZURE_OPENAI_API_VERSION'],
        openai_api_key=os.environ['AZURE_OPENAI_API_KEY'],
        openai_organization=os.environ['AZURE_OPENAI_ORGANIZATION'],
        model="text-embedding-3-small"
    )

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    parent_docs = parent_splitter.split_documents(docs)
    logging.info(f"[SPLIT] Created {len(parent_docs)} parent chunks")

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    child_docs = child_splitter.split_documents(parent_docs)
    logging.info(f"[SPLIT] Created {len(child_docs)} child chunks from parent chunks")

    # Assign unique doc_id to parents and pass to children
    parent_store = InMemoryStore()
    parent_map = []

    for i, doc in enumerate(parent_docs):
        doc_id = f"doc_{i}"
        doc.metadata["doc_id"] = doc_id
        parent_map.append((doc_id, doc))
        logging.info(f"[PARENT] Assigned doc_id={doc_id} to parent chunk: {doc.page_content[:80]}...")

    parent_store.mset(parent_map)

    # Ensure each child doc gets the right doc_id from its parent
    unmatched_children = 0
    for child in child_docs:
        matched = False
        for parent in parent_docs:
            if child.page_content in parent.page_content:
                child.metadata["doc_id"] = parent.metadata["doc_id"]
                matched = True
                break
        if not matched:
            unmatched_children += 1
            logging.warning(f"[CHILD] No parent match found for child: {child.page_content[:80]}...")

    logging.info(f"[MAPPING] Mapped doc_id to {len(child_docs) - unmatched_children} child chunks")
    if unmatched_children > 0:
        logging.warning(f"[MAPPING] {unmatched_children} child chunks could not be matched to a parent")

    # Build or load vectorstore
    persist_directory = "umich_pdr_chroma_store"
    try:
        vectorstore = Chroma(
            collection_name="split_parents",
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        test_results = vectorstore.similarity_search("test", k=1)
        logging.info(f"[VECTORSTORE] Test query returned {len(test_results)} results")
        if not test_results:
            raise ValueError("Vectorstore appears empty")
    except Exception as e:
        logging.warning(f"[VECTORSTORE] Rebuilding due to: {e}")
        vectorstore = Chroma.from_documents(
            documents=child_docs,
            embedding=embeddings,
            collection_name="split_parents",
            persist_directory=persist_directory
        )
        if hasattr(vectorstore, "persist"):
            vectorstore.persist()
            logging.info(f"[VECTORSTORE] Persisted new vectorstore with {len(child_docs)} child chunks")

    return vectorstore, parent_store

# ---------------------- Prompt Templates ---------------------- #
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

# ---------------------- Load Vectorstore + Retriever ---------------------- #
child_vectorstore, parent_store = load_split_parent_vectorstore()


# Create the parent document retriever
retriever = ParentDocumentRetriever(
    vectorstore=child_vectorstore,
    docstore=parent_store,
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
)

#start of logging 
def log_and_retrieve(query):
    results = retriever.invoke(query)
    
    logging.info(f"Query: {query}")
    logging.info(f"Retrieved {len(results)} documents:")
    for i, doc in enumerate(results):
        content_preview = doc.page_content
        #[:300].replace('\n', ' ')
        logging.info(f"Doc {i+1}: {content_preview} ...")
    
    return results

logging_retriever = RunnableLambda(log_and_retrieve)
#end of logging 

# Use the wrapped retriever in the chain
history_aware_retriever = create_history_aware_retriever(
    llm,
    logging_retriever,
    contextualize_q_prompt
)
qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# ---------------------- Chat Memory ---------------------- #
chat_store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_store:
        chat_store[session_id] = ChatMessageHistory()
    return chat_store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# ---------------------- Streamlit UI ---------------------- #
if "session_id" not in st.session_state:
    st.session_state.session_id = "user-session"

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

# Chat history toggle
if st.toggle("Show chat history"):
    history = chat_store.get(st.session_state.session_id)
    if history:
        for msg in history.messages:
            role = "You" if msg.type == "human" else "Bot"
            st.markdown(f"**{role}:** {msg.content}")
