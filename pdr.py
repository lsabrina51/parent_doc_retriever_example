from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

# Sample Q&A RAG application over a text data source
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory


# Sets the current working directory to be the same as the file.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load environment file for secrets.
try:
    if load_dotenv('.env') is False:
        raise TypeError
except TypeError:
    print('Unable to load .env file.')
    quit()

# Settings for embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], 
    openai_api_version=os.environ['AZURE_OPENAI_API_VERSION'],  
    openai_api_key=os.environ['AZURE_OPENAI_API_KEY'],   
    openai_organization=os.environ['AZURE_OPENAI_ORGANIZATION'],
    model="text-embedding-3-small"  
)

# Define llm parameters
llm = AzureChatOpenAI(
    deployment_name=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
    openai_api_version=os.environ['AZURE_OPENAI_API_VERSION'],
    openai_api_key=os.environ['AZURE_OPENAI_API_KEY'],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_organization=os.environ['AZURE_OPENAI_ORGANIZATION']
)

# Replace with the document(s) you wish to use
print("Loading document...")
loader = PyPDFLoader("umich-example.pdf")

docs = loader.load()

# This text splitter is used to create the parent documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
# This text splitter is used to create the child documents
# It should create documents smaller than the parent
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="split_parents", embedding_function=embeddings
)
# The storage layer for the parent documents
store = InMemoryStore()

#retrieves 
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(docs, ids=None)
list(store.yield_keys())

query = "GPA"

#normal similarity search for child splitter layer 
sub_docs = vectorstore.similarity_search(query)
if sub_docs:
    print("Similarity Search: ")
    print(sub_docs[0].page_content)
else:
    print("No relevant documents found.")

retrieved_docs = retriever.invoke(query)
if retrieved_docs: 
    print("Parent Document:")
    print(retrieved_docs[0].page_content)
else:
    print("No relevant documents found.")

# Contextualize question
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Manage chat history
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

# User requests
while True:
    text = input('Enter your query (Example: How many undergrad students are at U of M?): --> ') # Example: How many undergrad students are at U of M?
    #call
    print(conversational_rag_chain.invoke(
        {"input": text},
        config={"configurable": {"session_id": "0"}},
    )["answer"])



#Questions: 
#Are we using Reserved Instances (RIs) or Savings Plans effectively?


