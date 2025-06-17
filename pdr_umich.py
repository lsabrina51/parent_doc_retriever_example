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
f = open("umich_output_pdr.txt", "a")


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
# while True:
#     text = input('Enter your query (Example: How many undergrad students are at U of M?): --> ') # Example: How many undergrad students are at U of M?
#     #call
#     print(conversational_rag_chain.invoke(
#         {"input": text},
#         config={"configurable": {"session_id": "0"}},
#     )["answer"])

questions = [
    "What is the enrollment at the University of Michigan?",
    "How many undergrad students are at U of M?", 
    "tell me more about 2015 cohort"
]
session_id = "0"
# answer = conversational_rag_chain.invoke({"input": "How many undergrad students are at U of M?"}, config={"configurable": {"session_id": session_id}})
# f.write(f"A: {answer}\n")
# answer = conversational_rag_chain.invoke({"input": "tell me more about the 2015 cohort"}, config={"configurable": {"session_id": session_id}})
# f.write(f"A: {answer}\n")
# for question in questions:
#     f.write(f"Q: {question}\n")

#     # Run RAG
#     answer = conversational_rag_chain.invoke(
#         {"input": question},
#         config={"configurable": {"session_id": session_id}},
#     )["answer"]

#     # Similarity search on vectorstore
#     sub_docs = vectorstore.similarity_search(question)
#     if sub_docs:
#         f.write("Similarity Search:\n")
#         f.write(sub_docs[0].page_content + "\n\n")
#     else:
#         f.write("No relevant documents found in similarity search.\n\n")

#     # Parent document retrieval
#     retrieved_docs = retriever.invoke(question)
#     if retrieved_docs:
#         f.write("Parent Document:\n")
#         f.write(retrieved_docs[0].page_content + "\n\n")
#     else:
#         f.write("No relevant parent documents found.\n\n")

#     f.write(f"A: {answer}\n")
#     f.write("-" * 60 + "\n\n")

session_id = 0  # unique or fixed depending on stateless or contextual behavior

# 1. First Question
question1 = "How many undergrad students are at U of M?"
f.write(f"Q: {question1}\n")

answer1 = conversational_rag_chain.invoke(
    {"input": question1},
    config={"configurable": {"session_id": session_id}},
)["answer"]
f.write(f"A: {answer1}\n")

# Child-level similarity search
sub_docs1 = vectorstore.similarity_search(question1)
if sub_docs1:
    f.write("Similarity Search (Child-level):\n")
    for doc in sub_docs1:
        f.write(doc.page_content + "\n---\n")
else:
    f.write("No relevant child documents found.\n")

# Parent-level retrieval
parent_docs1 = retriever.invoke(question1)
if parent_docs1:
    f.write("Parent Document(s):\n")
    for doc in parent_docs1:
        f.write(doc.page_content + "\n---\n")
else:
    f.write("No relevant parent documents found.\n")

f.write("="*80 + "\n")

# 2. Second Question
question2 = "tell me more about the 2015 cohort"
f.write(f"Q: {question2}\n")

answer2 = conversational_rag_chain.invoke(
    {"input": question2},
    config={"configurable": {"session_id": session_id}},
)["answer"]
f.write(f"A: {answer2}\n")

# Child-level similarity search
sub_docs2 = vectorstore.similarity_search(question2)
if sub_docs2:
    f.write("Similarity Search (Child-level):\n")
    for doc in sub_docs2:
        f.write(doc.page_content + "\n---\n")
else:
    f.write("No relevant child documents found.\n")

# Parent-level retrieval
parent_docs2 = retriever.invoke(question2)
if parent_docs2:
    f.write("Parent Document(s):\n")
    for doc in parent_docs2:
        f.write(doc.page_content + "\n---\n")
else:
    f.write("No relevant parent documents found.\n")

f.close()




