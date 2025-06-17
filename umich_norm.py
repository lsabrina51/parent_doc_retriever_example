# Sample Q&A RAG application over a text data source
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

#logging
import logging

logging.basicConfig(
    filename='nlog.txt',       # Log output file
    filemode='w',             # Overwrite the file each run; use 'a' to append
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log message format
    level=logging.INFO        # Minimum level to log
)

logging.getLogger("langchain.retrievers").setLevel(logging.INFO)

#for loader
from langchain.retrievers.multi_query import MultiQueryRetriever

# Sets the current working directory to be the same as the file.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load environment file for secrets.
try:
    if load_dotenv('.env', override=True) is False:
        raise TypeError
except TypeError:
    print('Unable to load .env file.')
    quit()

endpoint = os.environ['AZURE_OPENAI_ENDPOINT']

#Remove OPENAI_API_BASE from the environment variables as this causes an error with AzureOpenAIEmbeddings
# if "OPENAI_API_BASE" in os.environ:
#     del os.environ["OPENAI_API_BASE"]

# Define llm parameters
llm = AzureChatOpenAI(
    deployment_name=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
    openai_api_version=os.environ['AZURE_OPENAI_API_VERSION'],
    openai_api_key=os.environ['AZURE_OPENAI_API_KEY'],
    azure_endpoint=endpoint,
    openai_organization=os.environ['AZURE_OPENAI_ORGANIZATION']
    )

# Replace with the document(s) you wish to use
print("Loading document...")
loader = PyPDFLoader("umich-example.pdf")

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print("Document loaded.")

# Settings for embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=endpoint, 
    openai_api_version=os.environ['AZURE_OPENAI_API_VERSION'],  
    openai_api_key=os.environ['AZURE_OPENAI_API_KEY'],   
    openai_organization=os.environ['AZURE_OPENAI_ORGANIZATION'],
    model="text-embedding-3-small" 
)

print("Embedding documents...")
# Create the vectorstore
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

print("Documents embedded.")
# Retrieve and generate using the relevant snippets of the data
retriever = vectorstore.as_retriever(search_kwargs={"k": min(4, len(splits))}) # 4 is default k, ensures we aren't indexing greater than num elts available


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

##start of testing q's---------------------
# Assuming you're writing to a file
with open("umich_output_norm.txt", "w") as f:
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

    f.write("="*80 + "\n")


#end of testing q's----------------------

# # User requests
# while True:
#     text = input('Enter your query (Example: How many undergrad students are at U of M?): --> ') # Example: How many undergrad students are at U of M?
#     #call
#     print(conversational_rag_chain.invoke(
#         {"input": text},
#         config={"configurable": {"session_id": "0"}},
#     )["answer"])