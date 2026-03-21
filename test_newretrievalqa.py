from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

document = """
FastAPI is a modern, fast web framework for building APIs with Python.
It supports async programming and auto-generates interactive documentation.
FastAPI uses Pydantic for data validation and SQLAlchemy for database operations.
JWT tokens are commonly used for authentication in FastAPI applications.
FastAPI is one of the fastest Python frameworks available.
"""

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_text(document)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_texts(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.
Context: {context}
Question: {question}
""")

def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

response = chain.invoke("What is FastAPI used for?")
print(response.content)