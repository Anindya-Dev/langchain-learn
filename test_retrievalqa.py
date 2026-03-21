from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

llm=ChatGroq(
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

splitter= RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20)
chunks=splitter.split_text(document)

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore=Chroma.from_texts(chunks,embeddings)

qa_chain=RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k":2})
)

response=qa_chain.invoke({
    "query":"What is FastAPI used for?"
})
print(response['result'])