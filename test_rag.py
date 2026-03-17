from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

llm=ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# Sample doc
document="""
FastAPI is a modern, fast web framework for building APIs with python. 
It is based on standard Python type hints and is very easy to learn. 
FastAPI automatically gemerates interactive API documentation. 
FastAPI supports async programming and is one of the fastest Python frameworks. 
FastAPI uses Pydantic for data validation. SQLAlchemy is used with FastAPI for database operations. 
JWT tokens are used for authentication in FastAPI applications.
"""

# Split document into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks= splitter.split_text(document)

print(f"Document split into {len(chunks)} chunks:")
for i, chunk in enumerate(chunks):
    print(f"Chunk{i+1}:{chunk}")

embeddings= HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore= Chroma.from_texts(chunks,embeddings)

retriever= vectorstore.as_retriever(search_kwargs={"k":2})

query="What is FastAPI used for?"
relevant_chunks= retriever.invoke(query)

for doc in relevant_chunks:
    print(doc.page_content)

prompt= ChatPromptTemplate.from_messages([
    ("system","""You are a helpful assistanat. Answer the question based only on the context provided below.
Context:{context}"""),
    ("human","{question}")
])

chain= prompt|llm

context="\n".join([doc.page_content for doc in relevant_chunks])
response= chain.invoke({
    "context":context,
    "question":"What is FastAPI used for?"
})

print(response.content)