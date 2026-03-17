from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

prompt= ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant that only answers questions about Python programming. If asked anything else, politely decline."),
    ("human","{input}")
])

chain= prompt| llm

# response =chain.invoke({"input": "What is a list Comprehension in Python"})
# print(response.content)

response2=chain.invoke({"input":"What is the capital of France?"})
print(response2.content)