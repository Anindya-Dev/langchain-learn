from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile" 
)

response= llm.invoke([HumanMessage(content="What is LangChain in one sentence?")])
print(response.content)
