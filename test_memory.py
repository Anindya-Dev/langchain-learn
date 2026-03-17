from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

llm= ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

conversation_history=[]

def chat(user_input):
    conversation_history.append(HumanMessage(content=user_input))
    response=llm.invoke(conversation_history)
    conversation_history.append(AIMessage(content=response.content))
    return response.content

print(chat("My name is Anindya and I am learning AI engineering"))
print(chat("What is my name?"))
print(chat("What am I learning"))
