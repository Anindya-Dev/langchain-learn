from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent,AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

llm=ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

@tool
def add_numbers(a:float,b:float)->float:
    """Add two numbers together"""
    return a+b
@tool
def multiply_numbers(a:float,b:float)->float:
    """Multiply two numbers together"""
    return a*b
@tool
def calculate_percentage(value: float, percentage: float) -> float:
    """Calculate what percentage of a value is"""
    return (value/100)*percentage

tools=[add_numbers,multiply_numbers,calculate_percentage]

prompt=ChatPromptTemplate.from_messages([
    ("system","You are a helpgul assistant with access to math tools."),
    ("human","{input}"),
    ("placeholder","{agent_scratchpad}")
])

agent=create_tool_calling_agent(llm,tools,prompt)
executor=AgentExecutor(agent=agent, tools=tools, verbose=True)

response=executor.invoke({"input":"What is 20 percentage of 350, then add 15 to it?"})
response['output']