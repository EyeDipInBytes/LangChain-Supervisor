from langgraph.prebuilt import create_react_agent
from llm import llModel
from agents.tools import placeholder_tool

description = """You are a code extraction agent. 
Your task is to identify and extract any code snippets from the user's input and return them without any additional text or explanation."""

coder_agent = create_react_agent(
    model=llModel, tools=[placeholder_tool], state_modifier=description
)
