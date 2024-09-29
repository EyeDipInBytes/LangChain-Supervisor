from langgraph.prebuilt import create_react_agent
from llm import llModel
from agents.tools import tavily_tool

researcher_agent = create_react_agent(model=llModel, tools=[tavily_tool])
