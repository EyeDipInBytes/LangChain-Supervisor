from langgraph.prebuilt import create_react_agent
from llm import llModel
from agents.tools import write_file_tool

description = """You are a file manager that can write files to the file system. 
You know which file to create based on the user request, for example, 
if the user wants to create a TypeScript (.ts) or Python (.py) file."""

file_manager = create_react_agent(
    model=llModel, tools=[write_file_tool], state_modifier=description
)
