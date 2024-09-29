from langgraph.prebuilt import create_react_agent
from llm import llModel
from agents.tools import write_file_tool, read_file_tool

description = """You are a file manager that can read and write files in the file system.
You know which file to create or read based on the user request and the current project structure."""

file_manager = create_react_agent(
    model=llModel, tools=[write_file_tool, read_file_tool], state_modifier=description
)
