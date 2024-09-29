from langgraph.prebuilt import create_react_agent
from llm import llModel
from agents.tools import python_repl_tool, write_file_tool

description = """You are a coding agent. 
Your task is to write, execute, and debug Python code based on the user's requirements.
You can create new files and write code to them. Always use the write_file_tool to create or update files.
When writing code, consider best practices, error handling, and code organization."""

coder_agent = create_react_agent(
    model=llModel, tools=[python_repl_tool, write_file_tool], state_modifier=description
)
