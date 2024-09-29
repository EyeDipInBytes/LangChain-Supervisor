from langgraph.prebuilt import create_react_agent
from llm import llModel
from agents.tools import python_repl_tool, read_file_tool

description = """You are a testing agent.
Your task is to test the Python code created by the Coder agent.
You can read files and execute Python code to verify its functionality.
Report any issues found and suggest improvements.
Remember that all files are located in the 'workspace' directory."""

tester_agent = create_react_agent(
    model=llModel, tools=[python_repl_tool, read_file_tool], state_modifier=description
)
