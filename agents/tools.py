import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from langchain_community.agent_toolkits import FileManagementToolkit

work_dir = os.path.join(os.getcwd(), "workspace")
os.makedirs(work_dir, exist_ok=True)

tool_kit = FileManagementToolkit(
    root_dir=work_dir,
    selected_tools=["write_file"],
)


@tool
def placeholder_tool(input: str):
    """A placeholder tool that does nothing."""
    return input


write_file_tool = tool_kit.get_tools()[0]  # File write tool
tavily_tool = TavilySearchResults(max_results=5)  # Web search tool
python_repl_tool = PythonREPLTool()  # Python execution tool
empty_tool = placeholder_tool  # Empty tool
