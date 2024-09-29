import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain_community.agent_toolkits import FileManagementToolkit

work_dir = os.path.join(os.getcwd(), "workspace")
os.makedirs(work_dir, exist_ok=True)

tool_kit = FileManagementToolkit(
    root_dir=work_dir,
    selected_tools=["write_file", "read_file"],
)

write_file_tool = tool_kit.get_tools()[0]
read_file_tool = tool_kit.get_tools()[1]
tavily_tool = TavilySearchResults(max_results=5)  # Web search tool


class WorkspacePythonREPLTool(PythonREPLTool):
    def _run(self, command: str) -> str:
        original_dir = os.getcwd()
        try:
            os.chdir(work_dir)
            return super()._run(command)
        finally:
            os.chdir(original_dir)


python_repl_tool = WorkspacePythonREPLTool()  # Python execution tool
