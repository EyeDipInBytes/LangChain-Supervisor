from typing import TypedDict, List, Annotated
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import MessagesPlaceholder
from agents.helpers import agent_node
from llm import llModel
import os
from github import Github
import operator
import functools

g = Github(os.environ["GITHUB_TOKEN"])


class AgentState(TypedDict):
    user_input: str
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: str
    next_agent: str
    code_context: str
    relevant_files: List[str]
    suggested_packages: List[str]
    tasks: List[str]


# Tool functions
def fetch_repository(repo_name: str) -> str:
    """Fetches the repository information from GitHub."""
    repo = g.get_repo(repo_name)
    return f"Repository '{repo.name}' fetched. Description: {repo.description}"


def list_files(repo_name: str, path: str = "") -> List[str]:
    """Lists files in the repository."""
    repo = g.get_repo(repo_name)
    contents = repo.get_contents(path)
    files = []
    for content_file in contents:
        if content_file.type == "dir":
            files.extend(list_files(repo_name, content_file.path))
        else:
            files.append(content_file.path)
    return files


def get_file_content(repo_name: str, file_path: str) -> str:
    """Gets the content of a specific file."""
    repo = g.get_repo(repo_name)
    content_file = repo.get_contents(file_path)
    return content_file.decoded_content.decode("utf-8")


# Create tools
tools = [
    Tool(
        name="FetchRepository",
        func=fetch_repository,
        description="Fetches the repository information from GitHub.",
    ),
    Tool(
        name="ListFiles",
        func=list_files,
        description="Lists files in the repository.",
    ),
    Tool(
        name="GetFileContent",
        func=get_file_content,
        description="Gets the content of a specific file.",
    ),
]

# Create React agent
repo_inference_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant that extracts GitHub repository names from user inputs. Respond with only the repository name in the format 'username/repo'.",
        ),
        ("human", "{user_input}"),
    ]
)

repo_inference_chain = repo_inference_prompt | llModel

context_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Context Agent responsible for analyzing codebases and providing relevant information. "
            "You have access to the following tools:\n\n{tools}\n\n"
            "Use the following format:\n\n"
            "Question: the input question you must answer\n"
            "Thought: you should always think about what to do\n"
            "Action: the action to take, should be one of [{tool_names}]\n"
            "Action Input: the input to the action\n"
            "Observation: the result of the action\n"
            "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
            "Thought: I now know the final answer\n"
            "Final Answer: the final answer to the original input question",
        ),
        ("human", "Analyze the following repository and provide context: {repo_name}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

context_agent_chain = create_react_agent(
    model=llModel, tools=tools, state_modifier=context_agent_prompt
)


def prelude(state: AgentState) -> AgentState:
    user_input = state.get("user_input", "")
    repo_name = repo_inference_chain.invoke({"user_input": user_input})
    repo_name = repo_name.strip()

    if not repo_name or "/" not in repo_name:
        return {
            **state,
            "messages": state["messages"]
            + [
                AIMessage(
                    content="I couldn't identify a GitHub repository in your request. Could you please provide the repository name in the format 'username/repo'?",
                    name="ContextAgent",
                )
            ],
            "next_agent": "ProductManager",
        }

    return {**state, "repo_name": repo_name}


context_aware_agent = prelude | context_agent_chain

context_agent_node = functools.partial(
    agent_node, agent=context_aware_agent, name="ContextAgent"
)


async def context_agent(state: AgentState) -> AgentState:
    try:
        result = await context_agent_node(state)

        state["code_context"] = result.get("output", "")
        state["relevant_files"] = result.get("relevant_files", [])
        state["next_agent"] = "ProductManager"

        analysis_message = f"""
        Code Analysis Results for {state.get('repo_name', 'Unknown Repository')}:
        {state['code_context']}
        Relevant Files: {', '.join(state['relevant_files'])}
        """
        state["messages"].append(
            AIMessage(content=analysis_message, name="ContextAgent")
        )

    except Exception as e:
        error_message = f"An error occurred while analyzing the repository: {str(e)}"
        state["messages"].append(AIMessage(content=error_message, name="ContextAgent"))
        state["next_agent"] = "ProductManager"

    return state
