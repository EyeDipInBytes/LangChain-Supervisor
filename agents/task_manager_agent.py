from typing import TypedDict, List, Annotated
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.prebuilt import create_react_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from agents.helpers import agent_node
from llm import llModel
import operator
import functools


class AgentState(TypedDict):
    user_input: str
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: str
    next_agent: str
    code_context: str
    relevant_files: List[str]
    suggested_packages: List[str]
    tasks: List[str]


def check_existing_tasks(tasks: List[str], user_request: str) -> str:
    matching_task = next((task for task in tasks if user_request in task), None)
    if matching_task:
        return f"Matching task found: {matching_task}"
    return "No matching task found."


def add_new_task(tasks: List[str], new_task: str) -> str:
    tasks.append(new_task)
    return f"New task added: {new_task}"


def update_task_status(tasks: List[str], task: str, new_status: str) -> str:
    if task in tasks:
        return f"Status of task '{task}' updated to '{new_status}'"
    return f"Task '{task}' not found."


tools = [
    Tool(
        name="CheckExistingTasks",
        func=check_existing_tasks,
        description="Checks if a matching task already exists in the task list.",
    ),
    Tool(
        name="AddNewTask",
        func=add_new_task,
        description="Adds a new task to the task list.",
    ),
    Tool(
        name="UpdateTaskStatus",
        func=update_task_status,
        description="Updates the status of a specific task.",
    ),
]

task_manager_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Task Manager Agent responsible for managing the task list or Kanban board. "
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
        (
            "human",
            "Manage the task list based on the following user request and current tasks:\n"
            "User Request: {user_request}\n"
            "Current Tasks: {tasks}\n\n"
            "1. Check if a matching task already exists\n"
            "2. If no matching task exists, add a new task\n"
            "3. Update task statuses if necessary\n"
            "4. Provide a summary of actions taken",
        ),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

task_manager_chain = create_react_agent(
    model=llModel, tools=tools, state_modifier=task_manager_prompt
)


def prelude(state: AgentState) -> AgentState:
    user_request = state.get("user_input", "")
    tasks = state.get("tasks", [])
    return {
        **state,
        "user_request": user_request,
        "tasks": tasks,
        "tools": tools,
        "tool_names": ", ".join([tool.name for tool in tools]),
        "agent_scratchpad": "",
    }


task_aware_agent = prelude | task_manager_chain

task_manager_node = functools.partial(
    agent_node, agent=task_aware_agent, name="TaskManagerAgent"
)


async def task_manager_agent(state: AgentState) -> AgentState:
    try:
        result = await task_manager_node(state)

        task_management_message = f"""
        Task Management Results:
        {result.get('output', '')}
        """
        state["messages"].append(
            AIMessage(content=task_management_message, name="TaskManagerAgent")
        )
        state["next_agent"] = "ProductManager"

        # Update tasks in the state
        # Note: This assumes that the tools have modified the tasks list in-place
        # You might need to adjust this based on how your agent returns the updated tasks
        state["tasks"] = result.get("tasks", state["tasks"])

    except Exception as e:
        error_message = f"An error occurred during task management: {str(e)}"
        state["messages"].append(
            AIMessage(content=error_message, name="TaskManagerAgent")
        )
        state["next_agent"] = "ProductManager"

    return state
