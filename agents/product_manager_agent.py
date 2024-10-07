from typing import List, Dict, TypedDict
from langchain_openai import ChatOpenAI
from llm import llModel
from agents.helpers import create_team_supervisor


class AgentState(TypedDict):
    user_input: str
    messages: List[str]
    next_agent: str
    code_context: str
    relevant_files: List[str]
    suggested_packages: List[str]
    tasks: List[str]


def create_product_manager_supervisor(llm: ChatOpenAI, team_members: List[str]):
    """Creates a Product Manager supervisor for routing and analysis."""
    return create_team_supervisor(
        llm,
        "You are a Product Manager overseeing the project. You are tasked with managing a conversation between the"
        " following team members: {team_members}. Given the current project state and user request,"
        " respond with the team member to act next. Each team member will perform a"
        " task and respond with their results and status. When the project is complete,"
        " respond with FINISH.",
        team_members,
    )


def product_manager_node(state: Dict) -> Dict:
    """Product Manager Node that coordinates the project flow."""
    team_members = [
        "Context",
        "CodeAnalysis",
        "TaskManager",
    ]

    supervisor = create_product_manager_supervisor(llModel, team_members)

    result = supervisor.invoke(
        {"messages": state["messages"] + [f"Current State: {state}"]}
    )

    # Update state based on the result
    state["next_agent"] = result["next"]
    state["messages"].append(result["messages"][-1])

    return state
