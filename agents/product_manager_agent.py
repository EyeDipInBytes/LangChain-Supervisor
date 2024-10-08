from typing import List, TypedDict
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


def product_manager_supervisor(llm: ChatOpenAI, team_members: List[str]):
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


def get_product_manager_agent():
    return product_manager_supervisor(
        llModel, ["ContextAgent", "CodeAnalysisAgent", "TaskManagerAgent"]
    )
