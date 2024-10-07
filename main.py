from typing import List, TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, START
import functools
import asyncio

from agents.product_manager_agent import product_manager_node
from agents.context_agent import context_agent_node
from agents.code_analysis_agent import code_analysis_node
from agents.task_manager_agent import task_manager_node


class AgentState(TypedDict):
    user_input: str
    messages: List[str]
    next_agent: str
    code_context: str
    relevant_files: List[str]
    suggested_packages: List[str]
    tasks: List[str]


# Create the graph
workflow = StateGraph(AgentState)

# Define the nodes
workflow.add_node("ProductManager", product_manager_node)
workflow.add_node("Context", context_agent_node)
workflow.add_node("CodeAnalysis", code_analysis_node)
workflow.add_node("TaskManager", task_manager_node)


# Define the routing logic
def route_next_agent(state: AgentState) -> str:
    return state["next_agent"]


# Add conditional edges
workflow.add_conditional_edges(
    "ProductManager",
    route_next_agent,
    {
        "Context": "Context",
        "CodeAnalysis": "CodeAnalysis",
        "TaskManager": "TaskManager",
        "FINISH": END,
    },
)

workflow.add_edge("Context", "ProductManager")
workflow.add_edge("CodeAnalysis", "ProductManager")
workflow.add_edge("TaskManager", "ProductManager")

# Set the entry point
workflow.add_edge(START, "ProductManager")

# Compile the graph
project_management_team = workflow.compile()


# Function to initialize the chain state
def enter_chain(message: str, members: List[str]):
    return AgentState(
        user_input=message,
        messages=[HumanMessage(content=message)],
        next_agent="ProductManager",
        code_context="",
        relevant_files=[],
        suggested_packages=[],
        tasks=[],
        team_members=", ".join(members),
    )


# Create the full chain
project_management_chain = (
    functools.partial(enter_chain, members=workflow.nodes) | project_management_team
)


async def manage_project(user_input: str):
    """
    Run the project management team with a given user input.
    """
    result = await project_management_chain.ainvoke(user_input)
    print(result)


if __name__ == "__main__":
    user_input = input("Enter your request for the project management team: ")
    asyncio.run(manage_project(user_input))
