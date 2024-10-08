import logging
from typing import List, TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, START
import functools
import asyncio

from agents.context_agent import context_agent_node
from agents.code_analysis_agent import code_analysis_node
from agents.task_manager_agent import task_manager_node
from agents.product_manager_agent import get_product_manager_agent


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
workflow.add_node("ContextAgent", context_agent_node)
workflow.add_node("CodeAnalysisAgent", code_analysis_node)
workflow.add_node("TaskManagerAgent", task_manager_node)
workflow.add_node("supervisor", get_product_manager_agent())


# Define the routing logic
def route_next_agent(state: AgentState) -> str:
    return state["next_agent"]


# Add conditional edges
workflow.add_conditional_edges(
    "supervisor",
    route_next_agent,
    {
        "ContextAgent": "ContextAgent",
        "CodeAnalysisAgent": "CodeAnalysisAgent",
        "TaskManagerAgent": "TaskManagerAgent",
        "FINISH": END,
    },
)

workflow.add_edge("ContextAgent", "supervisor")
workflow.add_edge("CodeAnalysisAgent", "supervisor")
workflow.add_edge("TaskManagerAgent", "supervisor")

# Set the entry point
workflow.add_edge(START, "supervisor")

# Compile the graph
project_management_team = workflow.compile()


# Function to initialize the chain state
def enter_chain(message: str, members: List[str]):
    return AgentState(
        user_input=message,
        messages=[HumanMessage(content=message)],
        next_agent="supervisor",
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
    logging.debug(f"Starting project management with user input: {user_input}")
    try:
        result = await project_management_chain.ainvoke(user_input)
        logging.debug(f"Project management result: {result}")
        print("Final result:", result)
    except Exception as e:
        logging.error(f"Error in manage_project: {str(e)}", exc_info=True)
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    user_input = input("Enter your request for the project management team: ")
    asyncio.run(manage_project(user_input))
