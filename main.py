import functools
import asyncio
import operator
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.messages import BaseMessage

from langgraph.graph import END, StateGraph, START

from agents.file_manager import file_manager
from agents.helpers import agent_node
from agents.researcher import researcher_agent
from agents.coder import coder_agent
from agents.tester import tester_agent
from agents.supervisor import supervisor_agent
from agents.supervisor import members


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


research_node = functools.partial(agent_node, agent=researcher_agent, name="Researcher")
code_node = functools.partial(agent_node, agent=coder_agent, name="Coder")
file_manager_node = functools.partial(
    agent_node, agent=file_manager, name="FileManager"
)
tester_node = functools.partial(agent_node, agent=tester_agent, name="Tester")

workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("FileManager", file_manager_node)
workflow.add_node("Tester", tester_node)
workflow.add_node("supervisor", supervisor_agent)

# Researcher -> Supervisor
workflow.add_edge("Researcher", "supervisor")

# Coder -> Supervisor
workflow.add_edge("Coder", "supervisor")

# FileManager -> Coder
workflow.add_edge("FileManager", "Coder")

# Tester -> Supervisor
workflow.add_edge("Tester", "supervisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
workflow.add_edge(START, "supervisor")

graph = workflow.compile()


async def main():
    user_input = input(
        "Please enter your project idea (e.g., 'build me a tic tac toe game with a basic UI'): "
    )
    async for s in graph.astream({"messages": [HumanMessage(content=user_input)]}):
        if "__end__" not in s:
            print(s)
            print("----")
    print("Project completed. Check the workspace directory for the generated files.")


if __name__ == "__main__":
    asyncio.run(main())
