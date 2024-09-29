import functools
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

workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("FileManager", file_manager_node)
workflow.add_node("supervisor", supervisor_agent)


for member in members:
    # Workers always report back to the supervisor
    workflow.add_edge(member, "supervisor")
# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
workflow.add_edge(START, "supervisor")

graph = workflow.compile()


def main():
    user_input = input("Please enter your question: ")
    for s in graph.stream({"messages": [HumanMessage(content=user_input)]}):
        if "__end__" not in s:
            print(s)
            print("----")


if __name__ == "__main__":
    main()
