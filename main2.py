import asyncio
import functools
import operator
import os
from typing import Annotated, List, TypedDict

from github import Github, GithubException
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from llm import llModel
from langgraph.graph import StateGraph, START, END
from agents.helpers import agent_node, create_team_supervisor
from langchain_core.tools import tool


# ResearchTeam graph state
class ResearchTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    # The team members are tracked so they are aware of
    # the others' skill-sets
    team_members: List[str]
    next: str


g = Github(os.environ["GITHUB_TOKEN"])


@tool
def fetch_repository(repo_name: str) -> str:
    """Fetches the repository information from GitHub."""
    repo = g.get_repo(repo_name)
    return f"${repo}"


@tool
def list_files(repo_name: str, path: str = "") -> str:
    """Lists files in the repository."""
    try:
        print(f"Fetching files for {repo_name} with path {path}")
        repo = g.get_repo(repo_name)
        contents = repo.get_contents(path)
        files = []
        for content_file in contents:
            if content_file.type == "dir":
                files.append(f"{content_file.path}/ (directory)")
            else:
                files.append(content_file.path)
        return (
            "\n".join(files)
            if files
            else "No files found in this repository or directory."
        )
    except GithubException as e:
        if e.status == 404:
            return f"Repository '{repo_name}' not found. Please check the repository name and try again."
        elif e.status == 403:
            return f"Access denied to repository '{repo_name}'. This may be due to repository privacy settings or API rate limits."
        else:
            return f"An error occurred while accessing the repository: {str(e)}"
    except Exception as e:
        print(e)
        return f"An unexpected error occurred: {str(e)}"


find_repo_agent = create_react_agent(model=llModel, tools=[fetch_repository])
find_repo_node = functools.partial(
    agent_node, agent=find_repo_agent, name="SearchRepository"
)

list_repo_files_agent = create_react_agent(model=llModel, tools=[list_files])
list_repo_files_node = functools.partial(
    agent_node, agent=list_repo_files_agent, name="ListRepoFiles"
)

supervisor_agent = create_team_supervisor(
    llModel,
    "You are a knowledgeable supervisor tasked with managing a conversation and a team of workers. "
    "Your team includes: SearchRepository and ListRepoFiles. "
    "You have broad knowledge and can often answer questions directly. "
    "Only use your team members when you need specific information you don't have. "
    "If you can answer the user's question without using any tools, do so directly. "
    "If you need to use a tool, specify which one and why. "
    "When all necessary information is gathered, provide a final answer to the user. "
    "Always provide a response to the user, even for simple greetings or questions. "
    "Your primary goal is to guide the conversation towards repository-related topics. "
    "Whenever appropriate, encourage the user to ask questions about repositories or to explore repository contents. "
    "If the user's query is not related to repositories, try to steer the conversation in that direction by:"
    "1. Answering their question briefly, then asking if they'd like to know about any repositories related to the topic."
    "2. Suggesting they might find interesting information in certain repositories."
    "3. Asking if they'd like to see a list of files in a relevant repository."
    "4. Mentioning interesting facts about repositories that might pique their curiosity."
    "Remember to be subtle and natural in your guidance, maintaining a helpful and friendly tone throughout the conversation.",
    ["SearchRepository", "ListRepoFiles"],
)


research_graph = StateGraph(ResearchTeamState)
research_graph.add_node("SearchRepository", find_repo_node)
research_graph.add_node("ListRepoFiles", list_repo_files_node)
research_graph.add_node("supervisor", supervisor_agent)

# Define the control flow
research_graph.add_edge("SearchRepository", "supervisor")
research_graph.add_edge("ListRepoFiles", "supervisor")
research_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "SearchRepository": "SearchRepository",
        "ListRepoFiles": "ListRepoFiles",
        "FINISH": END,
        "WAIT_FOR_INPUT": END,
    },
)

research_graph.add_edge(START, "supervisor")
chain = research_graph.compile()


# The following functions interoperate between the top level graph state
# and the state of the research sub-graph
# this makes it so that the states of each graph don't get intermixed
def enter_chain(message: str):
    results = {
        "messages": [HumanMessage(content=message)],
    }
    return results


research_chain = enter_chain | chain


async def process_input(user_input):
    async for step in research_chain.astream(user_input, {"recursion_limit": 100}):
        if "__end__" not in step:
            if "supervisor" in step:
                response = step["supervisor"]["messages"][-1].content
                print("AI:", response)
                if step["supervisor"]["next"] == "WAIT_FOR_INPUT":
                    return True  # Indicates that we need more user input
            elif "SearchRepository" in step or "ListRepoFiles" in step:
                agent_name = (
                    "SearchRepository"
                    if "SearchRepository" in step
                    else "ListRepoFiles"
                )
                print(f"{agent_name}:", step[agent_name]["messages"][-1].content)
            else:
                # Print other steps for debugging (optional)
                print("Step:", step)
        print("---")
    return False  # Indicates that the conversation can continue without immediate user input


async def main():
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("AI: Goodbye!")
            break
        await process_input(user_input)


if __name__ == "__main__":
    asyncio.run(main())
