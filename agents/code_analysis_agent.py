from typing import TypedDict, List, Annotated
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain.agents import create_react_agent
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


# Tool functions
def analyze_implementation(code_context: str, user_request: str) -> str:
    """Provides step-by-step guidance on code changes."""
    prompt = f"""
    Based on the following code context and user request:
    Code Context: {code_context}
    User Request: {user_request}

    Provide a step-by-step guide on how to implement the required changes.
    Break down complex tasks into smaller, manageable steps.
    """
    return llModel(prompt)


def identify_files(code_context: str, user_request: str) -> str:
    """Specifies which files need modification."""
    prompt = f"""
    Based on the following code context and user request:
    Code Context: {code_context}
    User Request: {user_request}

    Specify which files need to be modified, added, or deleted.
    If new files are needed, suggest appropriate names and locations.
    """
    return llModel(prompt)


def explain_rationale(code_context: str, user_request: str) -> str:
    """Explains why changes are necessary and the approach taken."""
    prompt = f"""
    Based on the following code context and user request:
    Code Context: {code_context}
    User Request: {user_request}

    Explain why each change is necessary.
    Describe the reasoning behind the chosen approach.
    Highlight any potential impacts or considerations for each change.
    """
    return llModel(prompt)


# Create tools
tools = [
    Tool(
        name="AnalyzeImplementation",
        func=analyze_implementation,
        description="Provides step-by-step guidance on code changes.",
    ),
    Tool(
        name="IdentifyFiles",
        func=identify_files,
        description="Specifies which files need modification.",
    ),
    Tool(
        name="ExplainRationale",
        func=explain_rationale,
        description="Explains why changes are necessary and the approach taken.",
    ),
]

code_analysis_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized Code Analysis Agent responsible for analyzing code and providing implementation plans. "
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
            "Analyze the following code context and user request:\n"
            "Code Context: {code_context}\n"
            "User Request: {user_request}\n\n"
            "Provide a detailed analysis and implementation plan including:\n"
            "1. Step-by-step implementation guidance\n"
            "2. File identification for modifications\n"
            "3. Rationale explanation for changes",
        ),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

code_analysis_chain = create_react_agent(
    llm=llModel, tools=tools, prompt=code_analysis_prompt
)


def prelude(state: AgentState) -> AgentState:
    code_context = state.get("code_context", "")
    user_request = next(
        (msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)), ""
    )
    return {
        **state,
        "code_context": code_context,
        "user_request": user_request,
        "tools": tools,
        "tool_names": ", ".join([tool.name for tool in tools]),
        "agent_scratchpad": "",
    }


code_aware_agent = prelude | code_analysis_chain

code_analysis_node = functools.partial(
    agent_node, agent=code_aware_agent, name="CodeAnalysisAgent"
)


async def code_analysis_agent(state: AgentState) -> AgentState:
    try:
        result = await code_analysis_node(state)

        analysis_message = f"""
        Code Analysis Results:
        {result.get('output', '')}
        """
        state["messages"].append(
            AIMessage(content=analysis_message, name="CodeAnalysisAgent")
        )
        state["next_agent"] = "ProductManager"

    except Exception as e:
        error_message = f"An error occurred during code analysis: {str(e)}"
        state["messages"].append(
            AIMessage(content=error_message, name="CodeAnalysisAgent")
        )
        state["next_agent"] = "ProductManager"

    return state
