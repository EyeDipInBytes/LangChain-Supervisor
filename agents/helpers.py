from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from llm import llModel

trimmer = trim_messages(
    max_tokens=100000,
    strategy="last",
    token_counter=llModel,
    include_system=True,
)


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {
        "messages": [HumanMessage(content=result["messages"][-1].content, name=name)]
    }


def create_team_supervisor(llm: ChatOpenAI, system_prompt, members) -> str:
    """An LLM-based router with response generation."""
    options = ["FINISH", "WAIT_FOR_INPUT"] + members
    function_def = {
        "name": "route",
        "description": "Select the next action and provide a response.",
        "parameters": {
            "type": "object",
            "properties": {
                "next": {
                    "type": "string",
                    "enum": options,
                    "description": "The next action to take. Use WAIT_FOR_INPUT if user input is required.",
                },
                "response": {
                    "type": "string",
                    "description": "The response to the user's input or query.",
                },
                "agent_input": {
                    "type": "string",
                    "description": "The specific input to send to the next agent, if applicable.",
                },
            },
            "required": ["next", "response"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, provide a response to the user and decide the next action."
                " If you need more information from the user, use WAIT_FOR_INPUT."
                " If you need to use another agent, specify which one and provide a concise, targeted input for that agent."
                " Otherwise, select FINISH if the task is complete."
                " Always provide a response, even for simple greetings or questions.",
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(members))

    def add_response_to_messages(result):
        return {
            "messages": [AIMessage(content=result["response"])],
            "next": result["next"],
            "agent_input": result.get("agent_input", ""),
        }

    return (
        prompt
        | trimmer
        | llm.bind(functions=[function_def], function_call={"name": "route"})
        | JsonOutputFunctionsParser()
        | add_response_to_messages
    )
