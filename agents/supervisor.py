from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel
from typing import Literal

from llm import llModel

members = ["Researcher", "Coder", "FileManager", "Tester"]
system_prompt = (
    "You are a supervisor tasked with managing a project creation process between the "
    "following workers: {members}. Given the user request to build a project, "
    "coordinate the workers to research, code, manage files, and test the project. "
    "Respond with the worker to act next. When the project is complete and tested, "
    "respond with FINISH."
)
options = ["FINISH"] + members


class routeResponse(BaseModel):
    next: Literal["FINISH", "Researcher", "Coder", "FileManager", "Tester"]


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

supervisor_agent = prompt | llModel.with_structured_output(routeResponse)
