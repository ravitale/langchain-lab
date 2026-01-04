import os
import random
from pprint import pprint
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.utils import AnyMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


# state class with list of Messages
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# create an add tool for adding two numbers and create a llm with this tool and invoke it

initial_messages = [
    AIMessage(
        content=f"Please tell me how can I help you today?", name="personal_assistant"
    )
]
initial_messages.append(
    HumanMessage(content="I wish to know something about India", name="the_boss")
)
initial_messages.append(
    AIMessage(
        content="What do you want to know about India?", name="personal_assistant"
    )
)


@tool
def add(a, b) -> int:
    """Performs addition of two numbers a and b"""
    return a + b


def super_bot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


tool_list = [add]
llm = ChatOpenAI(model_name="gpt-4o", temperature=0).bind_tools(tool_list)

# graph
graph = StateGraph(State)

graph.add_node("SuperBot", super_bot)
graph.add_node("tools", ToolNode(tool_list))

graph.add_edge(START, "SuperBot")
graph.add_conditional_edges("SuperBot", tools_condition)
graph.add_edge("tools", "SuperBot")
graph.add_edge("SuperBot", END)

# compile graph
graph_builder = graph.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "1"}}

# invoke graph
graph_builder.invoke({"messages": initial_messages}, config=config)


# for chunk in graph_builder.stream(
#     {"messages": "In less that 200 words tell me about its weather and seasons"},
#     config,
#     stream_mode="updates"
# ):
#     print(chunk)
for chunk in graph_builder.stream(
    {"messages": "In less that 200 words tell me about its weather and seasons"},
    config,
    stream_mode="values"
):
    print(chunk)
