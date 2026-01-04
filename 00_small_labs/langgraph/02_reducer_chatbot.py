import os
import random
from pprint import pprint

# import reducers
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()


# State
class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# define the bot


def super_bot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# Graph
graph = StateGraph(State)

graph.add_node("SuperBot", super_bot)
graph.add_edge(START, "SuperBot")
graph.add_edge("SuperBot", END)

# compile graph
graph_builder = graph.compile()

# display graph
graph_builder.get_graph().draw_mermaid_png(output_file_path="graph_02.png")

# invoke graph
pprint(graph_builder.invoke({"messages": "Hello how many hours are in a day"}))
