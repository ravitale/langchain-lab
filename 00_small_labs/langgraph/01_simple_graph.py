import random
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


# State
class State(TypedDict):
    graph_info: str
    name: str


# randomly select cricket or football and return 0 or 1. 0 for cricket and 1 for football
def select_random_game(state: State) -> Literal["cricket", "football"]:
    print("=== Selecting game ===")
    game = random.randint(0, 1)
    if game == 0:
        print("Selected cricket")
        return "cricket"
    else:
        print("Selected football")
        return "football"


# Node
def start_play(state: State) -> State:
    print("=== Start playing ===")
    # update state
    state["graph_info"] += "\nStart playing "
    return state


# Node
def play_cricket(state: State) -> State:
    print("=== Start playing cricket ===")
    return {"name": state["name"]}


# Node
def play_football(state: State) -> State:
    print("=== Start playing football ===")
    return {"name": state["name"]}


# Graph
graph = StateGraph(State)
graph.add_node("start_play", start_play)
graph.add_node("cricket", play_cricket)
graph.add_node("football", play_football)

# Edges
graph.add_edge(START, "start_play")
graph.add_conditional_edges("start_play", select_random_game)
graph.add_edge("cricket", END)
graph.add_edge("football", END)

# compile graph
graph_builder = graph.compile()

# display graph
graph_builder.get_graph().draw_mermaid_png(output_file_path="graph_01.png")

# invoke graph
print(graph_builder.invoke({"graph_info": "Hello", "name": 123}))
