from dotenv import load_dotenv
from langgraph.graph import END, MessagesState
from langgraph.prebuilt import ToolNode

from react import llm, tools

load_dotenv()

SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer user questions.
"""


# define agent reasoning
def run_agent_reasoning(state: MessagesState) -> MessagesState:
    print("=============================== run_agent_reasoning ===============================")
    response = llm.invoke(
        [{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]]
    )

    print(state["messages"][-1].type)
    print(state["messages"][-1].content)

    # for msg in state["messages"]:
    #    print(f"Type: {msg.type} | Content: {msg.content}")
    
    return {"messages": [response]}


# define tool node
tool_node = ToolNode(tools)
