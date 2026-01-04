from pprint import pprint

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()


# Tool
@tool
def get_snp500_last_close():
    """Gets the last close price of the S&P 500"""
    print(
        "=============== Tool called : Getting S&P 500 last close price ==============="
    )
    return "S&P 500 last close was 4200"


# Agent
agent = create_agent(
    model=ChatOpenAI(model="gpt-3.5-turbo"),
    tools=[get_snp500_last_close],
    system_prompt="You are a financial analyst. Use the tools if you need to answer the question correctly.",
)

response = agent.invoke(
    {"messages": [("user", "What is the last close price of the S&P 500?")]}
)
pprint(response["messages"])

# tool should not be called
# response = agent.invoke({"messages":[("user", "What is the Nifty 50?")]})

# pprint(response["messages"])

# Graph
agent.get_graph().draw_mermaid_png(output_file_path="agent_02.png")
