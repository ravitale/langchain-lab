from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain.agents import create_agent

load_dotenv()

agent = create_agent(
    model=ChatOpenAI(gpt_model="gpt-3.5-turbo"),
    tools=[],
    system_prompt="what is the capital of France? ",
)

agent.get_graph().draw_mermaid_png(output_file_path="agent_01.png")
