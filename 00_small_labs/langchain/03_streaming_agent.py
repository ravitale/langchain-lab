from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain.agents import create_agent

load_dotenv()

# create_agent can stream LLM output but cannot stream step-by-step graph events like LangGraph does.!!!!!!!!!!!

agent = create_agent(
    model=ChatOpenAI(model="gpt-3.5-turbo"),
    tools=[],
    system_prompt="You are a helpful assistant that can generate summaries of cities. ",
)

inputs = {
    "messages": [("user", "In about 100 words write an article about black holes")]
}
# response = agent.invoke({"messages":[("user", "What is the summary of New York City in 300 words?")]})

for chunk, metadata in agent.stream(inputs, stream_mode="messages"):

    # Check if the chunk contains text content (tokens)
    if chunk.content:
        # metadata['langgraph_node'] tells you if it's the 'agent' or a 'tool'
        print(chunk.content, end="", flush=True)


# Graph
agent.get_graph().draw_mermaid_png(output_file_path="agent_03.png")
