from dotenv import load_dotenv
from duckduckgo_search import DDGS
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import Tool

# Load OpenAI API key from .env
load_dotenv()


# -------------------------
# 1️⃣ Define DuckDuckGo tool
# -------------------------
def duckduckgo_search(query: str) -> str:
    """Perform a DuckDuckGo search and return top 5 results."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    return "\n".join(r["body"] for r in results)


tools = [
    Tool(
        name="DuckDuckGoSearch",
        func=duckduckgo_search,
        description="Performs live web searches using DuckDuckGo",
    )
]
# -------------------------
# 2️⃣ Initialize LLM
# -------------------------
# Classic ReAct works best with deterministic output
# You can use ChatOpenAI (gpt-4) or OpenAI (text-davinci-003)
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# -------------------------
# 3️⃣ Define classic ReAct prompt
# -------------------------
react_prompt = PromptTemplate(
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
    template="""
You are a ReAct agent.

You MUST follow this loop exactly:
Thought → Action → Action Input → Observation

Rules:
- Do NOT answer the question until you have received at least one Observation.
- After an Action, STOP and wait for the Observation.
- You may only use information that appears in Observations.
- Never skip the Observation step.

Available tools:
{tools}

Valid actions are: [{tool_names}]

Format your response exactly like this:

Question: {input}
Thought: <reasoning>
Action: <one of [{tool_names}]>
Action Input: <input>
Observation: <this will be provided to you>
{agent_scratchpad}
Thought: I now know the final answer
Final Answer: <answer derived ONLY from observations>
""",
)

# -------------------------
# 4️⃣ Create ReAct agent
# -------------------------
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

# -------------------------
# 5️⃣ AgentExecutor
# -------------------------
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # prints Thought → Action → Observation trace
)

# Alias for convenience
chain = agent_executor


# -------------------------
# 6️⃣ Main function
# -------------------------
def main():
    print("Hello from classic 0.1.0 ReAct agent!")
    query = "Addition of 2 and 2 is it 4 ?"
    result = chain.invoke(input={"input": query})
    print("\nFinal Answer:", result)


# -------------------------
# 7️⃣ Entry point
# -------------------------
if __name__ == "__main__":
    main()
