from langchain.agents.format_scratchpad.log import format_log_to_str
from langchain.tools.render import render_text_description
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from typing import Union
from langchain.schema import AgentFinish, AgentAction

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """return the length of the text by character"""
    text = text.strip("'\n'").strip('"')
    return len(text)


def find_tool_by_name(tools: list[tool], tool_name: str) -> tool:
    print(f"find_tool_by_name is called with tool_name {tool_name}")
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool {tool_name} not found")


tools = [get_text_length]

template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
     
    Question: {input}
    Thought: {agent_scratchpad}
    """
prompt = PromptTemplate.from_template(template).partial(
    tools=render_text_description(tools),
    tool_names=", ".join([tool.name for tool in tools]),
)

llm = ChatOpenAI(model_name="gpt-4", temperature=0).bind(
    stop=["\nObservation", "Observation:", "Observation"]
)

intermediate_steps = []

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
    }
    | prompt
    | llm
    | ReActSingleInputOutputParser()
)

agent_steps = ""
while not isinstance(agent_steps, AgentFinish):
    agent_step: Union[AgentFinish, AgentAction] = agent.invoke(
    {
        "input": "What is the length of Moon in characters?",
        "agent_scratchpad": intermediate_steps,
    }
    )
    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input
        observation = tool_to_use.func(str(tool_input))
        print(f"observation is {observation}")
        intermediate_steps.append((agent_step, str(observation)))

 
if isinstance(agent_step, AgentFinish):
    print(agent_step.return_values)