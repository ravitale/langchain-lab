from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

agent=create_agent(
    model="gpt-4o-mini",
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
        model="gpt-4o-mini",
        trigger=("messages",5),
        keep=("messages",1)
        )
        ]
)

config={"configurable":{"thread_id":"thread_1"}}

question= [
    "Who is Scarlett Johansson?",
    "What is the summary of Scarlett Johansson?",
    "Where does Scarlett live",
    "What is Scarlett's best known for?"
    "What is Scarlett's age?"
    "What is Scarlett's height?"
    "Who is Monica Bellucci?"
    "What is Monica Bellucci's age?"
    "What is Monica Bellucci's height?"
    "What is Monica Bellucci's best known for?"
    "Where does Monica Bellucci live?"
    "Whos is Chanel Santini?"
    "What is Chanel Santini's age?"
    "What is Chanel Santini's height?"
    "What is Chanel Santini's best known for?"
    "Where does Chanel Santini live?"
    ]

for q in question:
    response = agent.invoke({"messages":[HumanMessage(content=q)]}, config)
    print(f"messages: {response}")
    print(f"messages: {len(response['messages'])}")