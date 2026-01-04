from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from langchain.agents import create_agent

load_dotenv()


class Actor(BaseModel):
    name: str = Field(description="Name of the actor")
    age: int = Field(description="Age of the actor")
    country: str = Field(description="Country of the actor")
    best_known_for: str = Field(description="Actor best known for")


class Movie(BaseModel):
    name: str = Field(description="Name of the movie")
    release_year: int = Field(description="Release year of the movie")
    director: str = Field(description="Director of the movie")
    actors: list[str] = Field(description="Actors in the movie")


model = ChatOpenAI(model="gpt-3.5-turbo")
model_with_structure = model.with_structured_output(Movie, include_raw=True)
# print("Unstructured output is -------------------------------------------")
result = model.invoke("Information about Matrix?")
# print(result.content)
print("Structured output is -------------------------------------------")
structured_result = model_with_structure.invoke("Information about Matrix?")
print(structured_result)
