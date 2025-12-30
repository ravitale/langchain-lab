from typing import Dict, List

from pydantic import BaseModel, Field


class Source(BaseModel):
    """Schema for a source used by the agent."""

    url: str = Field(description="URL of the source")


class AgentResponse(BaseModel):
    """Schema for the agent response with answer and sources."""

    answer: str = Field(description="Agent's answer to the question")
    sources: List[Source] = Field(
        description="List of sources used by the agent", default_factory=list
    )
