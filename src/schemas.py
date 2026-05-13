from typing import List, Literal, Optional
from pydantic import BaseModel, Field

class StrategySelection(BaseModel):
    strategy: Literal["academic", "social_trending", "general_web"] = Field(
        description="The selected research strategy based on the query."
    )
    reasoning: str = Field(
        description="Explanation for why this strategy was selected."
    )

class SubQuery(BaseModel):
    query: str = Field(description="A specific search query string.")
    persona: str = Field(description="The persona or perspective this query represents (e.g., 'skeptic', 'academic researcher', 'tech enthusiast').")

class PlanningOutput(BaseModel):
    clarified_query: str = Field(
        description="The original user query rewritten to be clear, unambiguous, and suitable for deep research."
    )
    strategy: str = Field(description="The selected research strategy (academic, social_trending, general_web).")
    sub_queries: List[SubQuery] = Field(
        description="A list of generated sub-queries to gather comprehensive information from different perspectives."
    )
