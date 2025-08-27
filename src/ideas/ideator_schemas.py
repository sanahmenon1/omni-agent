from typing import List, Literal
from pydantic import BaseModel, Field

IdeaCategory = Literal["Digital", "Influencer", "Events", "Partnerships", "PR", "Community"]

class IdeaItem(BaseModel):
    category: IdeaCategory
    title: str
    concept: str
    execution_notes: str
    sources: List[str] = Field(default_factory=list)

class IdeasOutput(BaseModel):
    campaign_intent: str
    brand_name: str
    ideas: List[IdeaItem]
    total_ideas: int = Field(ge=18)