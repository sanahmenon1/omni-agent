from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

CampaignIntent = Literal["Awareness", "Launch", "Retention", "Virality", "Exclusivity"]

class Constraints(BaseModel):
    budget: Optional[Literal["low", "medium", "high"]] = None
    timeline: Optional[Literal["short", "long"]] = None

class InputPayload(BaseModel):
    brand_name: str
    brand_values: List[str]
    audience: str
    campaign_intent: CampaignIntent
    constraints: Optional[Constraints] = None

class Scores(BaseModel):
    brand_fit: int = Field(ge=1, le=5)
    audience_resonance: int = Field(ge=1, le=5)
    virality: int = Field(ge=1, le=5)
    feasibility: int = Field(ge=1, le=5)

class Idea(BaseModel):
    category: Literal["Digital","Influencers","Events","Partnerships","PR_Stunts","Community_Plays"]
    idea: str

class OutputPayload(BaseModel):
    campaign_intent: CampaignIntent
    brand_context_echo: dict
    ideas: List[Idea]
