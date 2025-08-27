import os 
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access your API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
   raise ValueError("OPENAI_API_KEY not found in .env file")


# Define the constraints for the brand context + campaign goal
class Constraints(BaseModel):
    budget: Optional[str]
    timeline: Optional[str]


class Audiences(BaseModel):
    name: str
    age_group: str
    gender_distribution: str
    geography: str 
    income_level: str
    psycographics: str
    behaviors: List[str]
    pain_points: List[str]
    motivations: List[str]
    purchase_drivers: List[str]
    preferred_channels: List[str]

# All the brand name, values, audience, goals, and contraints (optional)
class InputPayload(BaseModel):
    brand_name: str
    brand_values: List[str]
    audiences: List[Audiences]
    goal: str
    constraints: Optional[Constraints]