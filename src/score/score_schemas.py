from typing import Annotated
from pydantic import BaseModel, Field, field_validator

# Reusable constrained float type
Score = Annotated[float, Field(ge=0, le=5)]

class IdeaScore(BaseModel):
    brand_fit: Score
    audience: Score
    resonance: Score
    virality: Score
    feasibility: Score
    rationale: str

    # Round scores to 2 decimal places after they've passed range validation
    @field_validator('brand_fit', 'audience', 'resonance', 'virality', 'feasibility', mode='after')
    def round_to_2dp(cls, v: float) -> float:
        return round(v, 2)
