import os 
from agents import Agent
from dotenv import load_dotenv
from src.schemas import InputPayload

# Load environment variables from .env file
load_dotenv()

# Access your API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
   raise ValueError("OPENAI_API_KEY not found in .env file")

# Create an brand context extraction agent with structured output
context_extractor = Agent(
    name="Brand Context Extractor",
    instructions="""
        You are a parser. Extract brand context from the following dossier and campaign goal.

        ALWAYS return a valid JSON object matching InputPayload.

        Rules:
        - brand_name → exact brand name from dossier (e.g., "SKIMS")
        - brand_values → a list of 3–6 key values explicitly mentioned (e.g., inclusivity, luxury, comfort)
        - audiences → display all the audiences described; fill gender, geography, income, etc if possible
        - goal → must equal the CAMPAIGN GOAL section appended to input
        - constraints → if budget/timeline appear, set them; else leave empty

        If you can’t find something, return empty strings/lists. Do not hallucinate.
     """,
     output_type=InputPayload,
     model=os.getenv("OPENAI_MODEL_IDEATION", "gpt-4o-mini"),
)