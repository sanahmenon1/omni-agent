import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent
from agents.models.openai_responses import OpenAIResponsesModel
from agents import WebSearchTool
from src.ideas.ideator_schemas import IdeasOutput

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env")

# OpenAI async client for the Agents SDK Responses model
async_client = AsyncOpenAI(api_key=api_key)

# Use the Responses API model that supports deep research + tools
responses_model = OpenAIResponsesModel(
    model=os.getenv("OPENAI_MODEL_IDEATION", "o3-deep-research"),
    openai_client=async_client,
)

# The ideation agent will call web search / code tools as needed
ideator_agent = Agent(
    name="Ideator Agent",
    model=responses_model,
    tools=[
        WebSearchTool(),
    ],
    instructions="""
    You generate omnichannel marketing ideas grounded in brand context and a campaign goal.
    Use web_search as needed to ground ideas with real moments, platforms, or proof points.

    Output spec (STRICT JSON matching IdeasOutput):
    - At least 18 ideas total (>=3 per category): Digital, Influencer, Events, Partnerships, PR, Community.
    - For each idea: category, title, concept, execution_notes (optional), sources (list of URLs).
    - campaign_intent should echo the goal.
    - brand_name should echo the brand.

    Rules:
    - Be specific and actionable (what, where, who, when).
    - Avoid fluff. Prefer concrete hooks (platforms, tentpoles, partner types).
    - If research is used, include a few URLs in 'sources'.
    - Return ONLY JSON. No prose outside JSON.
    """,
    output_type=IdeasOutput,
)
