
import asyncio
import os
from agents import Runner
from dotenv import load_dotenv
from src.context_extractor import context_extractor
from src.schemas import InputPayload

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# 1) Return the structured object directly
async def process_context(brand_context: str) -> InputPayload:
    runner = Runner()
    run_result = await runner.run(context_extractor, brand_context)
    return run_result.final_output   # <- IMPORTANT

# 2) structure_output now ONLY prints; it DOES NOT call process_context
async def structure_output(result: InputPayload) -> None:
    print(f"Brand Name: {result.brand_name}")

    print("\nBrand Values:")
    for value in result.brand_values:
        print(f"- {value}")

    print("\nBrand Audience:")
    for audience in result.audiences:
        print(
            f"- Audience Name: {audience.name}, "
            f"Gender Distribution: {audience.gender_distribution}, "
            f"Geography: {audience.geography}, "
            f"Income Level: {audience.income_level}, "
            f"Psycographics: {audience.psycographics}, "
            f"Behaviors: {audience.behaviors}, "
            f"Pain Points: {audience.pain_points}, "
            f"Motivations: {audience.motivations}, "
            f"Purchase Drivers: {audience.purchase_drivers}, "
            f"Preferred Channels: {audience.preferred_channels}")

    print(f"Campaign Goal: {result.goal}")

    print("Contraints:")
    if result.constraints:
        print(f"- Budget: {result.constraints.budget}, "
        f"- Timeline: {result.constraints.timeline}")

async def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(base_dir, "examples", "skims.txt")

    with open(file_path, "r", encoding="utf-8") as f:
        dossier_text = f.read()

    campaign_goal = "Awareness for Fall 2025 international push"
    brand_context = f"""{dossier_text}

CAMPAIGN GOAL: {campaign_goal}
"""
    # 3) Run once, then print
    structured = await process_context(brand_context)
    print("Structured output:")
    await structure_output(structured)

if __name__ == "__main__":
    asyncio.run(main())
