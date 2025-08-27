# src/score/score_processor.py
import asyncio, os, json
from dotenv import load_dotenv
from agents import Runner
from src.score.score_extractor import score_agent  # <-- use your SCORE agent

load_dotenv()

BRIEF_FILE   = "outputs/brand_brief.json"      # from extractor step
IDEAS_FILE   = "outputs/ideas.json"            # from ideator step
OUTPUT_FILE  = "outputs/scored_ideas.json"     # we write this

def build_idea_payload(brief_obj: dict, idea_obj: dict) -> str:
    """Compose a simple per-idea scoring prompt (two JSON blocks)."""
    return (
        "BRAND_BRIEF (JSON):\n" + json.dumps(brief_obj, indent=2) +
        "\n\nIDEA (JSON):\n" + json.dumps({
            "category": idea_obj.get("category"),
            "title": idea_obj.get("title"),
            "concept": idea_obj.get("concept"),
            "execution_notes": idea_obj.get("execution_notes"),
            "sources": idea_obj.get("sources", []),
        }, indent=2) +
        "\n\nReturn IdeaScore JSON only."
    )

async def main():
    base = os.path.dirname(os.path.dirname(__file__))
    root_dir = os.path.abspath(os.path.join(base, ".."))

    brief_path = os.path.join(root_dir, "outputs", "brand_brief.json")
    ideas_path = os.path.join(root_dir, "outputs", "ideas.json")
    out_path   = os.path.join(root_dir, "outputs", "scored_ideas.json")

    # 1) load brief + ideas
    with open(brief_path, "r", encoding="utf-8") as f:
        brief_obj = json.load(f)

    with open(ideas_path, "r", encoding="utf-8") as f:
        ideas_bundle = json.load(f)

    ideas_list = ideas_bundle.get("ideas", [])
    if not isinstance(ideas_list, list) or not ideas_list:
        raise ValueError("No ideas found in outputs/ideas.json under key 'ideas'.")

    runner = Runner()

    # 2) score ideas (simple sequential loop to keep things minimal)
    scored_ideas = []
    for idea in ideas_list:
        prompt = build_idea_payload(brief_obj, idea)
        res = await runner.run(score_agent, prompt)
        score_dict = res.final_output.model_dump()  # IdeaScore -> dict
        scored_ideas.append({**idea, "scores": score_dict})

    # 3) write output
    out = {
        "campaign_intent": ideas_bundle.get("campaign_intent"),
        "brand_name": ideas_bundle.get("brand_name"),
        "total_ideas": len(scored_ideas),
        "ideas": scored_ideas,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Saved scored ideas → {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
