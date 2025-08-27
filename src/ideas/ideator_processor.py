import asyncio, os, json
from dotenv import load_dotenv
from agents import Runner
from src.ideas.ideator_extractor import ideator_agent

load_dotenv()

INPUT_FILE  = "outputs/brand_brief.json"
OUTPUT_FILE = "outputs/ideas.json"
DOSSIER_FILE = "examples/skims.txt"  # optional: pass to ideator for extra context

async def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    in_path  = os.path.join(base_dir, INPUT_FILE)
    out_path = os.path.join(base_dir, OUTPUT_FILE)
    dossier_path = os.path.join(base_dir, DOSSIER_FILE)

    # load structured brief
    with open(in_path, "r", encoding="utf-8") as f:
        brief_json = f.read()

    # include dossier text to give ideator richer hooks
    dossier_text = ""
    if os.path.exists(dossier_path):
        with open(dossier_path, "r", encoding="utf-8") as f:
            dossier_text = f.read()

    # build prompt
    ideation_prompt = (
        "BRAND BRIEF (JSON):\n" + brief_json +
        ("\n\nSELECTED DOSSIER EXCERPT:\n" + dossier_text[:4000] if dossier_text else "") +
        "\n\nGenerate ideas per spec."
    )

    runner = Runner()
    result = await runner.run(ideator_agent, ideation_prompt)
    ideas = result.final_output

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ideas.model_dump(), f, indent=2, ensure_ascii=False)

    print(f"Saved ideas → {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())