"""Live built-in web search check (uses API quota); never prints the key."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chatbot"))
from web_search import BuiltinWebSearch, SearchUnavailable


async def run(args):
    search = BuiltinWebSearch()
    bases = [search.base_url]
    if args.probe and "hkust.azure-api.net" in search.base_url:
        bases = ["https://hkust.azure-api.net/hkust-genai/v1", "https://hkust.azure-api.net/openai/v1"]
    for base in dict.fromkeys(bases):
        if not args.json:
            print("Testing " + base + "/responses", flush=True)
        try:
            result = await search.search(args.query, force=True, base_url=base)
        except SearchUnavailable as exc:
            if args.json:
                print(json.dumps({"error": str(exc)}))
            else:
                print("UNAVAILABLE: " + str(exc), flush=True)
            continue
        if args.json:
            print(json.dumps({"text": result.text, "sources": result.sources}, ensure_ascii=True))
        else:
            print("PASS: completed web_search call with source links.")
            print(result.text)
            for source in result.sources:
                print(source["title"] + ": " + source["url"])
        return 0
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="Find a simple tomato and egg stir-fry recipe online and cite its source.")
    parser.add_argument("--probe", action="store_true", help="check documented HKUST and Azure v1 routes")
    parser.add_argument("--json", action="store_true", help="machine-readable result for the desktop launcher")
    args = parser.parse_args()
    if args.probe and args.json:
        parser.error("--probe and --json cannot be combined")
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
