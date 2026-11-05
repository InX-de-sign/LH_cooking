"""Check settings without exposing keys; add --live to make billable API calls.

    python scripts/check_setup.py
    python scripts/check_setup.py --live --image path/to/food.jpg
"""
import argparse
import asyncio
import mimetypes
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chatbot"))

from config import load_azure_openai_config
from api_client import base_url
from vision import VisionRequest, load_vision_provider


async def check(args):
    config = load_azure_openai_config()
    vision = load_vision_provider()
    print(f"Settings file: {ROOT / 'chatbot' / '.env'}")
    print(f"Chat configured: {config.configured} (key is never displayed)")
    print(f"Endpoint: {base_url(config)}")
    print(f"Chat deployment: {config.deployment_name}")
    print(f"API version: {config.api_version}")
    print(f"Vision provider: {vision.name}; deployment: {vision.config.model or '(not set)'}")
    print(f"Vision configured: {vision.available}")
    if not args.live:
        print("Configuration check only; network access has not been tested.")
        print("Next: --live to test chat streaming; add --image food.jpg to test vision.")
        return 0 if config.configured else 2
    if not config.configured:
        print("FAIL: Fill OPENAI_API_KEY in chatbot/.env first, then restart the server.")
        return 2

    from streaming_openai import StreamingOpenAI
    chat = StreamingOpenAI()
    try:
        chunks = []
        async for chunk in chat.stream_response("Say hello in one sentence.", {}):
            chunks.append(chunk)
        if not "".join(chunks).strip():
            print("FAIL: Chat returned no text.")
            return 1
        print(f"PASS: Chat streaming returned text ({len(chunks)} chunks).")
        if args.image:
            if not vision.available:
                print("FAIL: Set VLM_PROVIDER=azure and VLM_MODEL to an image-capable deployment.")
                return 1
            path = Path(args.image)
            mime = mimetypes.guess_type(path.name)[0]
            if mime not in ("image/jpeg", "image/png", "image/webp"):
                print("FAIL: Use a JPEG, PNG or WebP image.")
                return 1
            image = path.read_bytes()
            if not image or len(image) > vision.config.max_image_bytes:
                print("FAIL: Image is empty or too large; use an image smaller than 1.5 MB.")
                return 1
            observation = await vision.observe(image, mime, VisionRequest("What food is visible?"))
            if observation is None:
                print("FAIL: Vision returned no usable observation; check deployment and server log.")
                return 1
            print("PASS: Vision returned an observation.")
        else:
            print("SKIP: Live vision test needs --image path/to/food.jpg.")
        return 0
    finally:
        await chat.client.close()
        if getattr(vision, "client", None):
            await vision.client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="call Azure (uses API quota)")
    parser.add_argument("--image", help="send this food photo to the configured vision deployment")
    args = parser.parse_args()
    if args.image and not args.live:
        parser.error("--image requires --live")
    try:
        return asyncio.run(check(args))
    except Exception as exc:
        # Do not print request headers or secrets, including arbitrary SDK errors.
        print(f"FAIL: {type(exc).__name__}; check endpoint, deployment, key and network access.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
