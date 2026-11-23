"""Convert generated recipe originals to the app's three responsive WebP slots.

Usage: python scripts/build_recipe_images.py SLUG SOURCE_PNG
Generation prompts and target dimensions live in chatbot/recipe_presentation.json.
"""
import json
import sys
from pathlib import Path
from PIL import Image, ImageOps

root = Path(__file__).resolve().parents[1]
slug, source = sys.argv[1:]
metadata = next(m for m in json.loads((root / "chatbot/recipe_presentation.json").read_text(encoding="utf-8")).values()
                if m["slug"] == slug)
output = root / "chatbot/static/recipes"
output.mkdir(exist_ok=True)
with Image.open(source) as original:
    original = ImageOps.exif_transpose(original).convert("RGB")
    for variant, size in metadata["dimensions"].items():
        suffix = "thumb" if variant == "thumbnail" else variant
        image = ImageOps.fit(original, tuple(size), Image.Resampling.LANCZOS, centering=(0.5, 0.56))
        image.save(output / f"stovetop-{slug}-{suffix}.webp", "WEBP", quality=83, method=6)
print(slug)
