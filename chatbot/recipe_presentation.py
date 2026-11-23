"""Presentation-only recipe metadata, independent of cooking instructions."""
import json
from pathlib import Path

CATALOG_MEDIA = json.loads(Path(__file__).with_name("recipe_presentation.json").read_text(encoding="utf-8"))
CATEGORIES = ("Main course", "Breakfast", "Soup", "Salad", "Dessert", "Snacks & sides", "Personal")
_PUBLIC = ("slug", "category", "image", "card_image", "thumbnail", "image_alt", "illustrative")


def presentation_for(name):
    metadata = CATALOG_MEDIA.get(name)
    return {key: metadata[key] for key in _PUBLIC} if metadata else {"category": "Personal", "illustrative": False}


def catalog_name(slug):
    if not isinstance(slug, str):
        raise ValueError("Unknown recipe.")
    for name, metadata in CATALOG_MEDIA.items():
        if metadata["slug"] == slug:
            return name
    raise ValueError("Unknown recipe.")


def present_recipe(recipe, source="personal"):
    return {**recipe, **presentation_for(recipe["name"]), "source": source, "personal": source == "personal"}
