"""Catalog media completeness and user-scoped favourites without model calls."""
import sys
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chatbot"))
from conversation_state import ConversationStore
from recipe_presentation import CATALOG_MEDIA, CATEGORIES, catalog_name, present_recipe


class RecipePresentationTests(unittest.TestCase):
    def test_every_catalog_recipe_has_unique_metadata_and_sized_local_images(self):
        recipes = yaml.safe_load((ROOT / "chatbot/recipes.yaml").read_text(encoding="utf-8"))
        self.assertEqual(set(CATALOG_MEDIA), {r["name"] for r in recipes})
        self.assertEqual(len({m["slug"] for m in CATALOG_MEDIA.values()}), len(recipes))
        for name, media in CATALOG_MEDIA.items():
            with self.subTest(recipe=name):
                self.assertIn(media["category"], CATEGORIES)
                self.assertLess(len(media["image_alt"]), 125)
                self.assertIn("No people", media["prompt"])
                self.assertEqual(catalog_name(media["slug"]), name)
                for key, dimension in (("image", "detail"), ("card_image", "card"), ("thumbnail", "thumbnail")):
                    filename = ROOT / "chatbot" / media[key].lstrip("/")
                    with Image.open(filename) as image:
                        self.assertEqual(image.size, tuple(media["dimensions"][dimension]))
                        self.assertEqual(image.format, "WEBP")

    def test_personal_recipe_presentation_has_an_honest_placeholder(self):
        row = present_recipe({"id": 7, "name": "My special noodles", "favorite": True})
        self.assertTrue(row["personal"])
        self.assertEqual(row["category"], "Personal")
        self.assertNotIn("image", row)
        catalog = present_recipe({"id": 1, "name": "Tomato Soup"}, "catalog")
        self.assertFalse(catalog["personal"])
        self.assertEqual(catalog["category"], "Soup")
        self.assertNotIn("prompt", catalog, "generation provenance stays out of public recipe payloads")

    def test_catalog_favorites_survive_reopen_and_are_isolated_from_personal_favorites(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "memory.db")
            store = ConversationStore(path)
            try:
                visit = store.current_visit("sam")
                custom_id = store.save_custom_recipe(visit, {"name": "My soup", "steps": [{"text": "Heat stock."}]})
                store.set_custom_favorite("sam", custom_id, True)
                self.assertTrue(store.set_catalog_favorite("sam", "tomato-soup", True))
                self.assertEqual(store.catalog_favorites("other"), [])
                with self.assertRaises(ValueError):
                    store.set_custom_favorite("other", custom_id, False)
                with self.assertRaises(ValueError):
                    store.set_catalog_favorite("sam", "invalid-slug", True)
                store.close(); store = ConversationStore(path)
                self.assertEqual(store.catalog_favorites("sam"), ["tomato-soup"])
                self.assertTrue(store.list_custom_recipes("sam")[0]["favorite"], "legacy personal favourites are unchanged")
                store.set_catalog_favorite("sam", "tomato-soup", False)
                self.assertEqual(store.catalog_favorites("sam"), [])
                self.assertTrue(store.list_custom_recipes("sam")[0]["favorite"])
                self.assertIsNone(store.current_visit("sam").step, "favouriting never starts cooking")
            finally:
                store.close()
