"""Timer names stay attached to the process, independent of recipe quantities."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chatbot"))
from context_builder import ContextBuilder, cooking_state
from conversation_state import ConversationStore
from knowledge_base import KnowledgeBase
from recipe_import import RecipeImportError, validate_recipe, validate_working_recipe
from timer_labels import clean_timer_label, timer_label_for_step


class TimerLabelTests(unittest.TestCase):
    def test_every_bundled_timer_has_an_explicit_process_name(self):
        recipes = yaml.safe_load((ROOT / "chatbot/recipes.yaml").read_text(encoding="utf-8"))
        count = 0
        for recipe in recipes:
            for step in recipe["steps"]:
                if step.get("timer_seconds"):
                    count += 1
                    with self.subTest(recipe=recipe["name"], step=step["text"]):
                        label = step.get("timer_label")
                        self.assertIsInstance(label, str)
                        self.assertTrue(label.strip())
                        self.assertLessEqual(len(label), 40)
                        self.assertNotRegex(label, r"^Step \d+$")
        self.assertGreater(count, 0)

    def test_legacy_names_describe_timed_action(self):
        cases = [
            ("Boil the spaghetti for 10 minutes.", "Spaghetti", "Boiling spaghetti"),
            ("Stir-fry noodles for 3 minutes.", "Noodles", "Stir-frying noodles"),
            ("Move chicken to a clean plate and rest for 5 minutes.", "Chicken", "Resting Chicken"),
            ("Mix batter, then bake banana bread for 50 minutes.", "Bread", "Baking banana bread"),
        ]
        for text, recipe, expected in cases:
            self.assertEqual(timer_label_for_step({"text": text}, recipe), expected)
        self.assertEqual(clean_timer_label("  My   pasta timer  "), "My pasta timer")
        self.assertLessEqual(len(clean_timer_label("A very long descriptive timer name " * 3)), 40)

    def test_new_timed_steps_require_labels(self):
        recipe = {"name": "Noodles", "serves": 2, "ingredients": ["200 g noodles"],
                  "steps": [{"text": "Boil noodles.", "timer_seconds": 180}]}
        for validate in (validate_recipe, validate_working_recipe):
            for label in (None, "", "   ", "x" * 41):
                with self.subTest(validator=validate.__name__, label=label):
                    recipe["steps"][0]["timer_label"] = label
                    with self.assertRaises(RecipeImportError):
                        validate(recipe)
            recipe["steps"][0]["timer_label"] = "Boiling noodles"
            self.assertEqual(validate(recipe)["steps"][0]["timer_label"], "Boiling noodles")

    def test_labels_survive_catalog_import_save_revision_and_scaling(self):
        with tempfile.TemporaryDirectory() as folder:
            kb = KnowledgeBase(str(Path(folder) / "knowledge.db"))
            store = ConversationStore(str(Path(folder) / "memory.db"))
            try:
                context = ContextBuilder(kb, store)
                visit = store.current_visit("timer-test")
                catalog = kb.resolve_subject("Garlic Soy Noodles")
                store.set_recipe(visit, catalog.id, catalog.name)
                timed = [s for s in context.active_recipe(visit).steps if s.get("timer_seconds")]
                self.assertEqual([s["timer_label"] for s in timed], ["Frying garlic", "Tossing noodles in sauce"])
                imported = validate_recipe({"name": "Spaghetti", "serves": 2, "ingredients": ["200 g spaghetti"],
                    "steps": [{"text": "Boil spaghetti for 10 minutes.", "timer_seconds": 600, "timer_label": "Boiling spaghetti"}]})
                custom_id = store.save_custom_recipe(visit, imported)
                store.set_recipe(visit, None, imported["name"], custom_id=custom_id)
                store.set_step(visit, 0)
                store.set_servings(visit, 4)
                scaled = context.active_recipe(visit)
                self.assertIn("400", scaled.ingredients[0])
                self.assertEqual(cooking_state(scaled, 0)["timer_label"], "Boiling spaghetti")
                revised = copy.deepcopy(imported)
                revised["steps"][0].update(text="Stir-fry spaghetti for 2 minutes.", timer_seconds=120,
                                            timer_label="Stir-frying spaghetti")
                store.set_recipe_draft(visit, validate_working_recipe(revised), "Change method", visit.recipe_revision)
                self.assertEqual(cooking_state(context.active_recipe(visit), 0)["timer_label"], "Stir-frying spaghetti")
                self.assertEqual(store.custom_recipe(custom_id, "timer-test")["steps"][0]["timer_label"], "Boiling spaghetti")
                # Older saved JSON has no timer_label; reading it supplies one.
                del imported["steps"][0]["timer_label"]
                legacy_id = store.save_custom_recipe(visit, imported)
                store.set_recipe(visit, None, imported["name"], custom_id=legacy_id)
                self.assertEqual(cooking_state(context.active_recipe(visit), 0)["timer_label"], "Boiling spaghetti")
            finally:
                kb.close()
                store.close()
