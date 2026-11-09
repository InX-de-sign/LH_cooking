"""Profile persistence, history pagination, and browser identity isolation."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chatbot"))
from conversation_state import ConversationStore, SESSION_IDLE_SECONDS


class ProfileHistoryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = str(Path(self.folder.name) / "memory.db")
        self.store = ConversationStore(self.path)

    def tearDown(self):
        self.store.close()
        self.folder.cleanup()

    def test_profile_changes_reach_active_visit_and_new_sessions_without_changing_recipe(self):
        with patch("conversation_state.time.time", return_value=2_000_000_000):
            visit = self.store.current_visit("sam")
            self.assertIsNone(visit.cook_name, "a new cook starts with no questionnaire and no profile")
            self.store.update_profile("sam", {"cook_name": "Sam", "dietary": "peanuts"})
            self.store.set_recipe(visit, 7, "Pasta")
            self.store.set_step(visit, 2)
            self.store.save_profile("sam", {"cook_name": "Alex", "skill_level": "confident", "dietary": "Peanut allergy; vegetarian"})
            updated = self.store.current_visit("sam")
            self.assertEqual((updated.cook_name, updated.dietary), ("Alex", "Peanut allergy; vegetarian"))
            self.assertEqual((updated.recipe_id, updated.step), (7, 2))
        self.store.close()
        self.store = ConversationStore(self.path)
        with patch("conversation_state.time.time", return_value=2_000_000_000 + SESSION_IDLE_SECONDS + 1):
            new_visit = self.store.current_visit("sam")
            self.assertNotEqual(new_visit.session_id, visit.session_id)
            self.assertEqual(new_visit.cook_name, "Alex")
            self.assertEqual(new_visit.dietary, "Peanut allergy; vegetarian")
            self.assertIsNone(new_visit.recipe_id)
            self.assertIsNone(new_visit.dish)

    def test_model_profile_updates_only_change_given_fields_and_form_can_clear(self):
        self.store.save_profile("sam", {"cook_name": "Sam", "skill_level": "beginner", "dietary": "Peanuts"})
        self.store.update_profile("sam", {"skill_level": "some experience"})
        self.assertEqual(self.store.get_profile("sam"),
                         {"cook_name": "Sam", "skill_level": "some experience", "dietary": "Peanuts", "notes": ""})
        with self.assertRaises(ValueError):
            self.store.update_profile("sam", {"dietary": "x" * 1001})
        self.store.save_profile("sam", {"cook_name": "", "skill_level": "", "dietary": ""})
        self.assertEqual(self.store.get_profile("sam")["dietary"], "")
        self.assertIsNone(self.store.current_visit("sam").dietary)

    def test_uploaded_recipe_is_private_to_its_browser_and_switchable(self):
        visit = self.store.current_visit("sam")
        recipe = {"name": "Grandma's noodles", "serves": 2, "ingredients": ["200 g noodles"],
                  "steps": [{"text": "Boil noodles.", "done_when": None, "timer_seconds": 180}]}
        custom_id = self.store.save_custom_recipe(visit, recipe)
        self.store.set_recipe(visit, None, recipe["name"], custom_id=custom_id)
        resumed = self.store.current_visit("sam")
        self.assertEqual((resumed.recipe_custom_id, resumed.recipe_id, resumed.step), (custom_id, None, None))
        self.assertTrue(resumed.has_recipe)
        self.assertEqual(self.store.custom_recipe(custom_id, "sam")["steps"][0]["timer_seconds"], 180)
        self.assertIsNone(self.store.custom_recipe(custom_id, "someone-else"))
        self.store.set_recipe(resumed, 7, "Pasta")
        switched = self.store.current_visit("sam")
        self.assertEqual((switched.recipe_id, switched.recipe_custom_id), (7, None))
        with self.assertRaises(ValueError):
            self.store.set_recipe(switched, 7, "Pasta", custom_id=custom_id)

    def test_legacy_profile_is_kept_on_database_upgrade(self):
        visit = self.store.current_visit("legacy")
        self.store._update_visit(visit.session_id, cook_name="Sam", skill_level="beginner",
                                 dietary="Peanuts", onboarding_step="complete")
        with self.store._lock:
            self.store._conn.execute("DROP TABLE profiles")
            self.store._conn.commit()
        self.store.close()
        self.store = ConversationStore(self.path)
        with self.store._lock:
            profile = self.store._conn.execute("SELECT * FROM profiles WHERE user_id = 'legacy'").fetchone()
        self.assertEqual(profile["cook_name"], "Sam")
        self.assertEqual(profile["dietary"], "Peanuts")

    def test_history_is_paginated_chronological_and_scoped_to_user(self):
        with patch("conversation_state.time.time", return_value=2_000_000_000):
            visit = self.store.current_visit("sam")
            other = self.store.current_visit("other")
            self.store.record_turn(other, None, "private", "other reply")
            for i in range(45):
                self.store.record_turn(visit, None, f"Question {i}", f"Answer {i}", vision="a tomato" if i == 44 else None)
        first = self.store.history_detail("sam", visit.session_id)
        self.assertEqual(len(first["turns"]), 40)
        self.assertEqual(first["turns"][0]["user_text"], "Question 5")
        self.assertEqual(first["turns"][-1]["vision"], "a tomato")
        older = self.store.history_detail("sam", visit.session_id, first["next_offset"])
        self.assertEqual(older["turns"][0]["user_text"], "Question 0")
        self.assertIsNone(older["next_offset"])
        with self.assertRaisesRegex(ValueError, "not found"):
            self.store.history_detail("other", visit.session_id)
        with patch("conversation_state.time.time", return_value=2_000_000_000 + SESSION_IDLE_SECONDS + 1):
            self.store.current_visit("sam")
        sessions = self.store.history("sam", limit=1)
        self.assertEqual(sessions["next_offset"], 1)
        older_sessions = self.store.history("sam", offset=1, limit=1)
        self.assertEqual(older_sessions["sessions"][0]["session_id"], visit.session_id)
        self.assertEqual(len(self.store.history("other")["sessions"]), 1)

    def test_invalid_profile_and_pagination_do_not_modify_saved_values(self):
        self.store.save_profile("sam", {"cook_name": "Sam", "dietary": "Peanuts"})
        for profile in (None, {"cook_name": []}, {"dietary": "x" * 1001}):
            with self.assertRaises(ValueError):
                self.store.save_profile("sam", profile)
        self.assertEqual(self.store.get_profile("sam")["dietary"], "Peanuts")
        for offset in (-1, "1", True):
            with self.assertRaises(ValueError):
                self.store.history("sam", offset=offset)


if __name__ == "__main__":
    unittest.main()
