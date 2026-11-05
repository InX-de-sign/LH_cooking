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
            self.store.start_visit("sam")
            for answer in ("Sam", "beginner", "peanuts", "pasta"):
                self.store.submit_onboarding_answer("sam", answer)
            visit = self.store.current_visit("sam")
            self.store.set_recipe(visit, 7, "Pasta")
            self.store.set_step(visit, 2)
            self.store.save_profile("sam", {"cook_name": "Alex", "skill_level": "confident", "dietary": "Peanut allergy; vegetarian"})
            updated = self.store.current_visit("sam")
            self.assertEqual((updated.cook_name, updated.dietary), ("Alex", "Peanut allergy; vegetarian"))
            self.assertEqual((updated.recipe_id, updated.step), (7, 2))
            self.assertTrue(updated.onboarding_complete)
        self.store.close()
        self.store = ConversationStore(self.path)
        with patch("conversation_state.time.time", return_value=2_000_000_000 + SESSION_IDLE_SECONDS + 1):
            new_visit = self.store.current_visit("sam")
            self.assertNotEqual(new_visit.session_id, visit.session_id)
            self.assertEqual(new_visit.cook_name, "Alex")
            self.assertEqual(new_visit.dietary, "Peanut allergy; vegetarian")
            self.assertEqual(new_visit.onboarding_step, "dish")
            self.assertIsNone(new_visit.recipe_id)
            self.assertIsNone(new_visit.dish)

    def test_profile_form_skips_repeated_setup_questions_and_can_clear_allergies(self):
        self.store.save_profile("sam", {"cook_name": "Sam", "skill_level": "beginner", "dietary": "Peanuts"})
        self.assertEqual(self.store.onboarding_state("sam")["step"], "dish")
        result = self.store.submit_onboarding_answer("sam", "pasta")
        self.assertTrue(result["just_completed"])
        self.store.save_profile("sam", {"cook_name": "", "skill_level": "", "dietary": ""})
        self.assertEqual(self.store.get_profile("sam")["dietary"], "")
        self.assertIsNone(self.store.current_visit("sam").dietary)

    def test_legacy_profile_is_kept_on_database_upgrade(self):
        self.store.start_visit("legacy")
        for answer in ("Sam", "beginner", "Peanuts", "pasta"):
            self.store.submit_onboarding_answer("legacy", answer)
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
