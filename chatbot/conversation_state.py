# conversation_state.py - Hierarchical conversation memory for the cooking coach
"""
Three levels of conversation memory, matching how a cooking session unfolds.

    VISIT      one cook, one session in the kitchen, ended by going quiet
      +- EPISODE   one subject they are asking about (a recipe, a technique)
           +- TURN     one question and one answer

The table and class are still called "visit" -- a cooking session is a visit
to the kitchen -- which keeps the schema stable for anything reading memory.db.

THE RECIPE LIVES ON THE VISIT, NOT THE EPISODE
----------------------------------------------
A cook halfway through the pasta asks "what does emulsify mean?". That opens
an episode about the technique, and the next "what's next?" must still land on
step 6 of the pasta. So the active recipe and the step pointer are visit-level
state; episodes only track what the conversation is about right now.

THE DELIVERED-FACT LEDGER
-------------------------
Every fact sent to the model is recorded against the visit. Retrieval then
excludes them, so asking "any other tips?" twice gives two different answers
rather than the same sentence again. Recipe steps are not facts and are never
excluded -- repeating a step is a normal request.

SUMMARIES ARE COMPUTED, NOT GENERATED
-------------------------------------
Episode summaries are assembled from the facets already covered rather than by
asking the model to summarise. A voice coach cannot spend a second and a half
of API latency on bookkeeping.

The visit profile (name, skill, dietary needs, today's dish) is stored only
with the visit. It is not a cross-session account.
"""

import json
import logging
import os
import re
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# How long a cook can be silent before their next question starts a fresh
# session. Generous, because a braise or a proving dough leaves long gaps, and
# the phone belongs to one person rather than being handed around.
SESSION_IDLE_SECONDS = int(os.getenv("SESSION_IDLE_SECONDS", str(3 * 60 * 60)))

# Episodes end on their own when the cook moves to a new subject, but a long
# gap on the same subject should still start a new one.
EPISODE_IDLE_SECONDS = int(os.getenv("EPISODE_IDLE_SECONDS", str(20 * 60)))

ONBOARDING_STEPS = ("name", "skill", "dietary", "dish", "complete")
ONBOARDING_QUESTIONS = {
    "name": "Hi, I'm your cooking coach! What's your name?",
    "skill": "How comfortable are you in the kitchen: beginner, some experience, or confident?",
    "dietary": "Any allergies or foods you avoid? Just say none if not.",
    "dish": "What are you cooking today?",
}
SKIP_ANSWERS = frozenset({"skip", "no", "none", "nope", "nothing", "no thanks",
                          "prefer not to say", "n/a"})

# Normalised skill levels. An answer that matches none of these is kept as the
# cook said it -- the model can read "I can boil an egg" perfectly well.
SKILL_LEVELS = (
    ("beginner", ("beginner", "new", "never", "first time", "not much", "not very",
                  "learning", "novice", "bad", "terrible", "basic")),
    ("confident", ("confident", "experienced", "very", "pro", "advanced", "chef",
                   "a lot", "expert", "good cook")),
    ("some experience", ("some", "okay", "ok", "bit", "intermediate", "sometimes",
                         "average", "decent", "fine", "alright")),
)

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS visits (
    session_id  TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    started_at  REAL NOT NULL,
    last_active REAL NOT NULL,
    language    TEXT,
    turn_count  INTEGER DEFAULT 0,
    cook_name   TEXT,
    skill_level TEXT,
    dietary     TEXT,
    dish        TEXT,             -- today's dish, as the cook said it
    recipe_id   INTEGER,          -- the recipe being cooked, if we know it
    recipe_name TEXT,
    step        INTEGER,          -- 0-based; NULL = not started
    onboarding_step TEXT NOT NULL DEFAULT 'name',
    onboarding_started_at REAL,
    onboarding_completed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_visits_user ON visits(user_id, last_active DESC);

CREATE TABLE IF NOT EXISTS episodes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES visits(session_id),
    user_id      TEXT NOT NULL,
    subject_type TEXT,
    subject_id   INTEGER,
    subject_name TEXT,
    opened_by    TEXT,            -- named | carried | camera | recipe | general
    opened_at    REAL NOT NULL,
    closed_at    REAL,
    last_active  REAL NOT NULL,
    turn_count   INTEGER DEFAULT 0,
    facets       TEXT DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id, opened_at DESC);

CREATE TABLE IF NOT EXISTS turns (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    episode_id    INTEGER,
    user_id       TEXT NOT NULL,
    ts            REAL NOT NULL,
    user_text     TEXT,
    assistant_text TEXT,
    language      TEXT,
    intent        TEXT,
    facet         TEXT,
    fact_ids      TEXT DEFAULT '[]',
    source        TEXT,
    vision        TEXT            -- what the camera saw this turn, if anything
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_turns_episode ON turns(episode_id, ts DESC);

-- The ledger. Keyed by visit, so a new session starts with every fact
-- available again.
CREATE TABLE IF NOT EXISTS delivered_facts (
    session_id   TEXT NOT NULL,
    fact_id      INTEGER NOT NULL,
    delivered_at REAL NOT NULL,
    PRIMARY KEY (session_id, fact_id)
);
"""


class Visit:
    """One cook's session in the kitchen."""

    __slots__ = (
        "session_id", "user_id", "started_at", "last_active", "language", "turn_count",
        "cook_name", "skill_level", "dietary", "dish", "recipe_id", "recipe_name", "step",
        "onboarding_step", "onboarding_started_at", "onboarding_completed_at",
    )

    def __init__(self, session_id, user_id, started_at, last_active, language, turn_count,
                 cook_name=None, skill_level=None, dietary=None, dish=None,
                 recipe_id=None, recipe_name=None, step=None,
                 onboarding_step="name", onboarding_started_at=None,
                 onboarding_completed_at=None):
        self.session_id = session_id
        self.user_id = user_id
        self.started_at = started_at
        self.last_active = last_active
        self.language = language
        self.turn_count = turn_count
        self.cook_name = cook_name
        self.skill_level = skill_level
        self.dietary = dietary
        self.dish = dish
        self.recipe_id = recipe_id
        self.recipe_name = recipe_name
        self.step = step
        self.onboarding_step = onboarding_step
        self.onboarding_started_at = onboarding_started_at
        self.onboarding_completed_at = onboarding_completed_at

    @property
    def minutes_so_far(self) -> int:
        return int((time.time() - self.started_at) / 60)

    @property
    def onboarding_complete(self) -> bool:
        return self.onboarding_step == "complete"

    def __repr__(self):
        return (f"Visit({self.session_id} user={self.user_id} turns={self.turn_count} "
                f"recipe={self.recipe_name!r} step={self.step} "
                f"onboarding={self.onboarding_step})")


class Episode:
    """A stretch of conversation about one subject."""

    __slots__ = ("id", "session_id", "subject_type", "subject_id", "subject_name",
                 "opened_by", "opened_at", "last_active", "turn_count", "facets")

    def __init__(self, id, session_id, subject_type, subject_id, subject_name,
                 opened_by, opened_at, last_active, turn_count, facets):
        self.id = id
        self.session_id = session_id
        self.subject_type = subject_type
        self.subject_id = subject_id
        self.subject_name = subject_name
        self.opened_by = opened_by
        self.opened_at = opened_at
        self.last_active = last_active
        self.turn_count = turn_count
        self.facets = facets or []

    @property
    def subject_key(self):
        return (self.subject_type, self.subject_id)

    def summary(self) -> Optional[str]:
        """A one-line account of this episode, built from what was covered.

        Deliberately not model-generated: this runs on the request path.
        """
        if not self.subject_name:
            return None
        if not self.facets:
            return f"Just started talking about {self.subject_name}."
        readable = ", ".join(self.facets)
        return (f"Already covered about {self.subject_name}: {readable}"
                f" ({self.turn_count} question{'s' if self.turn_count != 1 else ''}).")

    def __repr__(self):
        return f"Episode({self.id} {self.subject_name!r} turns={self.turn_count})"


class ConversationStore:
    """Persistence for the visit/episode/turn hierarchy and the fact ledger."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.getenv("MEMORY_DB_PATH", "memory.db")
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._migrate()
            self._conn.commit()
        logger.info("ConversationStore ready at %s", self.db_path)

    # -- visits -----------------------------------------------------------
    def _migrate(self):
        """Add columns introduced after a memory.db was first created."""
        for table, migrations in (
            ("visits", {
                "cook_name": "TEXT", "skill_level": "TEXT", "dietary": "TEXT",
                "dish": "TEXT", "recipe_id": "INTEGER", "recipe_name": "TEXT",
                "step": "INTEGER",
                "onboarding_step": "TEXT NOT NULL DEFAULT 'name'",
                "onboarding_started_at": "REAL", "onboarding_completed_at": "REAL",
            }),
            ("turns", {"vision": "TEXT"}),
        ):
            columns = {row["name"] for row in
                       self._conn.execute(f"PRAGMA table_info({table})").fetchall()}
            for name, definition in migrations.items():
                if name not in columns:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    @staticmethod
    def _visit_from_row(row: sqlite3.Row) -> Visit:
        return Visit(
            row["session_id"], row["user_id"], row["started_at"], row["last_active"],
            row["language"], row["turn_count"], row["cook_name"], row["skill_level"],
            row["dietary"], row["dish"], row["recipe_id"], row["recipe_name"], row["step"],
            row["onboarding_step"], row["onboarding_started_at"],
            row["onboarding_completed_at"],
        )

    def current_visit(self, user_id: str, language: str = None) -> Visit:
        """Resume this cook's open session, or start a new one."""
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM visits WHERE user_id = ? ORDER BY last_active DESC LIMIT 1",
                (user_id,),
            ).fetchone()

            if row and (now - row["last_active"]) <= SESSION_IDLE_SECONDS:
                return self._visit_from_row(row)

            session_id = f"{user_id}:{int(now)}"
            self._conn.execute(
                "INSERT INTO visits (session_id, user_id, started_at, last_active, language,"
                " turn_count, onboarding_step)"
                " VALUES (?, ?, ?, ?, ?, 0, 'name')",
                (session_id, user_id, now, now, language or "en"),
            )
            self._conn.commit()
            logger.info("New visit %s for %s", session_id, user_id)
            return Visit(session_id, user_id, now, now, language or "en", 0)

    def touch_visit(self, session_id: str):
        with self._lock:
            self._conn.execute("UPDATE visits SET last_active = ? WHERE session_id = ?",
                               (time.time(), session_id))
            self._conn.commit()

    def _update_visit(self, session_id: str, **columns):
        assignments = ", ".join(f"{column} = ?" for column in columns)
        with self._lock:
            self._conn.execute(
                f"UPDATE visits SET {assignments} WHERE session_id = ?",
                list(columns.values()) + [session_id],
            )
            self._conn.commit()

    # -- the recipe being cooked -------------------------------------------
    def set_recipe(self, visit: Visit, recipe_id: Optional[int], recipe_name: Optional[str]):
        """Start (or clear, with None) the recipe for this session, at no step."""
        self._update_visit(visit.session_id, recipe_id=recipe_id,
                           recipe_name=recipe_name, step=None)
        visit.recipe_id, visit.recipe_name, visit.step = recipe_id, recipe_name, None
        logger.info("Visit %s now cooking %r", visit.session_id, recipe_name)

    def set_step(self, visit: Visit, step: Optional[int]):
        self._update_visit(visit.session_id, step=step)
        visit.step = step

    # -- visit profile -----------------------------------------------------
    @staticmethod
    def _clean_answer(value: str, maximum_length: int) -> str:
        return " ".join(str(value or "").split())[:maximum_length]

    @classmethod
    def _is_skip_answer(cls, value: str) -> bool:
        return re.sub(r"[^\w\s/]", "", cls._clean_answer(value, 80).lower()) in SKIP_ANSWERS

    @staticmethod
    def _skill_from_answer(value: str) -> str:
        lowered = value.lower()
        for level, markers in SKILL_LEVELS:
            if any(re.search(rf"\b{re.escape(m)}\b", lowered) for m in markers):
                return level
        return value

    def start_visit(self, user_id: str) -> Optional[str]:
        """Create or resume a visit and return its next onboarding question."""
        visit = self.current_visit(user_id)
        if visit.onboarding_complete:
            return None

        now = time.time()
        with self._lock:
            self._conn.execute(
                "UPDATE visits SET onboarding_started_at = COALESCE(onboarding_started_at, ?),"
                " last_active = ? WHERE session_id = ?",
                (now, now, visit.session_id),
            )
            self._conn.commit()
        return ONBOARDING_QUESTIONS[visit.onboarding_step]

    def onboarding_state(self, user_id: str) -> Dict[str, Any]:
        """Return non-sensitive state that clients use to guide the onboarding UI."""
        visit = self.current_visit(user_id)
        return {
            "complete": visit.onboarding_complete,
            "step": visit.onboarding_step,
            "question": None if visit.onboarding_complete
            else ONBOARDING_QUESTIONS[visit.onboarding_step],
        }

    def submit_onboarding_answer(self, user_id: str, answer: str) -> Dict[str, Any]:
        """Store one answer and return the next question without invoking the model.

        Returns ``complete`` (onboarding finished, before or by this answer),
        ``message`` (what to say next, or None when the answer is a normal
        question for the coach) and ``just_completed`` (this answer finished
        it, so the caller can resolve the dish and greet).
        """
        visit = self.current_visit(user_id)
        if visit.onboarding_complete:
            return {"complete": True, "message": None, "just_completed": False}
        if visit.onboarding_started_at is None:
            return {"complete": False, "message": self.start_visit(user_id),
                    "just_completed": False}

        clean = self._clean_answer(answer, 250)
        step = visit.onboarding_step
        if not clean:
            return {"complete": False, "message": ONBOARDING_QUESTIONS[step],
                    "just_completed": False}

        updates: Dict[str, Any] = {}
        if step == "name":
            updates["cook_name"] = None if self._is_skip_answer(clean) else clean
            next_step = "skill"
        elif step == "skill":
            updates["skill_level"] = None if self._is_skip_answer(clean) \
                else self._skill_from_answer(clean)
            next_step = "dietary"
        elif step == "dietary":
            updates["dietary"] = None if self._is_skip_answer(clean) else clean
            next_step = "dish"
        else:  # dish
            updates["dish"] = None if self._is_skip_answer(clean) else clean
            next_step = "complete"

        now = time.time()
        updates["onboarding_step"] = next_step
        updates["last_active"] = now
        if next_step == "complete":
            updates["onboarding_completed_at"] = now
        self._update_visit(visit.session_id, **updates)

        if next_step != "complete":
            return {"complete": False, "message": ONBOARDING_QUESTIONS[next_step],
                    "just_completed": False}
        return {"complete": True, "message": None, "just_completed": True}

    # -- episodes ---------------------------------------------------------
    def open_episode(self, session_id: str, user_id: str, subject, opened_by: str) -> Episode:
        now = time.time()
        subject_type = getattr(subject, "type", None)
        subject_id = getattr(subject, "id", None)
        subject_name = getattr(subject, "name", None)
        with self._lock:
            self._conn.execute(
                "UPDATE episodes SET closed_at = ? WHERE session_id = ? AND closed_at IS NULL",
                (now, session_id))
            cur = self._conn.execute(
                "INSERT INTO episodes (session_id, user_id, subject_type, subject_id,"
                " subject_name, opened_by, opened_at, last_active, turn_count, facets)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, '[]')",
                (session_id, user_id, subject_type, subject_id, subject_name,
                 opened_by, now, now),
            )
            self._conn.commit()
            episode_id = cur.lastrowid
        logger.info("Episode %s opened on %r (%s)", episode_id, subject_name, opened_by)
        return Episode(episode_id, session_id, subject_type, subject_id, subject_name,
                       opened_by, now, now, 0, [])

    def open_episode_for(self, session_id: str, user_id: str, subject,
                         opened_by: str) -> Optional[Episode]:
        """Return the episode this turn belongs to, opening a new one if needed.

        Continues the current episode when the subject is unchanged, or when
        this turn resolved no subject at all.
        """
        active = self.active_episode(session_id)
        now = time.time()

        if subject is None:
            return active

        if active and active.subject_key == (subject.type, subject.id) \
                and (now - active.last_active) <= EPISODE_IDLE_SECONDS:
            return active

        return self.open_episode(session_id, user_id, subject, opened_by)

    def active_episode(self, session_id: str) -> Optional[Episode]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM episodes WHERE session_id = ? AND closed_at IS NULL"
                " ORDER BY opened_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return Episode(row["id"], row["session_id"], row["subject_type"], row["subject_id"],
                       row["subject_name"], row["opened_by"], row["opened_at"],
                       row["last_active"], row["turn_count"], json.loads(row["facets"] or "[]"))

    def close_episode(self, session_id: str):
        with self._lock:
            self._conn.execute(
                "UPDATE episodes SET closed_at = ? WHERE session_id = ? AND closed_at IS NULL",
                (time.time(), session_id))
            self._conn.commit()

    # -- turns ------------------------------------------------------------
    def record_turn(self, visit: Visit, episode: Optional[Episode], user_text: str,
                    assistant_text: str, language: str = None, intent: str = None,
                    facet: str = None, fact_ids: Sequence[int] = (), source: str = None,
                    vision: str = None):
        """Persist one exchange and roll the counters that depend on it."""
        now = time.time()
        fact_ids = list(fact_ids or ())
        with self._lock:
            self._conn.execute(
                "INSERT INTO turns (session_id, episode_id, user_id, ts, user_text,"
                " assistant_text, language, intent, facet, fact_ids, source, vision)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (visit.session_id, episode.id if episode else None, visit.user_id, now,
                 user_text, assistant_text, language, intent, facet,
                 json.dumps(fact_ids), source, vision),
            )
            self._conn.execute(
                "UPDATE visits SET last_active = ?, turn_count = turn_count + 1,"
                " language = COALESCE(?, language) WHERE session_id = ?",
                (now, language, visit.session_id))

            if episode is not None:
                facets = list(episode.facets)
                if facet and facet not in facets:
                    facets.append(facet)
                self._conn.execute(
                    "UPDATE episodes SET last_active = ?, turn_count = turn_count + 1,"
                    " facets = ? WHERE id = ?",
                    (now, json.dumps(facets), episode.id))
                episode.facets = facets
                episode.turn_count += 1
                episode.last_active = now

            for fact_id in fact_ids:
                self._conn.execute(
                    "INSERT OR IGNORE INTO delivered_facts (session_id, fact_id, delivered_at)"
                    " VALUES (?, ?, ?)",
                    (visit.session_id, fact_id, now))

            self._conn.commit()

        visit.turn_count += 1
        visit.last_active = now

    def recent_turns(self, session_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """The last few exchanges, oldest first, for verbatim replay in a prompt."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT user_text, assistant_text, facet FROM turns"
                " WHERE session_id = ? ORDER BY ts DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    # -- fact ledger ------------------------------------------------------
    def delivered_fact_ids(self, session_id: str) -> set:
        with self._lock:
            rows = self._conn.execute(
                "SELECT fact_id FROM delivered_facts WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        return {row["fact_id"] for row in rows}

    def forget_delivered(self, session_id: str):
        """Let facts become sayable again.

        Used when retrieval comes back empty because everything on a subject
        has been said. Repeating a good fact beats having nothing to say.
        """
        with self._lock:
            self._conn.execute("DELETE FROM delivered_facts WHERE session_id = ?",
                               (session_id,))
            self._conn.commit()
        logger.info("Cleared delivered-fact ledger for %s", session_id)

    # -- reporting --------------------------------------------------------
    def visit_summary(self, user_id: str) -> Dict[str, Any]:
        """Everything known about a session, for /api/context."""
        with self._lock:
            visit = self._conn.execute(
                "SELECT * FROM visits WHERE user_id = ? ORDER BY last_active DESC LIMIT 1",
                (user_id,),
            ).fetchone()
            if not visit:
                return {"known": False}
            episodes = self._conn.execute(
                "SELECT subject_name, turn_count, facets FROM episodes"
                " WHERE session_id = ? ORDER BY opened_at ASC",
                (visit["session_id"],),
            ).fetchall()
            delivered = self._conn.execute(
                "SELECT COUNT(*) FROM delivered_facts WHERE session_id = ?",
                (visit["session_id"],),
            ).fetchone()[0]

        return {
            "known": True,
            "session_id": visit["session_id"],
            "turn_count": visit["turn_count"],
            "minutes": int((time.time() - visit["started_at"]) / 60),
            "profile": {
                "name": visit["cook_name"],
                "skill_level": visit["skill_level"],
                "dietary": visit["dietary"],
                "dish": visit["dish"],
                "onboarding_complete": visit["onboarding_step"] == "complete",
            },
            "cooking": {
                "recipe": visit["recipe_name"],
                "step": visit["step"],
            },
            "facts_delivered": delivered,
            "episodes": [
                {
                    "subject": row["subject_name"],
                    "turns": row["turn_count"],
                    "facets": json.loads(row["facets"] or "[]"),
                }
                for row in episodes
            ],
        }

    def close(self):
        with self._lock:
            self._conn.close()
