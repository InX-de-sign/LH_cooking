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

Name, experience and dietary needs are remembered for the browser's user ID.
Today's dish and step remain specific to each visit. There is no scripted
setup questionnaire: the model learns profile details from the conversation
(update_cook_profile tool) or the Profile dialog.

The active recipe is either a catalog recipe (recipe_id, from knowledge.db) or
a recipe the cook uploaded (recipe_custom_id, stored here in custom_recipes).
"""

import json
from contextvars import ContextVar
import logging
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# How long a cook can be silent before their next question starts a fresh
# session. Generous, because a braise or a proving dough leaves long gaps, and
# the phone belongs to one person rather than being handed around.
SESSION_IDLE_SECONDS = int(os.getenv("SESSION_IDLE_SECONDS", str(3 * 60 * 60)))

# Episodes end on their own when the cook moves to a new subject, but a long
# gap on the same subject should still start a new one.
EPISODE_IDLE_SECONDS = int(os.getenv("EPISODE_IDLE_SECONDS", str(20 * 60)))

PROFILE_LIMITS = (("cook_name", 80), ("skill_level", 120), ("dietary", 1000))

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

CREATE TABLE IF NOT EXISTS profiles (
    user_id     TEXT PRIMARY KEY,
    cook_name   TEXT,
    skill_level TEXT,
    dietary     TEXT,
    updated_at  REAL NOT NULL
);

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

-- Recipes a cook uploaded (pasted text, a photo of a page, a text file),
-- parsed by the model into the same shape as a catalog recipe.
CREATE TABLE IF NOT EXISTS custom_recipes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    session_id  TEXT,
    name        TEXT NOT NULL,
    data        TEXT NOT NULL,    -- JSON: serves, time_minutes, ingredients, steps
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_custom_recipes_user ON custom_recipes(user_id, created_at DESC);
"""


class Visit:
    """One cook's session in the kitchen."""

    __slots__ = (
        "session_id", "user_id", "started_at", "last_active", "language", "turn_count",
        "cook_name", "skill_level", "dietary", "dish", "recipe_id", "recipe_name", "step",
        "onboarding_step", "onboarding_started_at", "onboarding_completed_at",
        "recipe_custom_id",
    )

    def __init__(self, session_id, user_id, started_at, last_active, language, turn_count,
                 cook_name=None, skill_level=None, dietary=None, dish=None,
                 recipe_id=None, recipe_name=None, step=None,
                 onboarding_step="complete", onboarding_started_at=None,
                 onboarding_completed_at=None, recipe_custom_id=None):
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
        self.recipe_custom_id = recipe_custom_id

    @property
    def minutes_so_far(self) -> int:
        return int((time.time() - self.started_at) / 60)

    @property
    def has_recipe(self) -> bool:
        return self.recipe_id is not None or self.recipe_custom_id is not None

    def __repr__(self):
        kind = "custom" if self.recipe_custom_id is not None else "catalog"
        return (f"Visit({self.session_id} user={self.user_id} turns={self.turn_count} "
                f"recipe={self.recipe_name!r} ({kind}) step={self.step})")


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
        # asyncio.to_thread propagates this per-WebSocket task context to
        # the existing synchronous store/retrieval methods.
        self._chat_id = ContextVar("cooking_chat_id", default=None)
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
                "step": "INTEGER", "chat_id": "TEXT",
                "onboarding_step": "TEXT NOT NULL DEFAULT 'name'",
                "onboarding_started_at": "REAL", "onboarding_completed_at": "REAL",
                "recipe_custom_id": "INTEGER",
            }),
            ("turns", {"vision": "TEXT"}),
        ):
            columns = {row["name"] for row in
                       self._conn.execute(f"PRAGMA table_info({table})").fetchall()}
            for name, definition in migrations.items():
                if name not in columns:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
        self._conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_visits_chat ON visits(user_id, chat_id) WHERE chat_id IS NOT NULL")
        # Preserve the latest completed profile when upgrading an existing database.
        self._conn.execute(
            "INSERT OR IGNORE INTO profiles (user_id, cook_name, skill_level, dietary, updated_at) "
            "SELECT v.user_id, v.cook_name, v.skill_level, v.dietary, v.last_active FROM visits v "
            "WHERE v.onboarding_step = 'complete' AND v.session_id = ("
            "SELECT latest.session_id FROM visits latest WHERE latest.user_id = v.user_id "
            "AND latest.onboarding_step = 'complete' ORDER BY latest.last_active DESC LIMIT 1)")

    @staticmethod
    def _visit_from_row(row: sqlite3.Row) -> Visit:
        return Visit(
            row["session_id"], row["user_id"], row["started_at"], row["last_active"],
            row["language"], row["turn_count"], row["cook_name"], row["skill_level"],
            row["dietary"], row["dish"], row["recipe_id"], row["recipe_name"], row["step"],
            row["onboarding_step"], row["onboarding_started_at"],
            row["onboarding_completed_at"], row["recipe_custom_id"],
        )

    def bind_chat(self, chat_id):
        """Bind a page's chat ID to its request task, never to the shared store."""
        return self._chat_id.set(chat_id)

    def reset_chat_binding(self, token):
        self._chat_id.reset(token)

    def current_visit(self, user_id: str, language: str = None) -> Visit:
        """Resume this cook's open session, or start a new one."""
        now = time.time()
        chat_id = self._chat_id.get()
        with self._lock:
            if chat_id:
                row = self._conn.execute("SELECT * FROM visits WHERE user_id = ? AND chat_id = ?",
                                         (user_id, chat_id)).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT * FROM visits WHERE user_id = ? ORDER BY last_active DESC LIMIT 1", (user_id,)).fetchone()

            if row and (chat_id or (now - row["last_active"]) <= SESSION_IDLE_SECONDS):
                return self._visit_from_row(row)

            session_id = f"{user_id}:{int(now)}:{uuid.uuid4().hex[:12]}"
            profile = self._conn.execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
            self._conn.execute(
                "INSERT INTO visits (session_id, user_id, started_at, last_active, language,"
                " turn_count, onboarding_step, cook_name, skill_level, dietary, chat_id)"
                " VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)",
                (session_id, user_id, now, now, language or "en", "complete",
                 profile["cook_name"] if profile else None, profile["skill_level"] if profile else None,
                 profile["dietary"] if profile else None, chat_id),
            )
            self._conn.commit()
            logger.info("New visit %s for %s", session_id, user_id)
            row = self._conn.execute("SELECT * FROM visits WHERE session_id = ?", (session_id,)).fetchone()
            return self._visit_from_row(row)

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
    def set_recipe(self, visit: Visit, recipe_id: Optional[int], recipe_name: Optional[str],
                   custom_id: Optional[int] = None):
        """Start (or clear, with None) the recipe for this session, at no step.

        Pass ``recipe_id`` for a catalog recipe or ``custom_id`` for an
        uploaded one; never both.
        """
        if recipe_id is not None and custom_id is not None:
            raise ValueError("A visit cooks either a catalog or an uploaded recipe, not both.")
        self._update_visit(visit.session_id, recipe_id=recipe_id, recipe_custom_id=custom_id,
                           recipe_name=recipe_name, step=None)
        visit.recipe_id, visit.recipe_custom_id = recipe_id, custom_id
        visit.recipe_name, visit.step = recipe_name, None
        logger.info("Visit %s now cooking %r", visit.session_id, recipe_name)

    def set_step(self, visit: Visit, step: Optional[int]):
        self._update_visit(visit.session_id, step=step)
        visit.step = step

    # -- uploaded recipes ---------------------------------------------------
    def save_custom_recipe(self, visit: Visit, recipe: Dict[str, Any]) -> int:
        """Store a parsed upload and return its id. ``recipe`` has a ``name``."""
        data = {key: value for key, value in recipe.items() if key != "name"}
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO custom_recipes (user_id, session_id, name, data, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (visit.user_id, visit.session_id, recipe["name"], json.dumps(data), time.time()))
            self._conn.commit()
            return cur.lastrowid

    def custom_recipe(self, custom_id: int, user_id: str) -> Optional[Dict[str, Any]]:
        """An uploaded recipe, only for the browser that uploaded it."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id, name, data FROM custom_recipes WHERE id = ? AND user_id = ?",
                (custom_id, user_id)).fetchone()
        if row is None:
            return None
        return {"id": row["id"], "name": row["name"], **json.loads(row["data"])}

    # -- visit profile -----------------------------------------------------
    def get_profile(self, user_id: str) -> Dict[str, Any]:
        visit = self.current_visit(user_id)
        return {"cook_name": visit.cook_name or "", "skill_level": visit.skill_level or "",
                "dietary": visit.dietary or ""}

    @staticmethod
    def _clean_profile(values: dict, partial: bool) -> Dict[str, Optional[str]]:
        if not isinstance(values, dict):
            raise ValueError("Provide a profile with name, experience and dietary needs.")
        clean = {}
        for field, limit in PROFILE_LIMITS:
            if partial and values.get(field) is None:
                continue
            value = values.get(field, "")
            if not isinstance(value, str) or len(value) > limit:
                raise ValueError(f"{field.replace('_', ' ').capitalize()} must be text of at most {limit} characters.")
            clean[field] = " ".join(value.split()) or None
        return clean

    def save_profile(self, user_id: str, values: dict) -> Dict[str, Any]:
        """Replace the whole profile (the Profile dialog)."""
        return self._store_profile(user_id, self._clean_profile(values, partial=False))

    def update_profile(self, user_id: str, values: dict) -> Dict[str, Any]:
        """Change only the fields given (the model's update_cook_profile tool)."""
        current = self.get_profile(user_id)
        merged = {key: current[key] or None for key, _ in PROFILE_LIMITS}
        merged.update(self._clean_profile(values, partial=True))
        return self._store_profile(user_id, merged)

    def _store_profile(self, user_id: str, clean: Dict[str, Optional[str]]) -> Dict[str, Any]:
        visit = self.current_visit(user_id)
        now = time.time()
        with self._lock:
            self._write_profile(user_id, clean, now)
            self._conn.execute(
                "UPDATE visits SET cook_name = ?, skill_level = ?, dietary = ?, last_active = ?"
                " WHERE session_id = ?",
                (clean["cook_name"], clean["skill_level"], clean["dietary"], now, visit.session_id))
            self._conn.commit()
        return {key: value or "" for key, value in clean.items()}

    def _write_profile(self, user_id, values, now):
        # Caller holds the lock and commits the surrounding transaction.
        self._conn.execute(
            "INSERT INTO profiles (user_id, cook_name, skill_level, dietary, updated_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET cook_name = excluded.cook_name, skill_level = excluded.skill_level, "
            "dietary = excluded.dietary, updated_at = excluded.updated_at",
            (user_id, values["cook_name"], values["skill_level"], values["dietary"], now))

    # -- history ----------------------------------------------------------
    def history(self, user_id: str, offset: int = 0, limit: int = 20) -> Dict[str, Any]:
        if type(offset) is not int or offset < 0:
            raise ValueError("History offset must be a non-negative integer.")
        limit = min(max(limit, 1), 50)
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_id, started_at, last_active, recipe_name, dish, turn_count FROM visits "
                "WHERE user_id = ? ORDER BY started_at DESC, session_id DESC LIMIT ? OFFSET ?",
                (user_id, limit + 1, offset)).fetchall()
        return {"sessions": [dict(row) for row in rows[:limit]],
                "next_offset": offset + limit if len(rows) > limit else None}

    def history_detail(self, user_id: str, session_id: str, offset: int = 0, limit: int = 40) -> Dict[str, Any]:
        if type(offset) is not int or offset < 0:
            raise ValueError("History offset must be a non-negative integer.")
        if not isinstance(session_id, str) or len(session_id) > 160:
            raise ValueError("Invalid history entry.")
        limit = min(max(limit, 1), 100)
        with self._lock:
            session = self._conn.execute(
                "SELECT session_id, started_at, last_active, recipe_name, dish, turn_count FROM visits "
                "WHERE user_id = ? AND session_id = ?", (user_id, session_id)).fetchone()
            if session is None:
                raise ValueError("History entry not found.")
            rows = self._conn.execute(
                "SELECT id, ts, user_text, assistant_text, vision FROM turns "
                "WHERE user_id = ? AND session_id = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (user_id, session_id, limit + 1, offset)).fetchall()
        return {"session": dict(session), "turns": [dict(row) for row in reversed(rows[:limit])],
                "next_offset": offset + limit if len(rows) > limit else None}

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
