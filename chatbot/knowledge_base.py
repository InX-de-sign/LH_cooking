# knowledge_base.py - Cooking knowledge store and retriever
"""
One SQLite database holding everything the coach knows, keyed the way home
cooks actually ask about it.

SOURCES
-------
Two hand-maintained YAML files, both reviewable in a diff:

  * ``recipes.yaml``  - recipes: ingredients, ordered steps, and facts
  * ``glossary.yaml`` - techniques, ingredients and kitchen topics

``knowledge.db`` is a disposable index built from them. Never edit it by hand.

THE UNIT OF RETRIEVAL IS A FACT
-------------------------------
Each fact is one finished sentence tagged with a *facet* -- the thing the cook
wanted to know:

    (recipe:Spaghetti Aglio e Olio, substitution) "Linguine works in place of..."
    (technique:al dente, doneness)                "Bite a piece: you want..."
    (topic:grease fire, safety)                   "Turn off the heat and cover..."

A question selects a subject and a facet, and only the matching facts are
retrieved. The conversation store keeps a ledger of facts already told, so
"any other tips?" produces new material rather than the same sentence again.

STEPS ARE NOT FACTS
-------------------
Recipe steps live in their own ordered table. They are navigated ("next",
"go back", "repeat") rather than retrieved, and they must never fall out of
reach because the ledger says they were already said -- repeating a step is
exactly what a cook with flour on their hands needs.

MATCHING
--------
Subjects resolve through an alias table: "aglio olio", "garlic pasta" and
"spaghetti aglio e olio" all map to one recipe, and the labels a VLM emits
("scallions", "raw chicken") map to glossary entries. Free text falls back to
FTS5 over the fact bodies.

Rebuild the database after editing either YAML file:

    python knowledge_base.py --rebuild
"""

import logging
import os
import re
import sqlite3
from contextlib import closing
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.getenv("KNOWLEDGE_DB_PATH", os.path.join(HERE, "knowledge.db"))

# Subject kinds. A recipe can be cooked step by step; the rest are glossary
# entries the cook can ask about at any point.
RECIPE = "recipe"
GLOSSARY_KINDS = ("technique", "ingredient", "topic")

# ---------------------------------------------------------------------------
# Facets - the axes a cook asks along.
# ---------------------------------------------------------------------------
# Ordered roughly by how often a question lands on them, which is also the
# order retrieval falls back through when a facet has nothing fresh left.
FACETS = (
    "overview",      # what the dish or term is
    "ingredients",   # what you need and how much
    "technique",     # how to do something
    "doneness",      # how to tell it is ready
    "heat",          # how hot
    "timing",        # how long
    "substitution",  # what to use instead
    "safety",        # food safety, fire, knives, burns
    "equipment",     # what to cook it in
    "storage",       # leftovers, fridge, reheating
    "tip",           # anything that makes it come out better
)

# Keywords are matched as substrings of the lowercased question, and the
# longest match wins -- so "how do i know" (doneness) beats "how do i"
# (technique), and "how much longer" (timing) beats "how much" (ingredients).
#
# Bare equipment nouns ("pan", "pot") are deliberately absent: they fire on
# "pan-seared chicken" and "potato", which are not equipment questions.
FACET_KEYWORDS: Dict[str, tuple] = {
    "overview": (
        "what is", "what's", "what are", "tell me about", "describe",
        "what kind of dish",
    ),
    "ingredients": (
        "ingredient", "ingredients", "what do i need", "how much", "how many",
        "quantity", "shopping list", "need for",
    ),
    "technique": (
        "how do i", "how to", "how should i", "what does it mean", "technique",
        "the right way", "properly",
    ),
    "doneness": (
        "is it done", "is it ready", "done yet", "ready yet", "cooked through",
        "is this done", "is that done", "are they done", "is this ready",
        "how do i know", "how will i know", "does this look", "does it look",
        "look right", "look okay", "look ok", "overcooked", "undercooked",
        "burnt", "burning", "too brown", "golden", "is this right",
    ),
    "heat": (
        "heat", "temperature", "how hot", "too hot", "not hot enough",
        "flame", "medium-high", "low heat", "high heat",
    ),
    "timing": (
        "how long", "how many minutes", "how much time", "how much longer",
        "minutes", "timer", "when do i", "when should i",
    ),
    "substitution": (
        "instead of", "instead", "can i use", "substitute", "replace", "swap", "don't have",
        "do not have", "dont have", "run out", "ran out", "out of",
        "alternative", "without",
    ),
    "safety": (
        "safe", "safety", "dangerous", "fire", "burn myself", "burned myself",
        "cut myself", "bacteria", "food poisoning", "allergy", "allergic",
        "smoking", "smoke", "flames", "boiling over", "spitting",
    ),
    "equipment": (
        "equipment", "tools", "utensil", "what pan", "which pan", "what pot",
        "which pot", "non-stick", "nonstick", "cast iron", "cook it in",
    ),
    "storage": (
        "store", "storage", "leftover", "leftovers", "fridge", "freeze",
        "freezer", "keep for", "reheat", "how long does it keep",
    ),
    "tip": (
        "tip", "tips", "trick", "secret", "advice", "make it better",
    ),
}

# Words carrying no retrieval signal. A question made only of these must return
# nothing, so the coach says it does not know instead of answering from
# whatever the full-text index ranked highest.
_STOPWORDS = frozenset("""
    the and but for with from that this these those there here they them their
    its it's you your yours our ours who whom whose what when where which why how
    have has had having are was were been being can could would should shall will
    does did doing done else more most some any all both each few other
    about into over under near next tell told say said know knows knew think
    like want need please thanks thank hello hey yes not don't didn't
    see seen look looking find finds found get gets got give gives given
    show shows shown come comes came take takes taken make makes made
    cook cooking cooked recipe dish food kitchen step steps thing things
    one two three now just still really okay
""".split())

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS subjects (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kind         TEXT NOT NULL,          -- recipe | technique | ingredient | topic
    name         TEXT UNIQUE NOT NULL,
    serves       INTEGER,
    time_minutes INTEGER,
    difficulty   TEXT
);

CREATE TABLE IF NOT EXISTS recipe_steps (
    recipe_id     INTEGER NOT NULL REFERENCES subjects(id),
    n             INTEGER NOT NULL,      -- 0-based
    text          TEXT NOT NULL,
    done_when     TEXT,
    timer_seconds INTEGER,
    PRIMARY KEY (recipe_id, n)
);

CREATE TABLE IF NOT EXISTS recipe_ingredients (
    recipe_id INTEGER NOT NULL REFERENCES subjects(id),
    n INTEGER NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (recipe_id, n)
);

-- Which glossary entries a recipe relies on. Lets "how do I emulsify?" asked
-- halfway through the pasta reach the technique's own facts.
CREATE TABLE IF NOT EXISTS recipe_terms (
    recipe_id INTEGER NOT NULL REFERENCES subjects(id),
    term_id   INTEGER NOT NULL REFERENCES subjects(id),
    PRIMARY KEY (recipe_id, term_id)
);

CREATE TABLE IF NOT EXISTS aliases (
    alias        TEXT NOT NULL,
    subject_id   INTEGER NOT NULL REFERENCES subjects(id),
    subject_kind TEXT NOT NULL,
    subject_name TEXT NOT NULL,
    PRIMARY KEY (alias, subject_id)
);
CREATE INDEX IF NOT EXISTS idx_aliases_alias ON aliases(alias);

CREATE TABLE IF NOT EXISTS facts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id   INTEGER NOT NULL REFERENCES subjects(id),
    subject_kind TEXT NOT NULL,
    subject_name TEXT NOT NULL,
    facet        TEXT NOT NULL,
    text         TEXT NOT NULL,
    source       TEXT
);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject_id, facet);
CREATE INDEX IF NOT EXISTS idx_facts_kind ON facts(subject_kind, facet);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    subject_name, facet, text,
    content='facts', content_rowid='id', tokenize='unicode61'
);
"""

MIN_FACT_CHARS = 20


def classify_facet(question: str) -> Optional[str]:
    """Pick the facet a question is aiming at, or None if it is open-ended.

    Longest keyword wins. Returning None is meaningful: it means "give them a
    spread of facets" rather than "no idea".
    """
    if not question:
        return None

    text = question.lower()
    best_facet, best_len = None, 0
    for facet, keywords in FACET_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text and len(keyword) > best_len:
                best_facet, best_len = facet, len(keyword)
    return best_facet


def _normalise(text: str) -> str:
    # Hyphens count as spaces so "garlic-butter" matches "garlic butter".
    return " ".join(str(text or "").lower().replace("-", " ").split())


def _content_terms(query: str) -> List[str]:
    """The words of a question that carry retrieval signal, lowercased."""
    return [
        term.lower() for term in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", query or "")
        if term.lower() not in _STOPWORDS
    ]


def _load_yaml(path: str) -> list:
    if not path or not os.path.exists(path):
        logger.warning("%s not found - skipped", path)
        return []
    import yaml
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or []


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

class _Builder:
    """Assembles knowledge.db from glossary.yaml plus recipes.yaml."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.ids: Dict[str, int] = {}      # normalised name -> subject id
        self.fact_count = 0

    # -- inserts ----------------------------------------------------------
    def add_subject(self, kind, name, serves=None, time_minutes=None,
                    difficulty=None) -> int:
        key = _normalise(name)
        if key in self.ids:
            raise ValueError(f"duplicate subject name: {name!r}")
        cur = self.conn.execute(
            "INSERT INTO subjects (kind, name, serves, time_minutes, difficulty)"
            " VALUES (?, ?, ?, ?, ?)",
            (kind, name, serves, time_minutes, difficulty),
        )
        subject_id = cur.lastrowid
        self.ids[key] = subject_id
        self.add_alias(name, subject_id, kind, name)
        return subject_id

    def add_alias(self, alias, subject_id, kind, name):
        alias = _normalise(alias)
        if not alias or len(alias) < 2:
            return
        self.conn.execute(
            "INSERT OR IGNORE INTO aliases (alias, subject_id, subject_kind, subject_name)"
            " VALUES (?, ?, ?, ?)",
            (alias, subject_id, kind, name),
        )

    def add_fact(self, subject_id, kind, name, facet, text, source):
        text = " ".join(str(text or "").split())
        if len(text) < MIN_FACT_CHARS:
            return
        if facet not in FACETS:
            logger.warning("%s: unknown facet %r - fact skipped", name, facet)
            return
        self.conn.execute(
            "INSERT INTO facts (subject_id, subject_kind, subject_name, facet, text, source)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (subject_id, kind, name, facet, text, source),
        )
        self.fact_count += 1

    def _add_entry_facts(self, subject_id, kind, name, entry, default_source):
        for fact in entry.get("facts") or []:
            facet = (fact.get("facet") or "").strip()
            text = (fact.get("text") or "").strip()
            if facet and text:
                source = (fact.get("source") or default_source).strip()
                self.add_fact(subject_id, kind, name, facet, text, source)

    # -- sources ----------------------------------------------------------
    def load_glossary(self, entries: list):
        for entry in entries:
            name = (entry.get("name") or "").strip()
            if not name:
                continue
            kind = (entry.get("kind") or "technique").strip().lower()
            if kind not in GLOSSARY_KINDS:
                logger.warning("%s: unknown kind %r - treated as topic", name, kind)
                kind = "topic"
            subject_id = self.add_subject(kind, name)
            for alias in entry.get("aliases") or []:
                self.add_alias(alias, subject_id, kind, name)
            self._add_entry_facts(subject_id, kind, name, entry, "glossary.yaml")

    def load_recipes(self, entries: list):
        for entry in entries:
            name = (entry.get("name") or "").strip()
            if not name:
                continue
            serves = entry.get("serves")
            minutes = entry.get("time_minutes")
            difficulty = (entry.get("difficulty") or "").strip() or None
            recipe_id = self.add_subject(RECIPE, name, serves, minutes, difficulty)

            for alias in entry.get("aliases") or []:
                self.add_alias(alias, recipe_id, RECIPE, name)

            steps = [s for s in entry.get("steps") or [] if (s or {}).get("text")]
            for n, step in enumerate(steps):
                self.conn.execute(
                    "INSERT INTO recipe_steps (recipe_id, n, text, done_when, timer_seconds)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (recipe_id, n, " ".join(step["text"].split()),
                     " ".join((step.get("done_when") or "").split()) or None,
                     step.get("timer_seconds")),
                )

            # The two generated facts. Everything else is written by hand.
            bits = []
            if serves:
                bits.append(f"serves {serves}")
            if minutes:
                bits.append(f"takes about {minutes} minutes")
            if difficulty:
                bits.append(f"is {difficulty}")
            if bits:
                summary = ", ".join(bits[:-1]) + (" and " if len(bits) > 1 else "") + bits[-1]
                self.add_fact(recipe_id, RECIPE, name, "overview",
                              f"{name} {summary}, in {len(steps)} steps.",
                              "recipes.yaml:summary")

            ingredients = [str(i).strip() for i in entry.get("ingredients") or [] if i]
            self.conn.executemany(
                "INSERT INTO recipe_ingredients (recipe_id, n, text) VALUES (?, ?, ?)",
                [(recipe_id, n, text) for n, text in enumerate(ingredients)],
            )
            if ingredients:
                serving = f" (serves {serves})" if serves else ""
                self.add_fact(recipe_id, RECIPE, name, "ingredients",
                              f"For {name}{serving} you need: {'; '.join(ingredients)}.",
                              "recipes.yaml:ingredients")

            self._add_entry_facts(recipe_id, RECIPE, name, entry, "recipes.yaml")

            for term in entry.get("techniques") or []:
                term_id = self._resolve_term(term)
                if term_id is None:
                    logger.warning("%s: technique %r is not in glossary.yaml", name, term)
                    continue
                self.conn.execute(
                    "INSERT OR IGNORE INTO recipe_terms (recipe_id, term_id) VALUES (?, ?)",
                    (recipe_id, term_id),
                )

    def _resolve_term(self, name: str) -> Optional[int]:
        key = _normalise(name)
        if key in self.ids:
            return self.ids[key]
        row = self.conn.execute(
            "SELECT subject_id FROM aliases WHERE alias = ? AND subject_kind != ?",
            (key, RECIPE),
        ).fetchone()
        return row[0] if row else None

    def dedupe_aliases(self):
        """Make every alias point at exactly one subject.

        "chicken" is both a glossary ingredient and an alias a recipe author
        might reach for. Left ambiguous, resolution picks whichever row the
        index happens to return. A recipe wins over a glossary entry, since
        naming a dish is the stronger signal; otherwise the subject with the
        most facts wins, because it can answer a follow-up.
        """
        rows = self.conn.execute(
            "SELECT alias FROM aliases GROUP BY alias HAVING COUNT(*) > 1"
        ).fetchall()
        for (alias,) in rows:
            best = self.conn.execute(
                "SELECT a.subject_id,"
                "       (SELECT COUNT(*) FROM facts f WHERE f.subject_id = a.subject_id) AS n"
                "  FROM aliases a WHERE a.alias = ?"
                " ORDER BY (a.subject_kind = ?) DESC, n DESC, a.subject_id ASC LIMIT 1",
                (alias, RECIPE),
            ).fetchone()
            self.conn.execute(
                "DELETE FROM aliases WHERE alias = ? AND subject_id != ?",
                (alias, best[0]),
            )
        if rows:
            logger.info("Resolved %d ambiguous aliases", len(rows))

    def reindex(self):
        self.conn.execute("INSERT INTO facts_fts(facts_fts) VALUES('rebuild')")


def build_knowledge_db(dest_path: str = DEFAULT_DB, recipes_yaml: str = None,
                       glossary_yaml: str = None) -> str:
    """(Re)build knowledge.db from glossary.yaml and recipes.yaml."""
    recipes_yaml = recipes_yaml or os.path.join(HERE, "recipes.yaml")
    glossary_yaml = glossary_yaml or os.path.join(HERE, "glossary.yaml")

    # Build away from the live file, then copy the complete SQLite database in
    # one backup operation. Renaming knowledge.db fails on Windows while another
    # process still has it open (including a recently stopped local server).
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)

    builder = _Builder(conn)
    # Glossary first, so recipes can link to the techniques they name.
    builder.load_glossary(_load_yaml(glossary_yaml))
    builder.load_recipes(_load_yaml(recipes_yaml))
    builder.dedupe_aliases()
    builder.reindex()
    conn.commit()

    counts = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("subjects", "recipe_steps", "recipe_terms", "facts", "aliases")
    }
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    with closing(sqlite3.connect(dest_path, timeout=10)) as target:
        conn.backup(target)
    conn.close()

    logger.info("Built %s: %s", dest_path, counts)
    return dest_path


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class Subject:
    """A resolved thing the cook is talking about."""

    __slots__ = ("type", "id", "name", "source")

    def __init__(self, type_: str, id_: int, name: str, source: str = "text"):
        self.type = type_          # recipe | technique | ingredient | topic
        self.id = id_
        self.name = name
        self.source = source

    @property
    def key(self):
        return (self.type, self.id)

    @property
    def is_recipe(self) -> bool:
        return self.type == RECIPE

    def __eq__(self, other):
        return isinstance(other, Subject) and self.key == other.key

    def __repr__(self):
        return f"Subject({self.type}:{self.id} {self.name!r} via {self.source})"


class KnowledgeBase:
    """Read side of knowledge.db: resolve a subject, then retrieve facts."""

    def __init__(self, db_path: str = None, auto_build: bool = True):
        self.db_path = db_path or DEFAULT_DB
        if auto_build:
            sources = (os.path.join(HERE, "recipes.yaml"),
                       os.path.join(HERE, "glossary.yaml"), __file__)
            db_exists = os.path.exists(self.db_path)
            source_is_newer = db_exists and any(
                os.path.exists(path) and os.path.getmtime(path) > os.path.getmtime(self.db_path)
                for path in sources
            )
            if not db_exists or source_is_newer:
                logger.info("knowledge.db missing or outdated - building it now")
                try:
                    build_knowledge_db(self.db_path)
                except PermissionError:
                    if not db_exists:
                        raise
                    logger.warning("knowledge.db is in use; restart the running app to load updated recipes")
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Aliases are small and hit on every turn; keep them in memory and
        # match longest-first so "garlic pasta" wins over "garlic".
        self._aliases = sorted(
            (
                (row["alias"], row["subject_kind"], row["subject_id"], row["subject_name"])
                for row in self._conn.execute(
                    "SELECT alias, subject_kind, subject_id, subject_name FROM aliases")
            ),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        logger.info("KnowledgeBase ready: %d aliases, %d facts",
                    len(self._aliases), self.fact_count())

    def fact_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    # -- subject resolution ----------------------------------------------
    def resolve_subject(self, text: str, prefer: str = None) -> Optional[Subject]:
        """Find what the cook is referring to.

        `prefer` biases towards a subject kind when two aliases of the same
        length both match.
        """
        if not text:
            return None
        haystack = _normalise(text)

        best = None
        for alias, kind, subject_id, name in self._aliases:
            if alias not in haystack:
                continue
            # Guard against matching inside a longer word ("egg" in "eggplant"),
            # while still accepting a plural ("eggs", "tomatoes").
            if alias.isascii() and not re.search(rf"\b{re.escape(alias)}(?:e?s)?\b", haystack):
                continue
            if best is None:
                best = (alias, kind, subject_id, name)
            elif prefer and kind == prefer and best[1] != prefer and len(alias) == len(best[0]):
                best = (alias, kind, subject_id, name)
            else:
                break
        if not best:
            return None

        _, kind, subject_id, name = best
        return Subject(kind, subject_id, name, source="text")

    def resolve_recipe(self, text: str) -> Optional[Subject]:
        """A recipe named in the text, ignoring glossary matches."""
        if not text:
            return None
        haystack = _normalise(text)
        for alias, kind, subject_id, name in self._aliases:
            if kind != RECIPE or alias not in haystack:
                continue
            if alias.isascii() and not re.search(rf"\b{re.escape(alias)}(?:e?s)?\b", haystack):
                continue
            return Subject(kind, subject_id, name, source="text")
        return None

    def subject_for_label(self, label: str) -> Optional[Subject]:
        """Resolve a label from the camera (a VLM item such as ``scallions``)."""
        if not label:
            return None
        cleaned = _normalise(str(label).replace("_", " "))
        subject = self.resolve_subject(cleaned)
        if subject:
            subject.source = "camera"
        return subject

    def subject_by_id(self, subject_id: int, source: str = "episode") -> Optional[Subject]:
        row = self._conn.execute(
            "SELECT id, kind, name FROM subjects WHERE id = ?", (subject_id,)
        ).fetchone()
        return Subject(row["kind"], row["id"], row["name"], source=source) if row else None

    # -- recipes ----------------------------------------------------------
    def steps(self, recipe_id: int) -> List[Dict[str, Any]]:
        return [dict(row) for row in self._conn.execute(
            "SELECT n, text, done_when, timer_seconds FROM recipe_steps"
            " WHERE recipe_id = ? ORDER BY n", (recipe_id,))]

    def recipes(self) -> List[Dict[str, Any]]:
        recipes = [dict(row) for row in self._conn.execute(
            "SELECT s.id, s.name, s.serves, s.time_minutes, s.difficulty,"
            "       (SELECT COUNT(*) FROM recipe_steps r WHERE r.recipe_id = s.id) AS steps,"
            "       (SELECT COUNT(*) FROM facts f WHERE f.subject_id = s.id) AS facts"
            "  FROM subjects s WHERE s.kind = ? ORDER BY s.name", (RECIPE,))]
        for recipe in recipes:
            recipe["ingredients"] = self.ingredients(recipe["id"])
            recipe["aliases"] = [row[0] for row in self._conn.execute(
                "SELECT alias FROM aliases WHERE subject_id = ? ORDER BY alias", (recipe["id"],))]
        return recipes

    def ingredients(self, recipe_id: int) -> List[str]:
        return [row[0] for row in self._conn.execute(
            "SELECT text FROM recipe_ingredients WHERE recipe_id = ? ORDER BY n", (recipe_id,))]

    def recipe_details(self, recipe_id: int) -> Dict[str, Any]:
        row = self._conn.execute(
            "SELECT serves, time_minutes FROM subjects WHERE id = ? AND kind = ?",
            (recipe_id, RECIPE),
        ).fetchone()
        return {**dict(row), "ingredients": self.ingredients(recipe_id)} if row else {}

    def ingredients_text(self, recipe_id: int) -> Optional[str]:
        row = self._conn.execute(
            "SELECT text FROM facts WHERE subject_id = ? AND facet = 'ingredients' LIMIT 1",
            (recipe_id,)).fetchone()
        return row["text"] if row else None

    # -- fact retrieval ---------------------------------------------------
    def retrieve(self, subject: Optional[Subject], facet: str = None,
                 query: str = None, exclude_ids=None, limit: int = 4,
                 recipe: Optional[Subject] = None) -> List[Dict[str, Any]]:
        """Fetch the facts most likely to answer this question.

        A cascade that exhausts every *facet-specific* source before it
        broadens, so "can I swap the parsley?" reaches the recipe's
        substitution notes before it falls back to anything at all about
        parsley:

          1. the asked-for facet on the subject
          2. the asked-for facet on the recipe being cooked
          3. the asked-for facet on that recipe's techniques
          4. the asked-for facet on kitchen topics (safety, storage...)
          5. any facet on the subject, in FACETS priority order
          6. full-text search, for subjects we could not pin down

        `exclude_ids` drops facts already told to this cook.
        """
        exclude = set(exclude_ids or ())
        results: List[Dict[str, Any]] = []
        seen = set()

        def take(rows) -> bool:
            """Append rows until the limit is hit; True means 'done'."""
            for row in rows:
                if row["id"] in exclude or row["id"] in seen:
                    continue
                seen.add(row["id"])
                results.append(dict(row))
                if len(results) >= limit:
                    return True
            return False

        columns = "id, subject_name, facet, text"

        def facts_on(subject_id, want_facet):
            sql = f"SELECT {columns} FROM facts WHERE subject_id = ?"
            params: List[Any] = [subject_id]
            if want_facet:
                sql += " AND facet = ?"
                params.append(want_facet)
            else:
                sql += (" ORDER BY CASE facet "
                        + " ".join(f"WHEN ? THEN {i}" for i in range(len(FACETS)))
                        + " ELSE 99 END")
                params.extend(FACETS)
            return self._conn.execute(sql, params)

        # Steps 3 and 4 are broad -- every doneness fact on every technique the
        # recipe uses -- so rows that share a word with the question go first.
        # Otherwise "is the garlic burnt?" is answered alongside "al dente".
        terms = _content_terms(query)

        def ranked(rows):
            rows = list(rows)
            if terms:
                rows.sort(key=lambda row: not any(
                    t in row["text"].lower() or t in row["subject_name"].lower()
                    for t in terms))
            return rows

        def recipe_term_facts(recipe_id, want_facet):
            return ranked(self._conn.execute(
                f"SELECT {columns} FROM facts WHERE facet = ? AND subject_id IN"
                " (SELECT term_id FROM recipe_terms WHERE recipe_id = ?)",
                (want_facet, recipe_id)))

        def topic_facts(want_facet):
            return ranked(self._conn.execute(
                f"SELECT {columns} FROM facts WHERE subject_kind = 'topic' AND facet = ?",
                (want_facet,)))

        if recipe is not None and subject is not None and recipe == subject:
            recipe = None  # step 2 would only repeat step 1

        if facet:
            if subject is not None and take(facts_on(subject.id, facet)):
                return results
            if recipe is not None and take(facts_on(recipe.id, facet)):
                return results
            linked = recipe or (subject if subject is not None and subject.is_recipe else None)
            if linked is not None and take(recipe_term_facts(linked.id, facet)):
                return results
            if take(topic_facts(facet)):
                return results

        # No facet, or nothing fresh on it. Before dumping the subject's facts
        # in priority order, prefer ones that share a word with the question,
        # across the subject, the recipe and its techniques -- so "how salty
        # should the water be?" reaches the pasta-water tip rather than the
        # recipe's substitution notes.
        if terms:
            pool = []
            for source in (subject, recipe):
                if source is not None:
                    pool.extend(facts_on(source.id, None))
            linked = recipe or (subject if subject is not None and subject.is_recipe else None)
            if linked is not None:
                pool.extend(self._conn.execute(
                    f"SELECT {columns} FROM facts WHERE subject_id IN"
                    " (SELECT term_id FROM recipe_terms WHERE recipe_id = ?)", (linked.id,)))
            def overlap(row):
                haystack = f"{row['subject_name']} {row['text']}".lower()
                return sum(t in haystack for t in terms)
            matching = sorted((row for row in pool if overlap(row)), key=overlap, reverse=True)
            if take(matching):
                return results

        if subject is not None and take(facts_on(subject.id, None)):
            return results

        if query:
            take(self._search(query, facet, limit))

        return results

    def _search(self, query: str, facet: str = None, limit: int = 4):
        """FTS5 over fact bodies.

        Stopwords are dropped before the match, and an all-stopword query
        returns nothing rather than falling back to bm25 noise -- an empty
        result is what tells the coach to say it does not have notes on this.
        """
        terms = _content_terms(query)
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        sql = (
            "SELECT f.id, f.subject_name, f.facet, f.text"
            " FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid"
            " WHERE facts_fts MATCH ?"
        )
        params: List[Any] = [match]
        if facet:
            sql += " AND f.facet = ?"
            params.append(facet)
        sql += " ORDER BY bm25(facts_fts, 4.0, 2.0, 1.0) LIMIT ?"
        params.append(limit * 3)
        try:
            return list(self._conn.execute(sql, params))
        except sqlite3.OperationalError as exc:
            logger.warning("FTS query failed for %r: %s", query, exc)
            return []

    def close(self):
        self._conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build or inspect knowledge.db")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild from recipes.yaml + glossary.yaml")
    parser.add_argument("--ask", help="resolve a question and show what would be retrieved")
    parser.add_argument("--recipe", help="treat this recipe as the one being cooked")
    args = parser.parse_args()

    if args.rebuild or not os.path.exists(DEFAULT_DB):
        build_knowledge_db()

    if args.ask:
        kb = KnowledgeBase()
        subject = kb.resolve_subject(args.ask)
        recipe = kb.resolve_recipe(args.recipe) if args.recipe else None
        facet = classify_facet(args.ask)
        print(f"subject={subject} recipe={recipe} facet={facet}")
        for fact in kb.retrieve(subject or recipe, facet, query=args.ask, recipe=recipe):
            print(f"  [{fact['facet']}] {fact['subject_name']}: {fact['text'][:120]}")
