# context_builder.py - Turns a cook's utterance into a layered prompt context
"""
The join between what the coach knows (knowledge_base), what it remembers
(conversation_state) and what the camera sees (vision).

The model drives the conversation; this module only gathers what it needs to
answer well, in labelled layers:

    [COOK]      profile: name, experience, allergies (or "not known yet")
    [RECIPE]    the active recipe: ingredients, every step, and where they are
    [CATALOG]   only when nothing is being cooked: what can be suggested
    [NOW]       what this turn is about, and whether camera frames are attached
    [FACTS]     retrieved notes, each tagged with the facet it answers
    [QUESTION]  the utterance itself

The last few exchanges are sent as real chat messages (see streaming_openai),
not pasted into this context.

STEP NAVIGATION FAST PATH
-------------------------
Short, unambiguous commands -- "next", "back", "repeat", "step 4", "start" --
move the step pointer before anything is rendered, so they are instant and
always right. Anything longer or fuzzier ("I've finished chopping, what now?")
goes to the model, which moves the pointer with its go_to_step tool. Choosing,
switching and stopping recipes is always the model's call (choose_recipe /
stop_recipe), so "give me suggestions" is never mistaken for a dish name.

GROUNDING
---------
Quantities, times and temperatures for the recipe come only from [RECIPE] and
[FACTS]. When retrieval comes back empty the context says so, and the coach
may offer general cooking advice but must not invent recipe numbers.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from knowledge_base import KnowledgeBase, Subject, classify_facet
from conversation_state import ConversationStore, Episode, Visit
from web_search import explicit_search

logger = logging.getLogger(__name__)

MAX_FACTS = 4
MAX_NAV_WORDS = 6

# Pronouns point back into the CONVERSATION: "how long does IT rest?" means
# whatever we were just discussing.
ANAPHORIC_MARKERS = ("it", "its", "it's", "them", "they", "their")

# A question that only makes sense as a continuation. Kept narrow: a false
# positive carries the previous subject into a question not about it.
FOLLOW_UP_MARKERS = (
    "tell me more", "more about", "anything else", "what else", "any more",
    "any other", "what about", "how about",
)

# Openers and pleasantries. A greeting retrieves no facts, and the grounding
# guard would otherwise answer "thanks!" with "my notes don't cover that".
SOCIAL_PATTERNS = (
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
    "thank you", "thanks", "cheers", "bye", "goodbye", "see you", "how are you",
)
MAX_SOCIAL_WORDS = 5

# --- Step navigation --------------------------------------------------------
# Checked in this order; the first match wins. "start again" is a restart,
# not a repeat, so START comes before REPEAT.
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "first": 1,
    "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}
NAV_PATTERNS = (
    ("start", r"\b(let'?s (start|begin|go)|start cooking|start over|start again|"
              r"from the (top|beginning|start)|i'?m ready to start|begin)\b"),
    ("previous", r"\b(go back|previous step|step before|back a step|back one step|"
                 r"before that)\b"),
    ("repeat", r"\b(repeat|say (that|it) again|again please|what was that|"
               r"come again|what step|which step|where (was|am) i|pardon)\b"),
    ("next", r"\b(next|what'?s next|what now|then what|what do i do (now|next)|"
             r"move on|carry on|i'?m done|i am done|all done|done|finished|"
             r"ready for the next)\b"),
)
_GOTO = re.compile(
    r"\bstep (\d{1,2}|" + "|".join(_NUMBER_WORDS) + r")\b")
# "ok start", "yes let's go", "i'm ready" -- a whole message that only says
# go. "how do I start the sauce" is not one. START (re)starts at step 1;
# READY only starts a recipe that has not begun (see ContextBuilder.build).
_GO_PREFIX = (r"^(ok(ay)?|yes|yeah|yep|sure|alright|right|great|cool|good)?\s*"
              r"(i'?m |i am |we'?re )?(let'?s )?")
START_COMMAND = re.compile(_GO_PREFIX + r"(start|begin|start cooking|get started)( now| please| then)?$")
READY_COMMAND = re.compile(_GO_PREFIX + r"(ready|go|do it|got everything)( now| please| then)?$")


def _clean(text: str) -> str:
    return re.sub(r"[^\w\s']", " ", (text or "").lower()).strip()


def detect_navigation(text: str) -> Optional[tuple]:
    """("next",) / ("previous",) / ("repeat",) / ("start",) / ("goto", n) for a
    short step command, else None.

    Only short commands qualify: "I'm done chopping, does this look right?"
    is a question for the model (and the camera), not a bare "next". A
    doneness question ("is it done?") is never navigation.
    """
    cleaned = _clean(text)
    if not cleaned or len(cleaned.split()) > MAX_NAV_WORDS:
        return None
    if classify_facet(text) == "doneness":
        return None
    if explicit_search(text):
        return None
    if re.search(r"\bnext time\b", cleaned) or "?" in (text or "") and "next" not in cleaned:
        return None
    if START_COMMAND.match(cleaned):
        return ("start",)
    if READY_COMMAND.match(cleaned):
        return ("ready",)
    if cleaned in ("back", "back please"):
        return ("previous",)
    goto = _GOTO.search(cleaned)
    if goto:
        raw = goto.group(1)
        return ("goto", int(raw) if raw.isdigit() else _NUMBER_WORDS[raw])
    for intent, pattern in NAV_PATTERNS:
        if re.search(pattern, cleaned):
            return (intent,)
    return None


def is_social(text: str) -> bool:
    """True for a bare greeting or pleasantry carrying no question."""
    stripped = _clean(text)
    if not stripped or len(stripped.split()) > MAX_SOCIAL_WORDS:
        return False
    return any(re.search(rf"\b{re.escape(p)}\b", stripped) for p in SOCIAL_PATTERNS)


def _has_marker(text: str, markers) -> bool:
    cleaned = _clean(text)
    return any(re.search(rf"\b{re.escape(m)}\b", cleaned) for m in markers)


def is_anaphoric(text: str) -> bool:
    return _has_marker(text, ANAPHORIC_MARKERS)


def is_follow_up(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in FOLLOW_UP_MARKERS)


# --- Language ----------------------------------------------------------------
# English only for now. The lever is kept so a second language is a config
# change plus content, not a code change.
DEFAULT_LANGUAGE = os.getenv("GUIDE_DEFAULT_LANGUAGE", "en").strip().lower()


def detect_language(text: str) -> str:
    return DEFAULT_LANGUAGE


# --- The active recipe ----------------------------------------------------------
@dataclass
class ActiveRecipe:
    """A catalog or uploaded recipe in one shape, for prompts and the phone."""
    kind: str                       # "catalog" | "custom"
    id: int
    name: str
    steps: List[Dict[str, Any]]     # {"n", "text", "done_when", "timer_seconds"}
    ingredients: List[str] = field(default_factory=list)
    serves: Optional[int] = None
    time_minutes: Optional[int] = None
    subject: Optional[Subject] = None   # knowledge-base subject, catalog only

    @property
    def custom(self) -> bool:
        return self.kind == "custom"


_QUANTITY = re.compile(
    r"^(about |around |a |an |\d+[\d/.,\s-]*|½|¼|¾)*"
    r"(g|kg|grams?|ml|millilitres?|milliliters?|l|litres?|liters?|tbsp|tablespoons?|"
    r"tsp|teaspoons?|cups?|cloves?|pinch(es)?|handfuls?|slices?|pieces?|cans?|tins?|"
    r"sticks?|stalks?|sprigs?|bunch(es)?|large|medium|small|of)?\b\s*", re.I)


def ingredient_core(text: str) -> str:
    """'200 grams spaghetti, dried' -> 'spaghetti'; for the compact catalog."""
    core = re.split(r"[,(]", text or "", maxsplit=1)[0].strip()
    for _ in range(3):
        stripped = _QUANTITY.sub("", core, count=1).strip()
        if stripped == core:
            break
        core = stripped
    return core.lower() or (text or "").lower()


class TurnContext:
    """Everything decided about one turn, plus the rendered prompt context."""

    __slots__ = ("visit", "episode", "subject", "subject_source", "facet", "language",
                 "facts", "text", "grounded", "message", "recipe", "nav",
                 "observation", "social", "frames")

    def __init__(self, **fields):
        for name in self.__slots__:
            setattr(self, name, fields.get(name))

    @property
    def fact_ids(self) -> List[int]:
        return [fact["id"] for fact in self.facts or []]

    @property
    def subject_name(self) -> Optional[str]:
        return self.subject.name if self.subject else None

    @property
    def step(self) -> Optional[int]:
        return self.visit.step if self.visit else None

    @property
    def steps(self) -> List[Dict[str, Any]]:
        return self.recipe.steps if self.recipe else []

    def __repr__(self):
        return (f"TurnContext(subject={self.subject_name!r} via {self.subject_source},"
                f" facet={self.facet}, nav={self.nav}, step={self.step},"
                f" facts={len(self.facts or [])}, frames={self.frames or 0},"
                f" observation={'yes' if self.observation else 'no'})")


class ContextBuilder:
    """Resolves a turn against the knowledge base and the conversation so far."""

    def __init__(self, knowledge: KnowledgeBase, store: ConversationStore):
        self.knowledge = knowledge
        self.store = store

    # -- recipe and step ---------------------------------------------------
    def active_recipe(self, visit: Visit) -> Optional[ActiveRecipe]:
        if visit.recipe_custom_id is not None:
            data = self.store.custom_recipe(visit.recipe_custom_id, visit.user_id)
            if data is None:
                return None
            return ActiveRecipe(
                kind="custom", id=data["id"], name=data["name"],
                steps=[{"n": i + 1, "text": s.get("text", ""), "done_when": s.get("done_when"),
                        "timer_seconds": s.get("timer_seconds")}
                       for i, s in enumerate(data.get("steps") or [])],
                ingredients=list(data.get("ingredients") or []),
                serves=data.get("serves"), time_minutes=data.get("time_minutes"))
        if visit.recipe_id is None:
            return None
        subject = self.knowledge.subject_by_id(visit.recipe_id, source="recipe")
        if subject is None:
            return None
        details = self.knowledge.recipe_details(visit.recipe_id)
        return ActiveRecipe(
            kind="catalog", id=visit.recipe_id, name=subject.name,
            steps=self.knowledge.steps(visit.recipe_id),
            ingredients=details.get("ingredients", []),
            serves=details.get("serves"), time_minutes=details.get("time_minutes"),
            subject=subject)

    def navigate(self, visit: Visit, nav: tuple, steps: List[Dict[str, Any]]) -> Optional[int]:
        """Move the step pointer. Returns the new 0-based step."""
        intent = nav[0]
        total = len(steps)
        current = visit.step
        if intent == "start" or (intent == "ready" and current is None):
            target = 0
        elif intent == "next":
            target = 0 if current is None else min(current + 1, total)
        elif intent == "previous":
            target = 0 if current is None else max(min(current, total) - 1, 0)
        elif intent == "repeat":
            target = 0 if current is None else current
        elif intent == "goto":
            target = min(max(nav[1] - 1, 0), max(total - 1, 0))
        else:
            return current
        if target != current:
            self.store.set_step(visit, target)
        return target

    # -- subject ----------------------------------------------------------
    def resolve_subject(self, message: str, episode: Optional[Episode],
                        recipe: Optional[ActiveRecipe]):
        """What this turn is about: named > pronoun > follow-up > the recipe."""
        named = self.knowledge.resolve_subject(message)
        if named is not None:
            return named, "named"
        carried = self._episode_subject(episode)
        if (is_anaphoric(message) or is_follow_up(message)) and carried is not None:
            return carried, "carried"
        if recipe is not None and recipe.subject is not None:
            return recipe.subject, "recipe"
        if carried is not None and classify_facet(message) is not None:
            return carried, "carried"
        return None, "none"

    def _episode_subject(self, episode: Optional[Episode]) -> Optional[Subject]:
        if episode is None or episode.subject_id is None:
            return None
        return Subject(episode.subject_type, episode.subject_id,
                       episode.subject_name, source="episode")

    # -- build ------------------------------------------------------------
    def build(self, user_id: str, message: str, observation=None, frames: int = 0,
              allow_nav: bool = True) -> TurnContext:
        visit = self.store.current_visit(user_id)
        language = visit.language or detect_language(message)
        recipe = self.active_recipe(visit)

        nav = detect_navigation(message) if allow_nav and recipe and recipe.steps else None
        if nav == ("ready",) and visit.step is not None:
            nav = None   # mid-recipe "ready" means something; let the model read it
        if nav:
            self.navigate(visit, nav, recipe.steps)

        active = self.store.active_episode(visit.session_id)
        subject, subject_source = self.resolve_subject(message, active, recipe)
        facet = classify_facet(message)
        social = not nav and subject_source in ("none", "recipe") and is_social(message)

        episode = None
        if not social and not nav and subject is not None:
            episode = self.store.open_episode_for(
                visit.session_id, user_id, subject, opened_by=subject_source)

        # Navigation reads a step; the step is the content, so no facts.
        facts: List[Dict[str, Any]] = []
        if not social and not nav:
            kb_recipe = recipe.subject if recipe else None
            delivered = self.store.delivered_fact_ids(visit.session_id)
            facts = self.knowledge.retrieve(subject, facet, query=message,
                                            exclude_ids=delivered, limit=MAX_FACTS,
                                            recipe=kb_recipe)
            # Everything relevant has been said. Rather than answer with
            # nothing, let the ledger go and repeat the best material.
            if not facts and subject is not None and delivered:
                self.store.forget_delivered(visit.session_id)
                facts = self.knowledge.retrieve(subject, facet, query=message,
                                                limit=MAX_FACTS, recipe=kb_recipe)

        turn = TurnContext(
            visit=visit, episode=episode, subject=subject, subject_source=subject_source,
            facet=facet, language=language, facts=facts, message=message,
            recipe=recipe, nav=nav, observation=observation, social=social,
            frames=frames, grounded=bool(social or nav or facts),
        )
        turn.text = self.render(turn)
        return turn

    # -- render -----------------------------------------------------------
    def render(self, turn: TurnContext) -> str:
        lines: List[str] = []
        lines += ["[COOK]", *self.cook_lines(turn.visit), ""]
        lines += ["[RECIPE]", *self.recipe_lines(turn.visit, turn.recipe), ""]
        if turn.recipe is None:
            lines += ["[CATALOG]", *self.catalog_lines(), ""]
        lines += ["[NOW]", *self._now_lines(turn), ""]

        lines.append("[FACTS]")
        if turn.social:
            lines.append("Small talk, not a question. Reply warmly in one short sentence.")
        elif turn.nav:
            lines.append("Step navigation: the current step in [RECIPE] is the content.")
        elif turn.facts:
            lines.append("Use these for anything specific to this recipe. Do not invent "
                         "quantities, times or temperatures.")
            for fact in turn.facts:
                lines.append(f"- ({fact['facet']}) {fact['subject_name']}: {fact['text']}")
        else:
            # The grounding guard. Left implicit, the model invents recipe
            # numbers with total confidence.
            lines.append("The recipe notes do not cover this question. You may give "
                         "general cooking advice. Do NOT invent quantities, times or "
                         "temperatures for this recipe. On food safety, if unsure, "
                         "recommend the cautious option.")
        lines.append("")
        lines += ["[QUESTION]", turn.message]
        return "\n".join(lines)

    @staticmethod
    def cook_lines(visit: Visit) -> List[str]:
        lines = [f"Name: {visit.cook_name or 'not known yet'}.",
                 f"Kitchen experience: {visit.skill_level or 'not known yet'}.",
                 f"Allergies or foods they avoid: {visit.dietary or 'not known yet'}."]
        lines.append(f"{visit.minutes_so_far} minutes into the session, "
                     f"{visit.turn_count} message{'s' if visit.turn_count != 1 else ''} so far.")
        return lines

    @staticmethod
    def recipe_lines(visit: Visit, recipe: Optional[ActiveRecipe]) -> List[str]:
        if recipe is None:
            return ["No recipe chosen yet."]
        steps, total = recipe.steps, len(recipe.steps)
        origin = "uploaded by the cook" if recipe.custom else "from the catalog"
        about = ", ".join(filter(None, [f"serves {recipe.serves}" if recipe.serves else "",
                                        f"about {recipe.time_minutes} min" if recipe.time_minutes else ""]))
        lines = [f"Cooking: {recipe.name} ({origin}{', ' + about if about else ''}, {total} steps)."]
        if recipe.ingredients:
            lines.append("Ingredients:")
            lines += [f"- {item}" for item in recipe.ingredients]
        lines.append("Steps:")
        for i, step in enumerate(steps):
            marker = "  <- CURRENT STEP" if visit.step == i else ""
            lines.append(f"{i + 1}. {step['text']}{marker}")
        if visit.step is None:
            lines.append("Status: not started yet (they are looking at the ingredients).")
        elif visit.step >= total:
            lines.append(f"Status: all {total} steps are finished.")
        else:
            step = steps[visit.step]
            lines.append(f"Current step {visit.step + 1} of {total}: {step['text']}")
            if step.get("done_when"):
                lines.append(f"Done when: {step['done_when']}")
            if step.get("timer_seconds"):
                lines.append(f"Timer for this step: about {spoken_duration(step['timer_seconds'])}.")
        return lines

    def catalog_lines(self) -> List[str]:
        lines = ["Recipes you can load with choose_recipe (suggest at most 2 at a time; "
                 "never list them all):"]
        for r in self.knowledge.recipes():
            key = ", ".join(dict.fromkeys(ingredient_core(i) for i in r["ingredients"][:6]))
            facts = [f"{r['time_minutes']} min" if r.get("time_minutes") else "",
                     r.get("difficulty") or ""]
            lines.append(f"- {r['name']} ({', '.join(filter(None, facts))}): {key}")
        return lines

    def _now_lines(self, turn: TurnContext) -> List[str]:
        lines: List[str] = []
        nav = turn.nav
        if nav:
            if turn.step is not None and turn.step >= len(turn.steps):
                lines.append("They finished the last step. Congratulate them briefly.")
            else:
                verb = {"next": "move to the next step", "previous": "go back a step",
                        "repeat": "hear the step again", "start": "start the recipe",
                        "ready": "start the recipe",
                        "goto": f"jump to step {nav[1] if len(nav) > 1 else ''}"}[nav[0]]
                lines.append(f"They asked to {verb}; the pointer has already moved. Give "
                             f"the current step from [RECIPE] with its quantities, then its "
                             f"'done when' cue or timer. Do not call go_to_step.")
        elif turn.subject is not None and turn.subject_source != "recipe":
            lines.append(f"They seem to be asking about {turn.subject.name}.")
        if turn.facet and not nav:
            lines.append(f"Question type: {turn.facet}.")
        if turn.frames:
            lines.append(f"{turn.frames} live camera frame{'s are' if turn.frames > 1 else ' is'} "
                         f"attached (oldest first, about a second apart). Use them if the "
                         f"question is about what they are doing or what the food looks like.")
        if turn.observation is not None:
            lines.extend(turn.observation.prompt_lines())
        if not lines:
            lines.append("A general message.")
        return lines

    # -- after the answer -------------------------------------------------
    def record(self, turn: TurnContext, response: str, intent: str = None,
               source: str = None, user_text: str = None, vision: str = None):
        """Persist the turn and mark its facts as delivered."""
        if vision is None and turn.observation is not None:
            vision = turn.observation.summary
        if vision is None and turn.frames:
            vision = f"Looked at {turn.frames} camera frame{'s' if turn.frames > 1 else ''}."
        self.store.record_turn(
            visit=turn.visit,
            episode=turn.episode,
            user_text=user_text if user_text is not None else turn.message,
            assistant_text=response,
            language=turn.language,
            intent=intent or (turn.nav[0] if turn.nav else None),
            facet=turn.facet,
            fact_ids=turn.fact_ids,
            source=source,
            vision=vision,
        )


def cooking_state(recipe: Optional[ActiveRecipe], step: Optional[int]) -> Dict[str, Any]:
    """The recipe/step payload the phone renders, shared by every reply path."""
    steps = recipe.steps if recipe else []
    current = steps[step] if step is not None and 0 <= step < len(steps) else {}
    return {
        "recipe": recipe.name if recipe else None,
        "custom": bool(recipe and recipe.custom),
        "step": step if recipe else None,
        "total_steps": len(steps),
        "steps": [s["text"] for s in steps],
        "step_text": current.get("text"),
        "done_when": current.get("done_when"),
        "timer_seconds": current.get("timer_seconds"),
        "ingredients": recipe.ingredients if recipe else [],
        "serves": recipe.serves if recipe else None,
        "time_minutes": recipe.time_minutes if recipe else None,
    }


def spoken_duration(seconds: int) -> str:
    """90 -> '1 and a half minutes'; for reading a timer aloud."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} seconds"
    minutes, rest = divmod(seconds, 60)
    if rest >= 30:
        return f"{minutes} and a half minutes"
    return f"{minutes} minute{'s' if minutes != 1 else ''}"
