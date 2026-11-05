# context_builder.py - Turns a cook's utterance into a layered prompt context
"""
The join between what the coach knows (knowledge_base), what it remembers
(conversation_state) and what the camera sees (vision).

The context is built in labelled layers, widest first:

    [COOK]      session profile: name, skill, dietary needs
    [RECIPE]    the recipe being cooked and the step they are on
    [NOW]       what this question is about, and what the camera sees
    [RECENT]    the last few exchanges, verbatim
    [FACTS]     retrieved facts, each tagged with the facet it answers
    [QUESTION]  the utterance itself

STEP NAVIGATION
---------------
"next", "go back", "say that again", "step 4", "let's start" move the step
pointer before anything is rendered, so the [RECIPE] layer already shows the
step the cook asked for. Navigation is decided by pattern, not by the model:
it has to be instant and it has to be right.

SUBJECT PRIORITY
----------------
What "it" refers to, in order:

    1. something named outright ("how do I emulsify?")
    2. a pronoun ("how long does IT rest?") -> the episode's subject
    3. a pointing word ("is THIS done?") -> what the camera recognised
    4. a follow-up ("anything else?") -> the episode's subject
    5. the recipe being cooked
    6. nothing - a general kitchen question

GROUNDING
---------
Recipe quantities, times and temperatures come only from [FACTS] and
[RECIPE]. When retrieval comes back empty the context says so, and the coach
may offer general cooking advice but must not invent recipe numbers.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

from knowledge_base import KnowledgeBase, Subject, classify_facet
from conversation_state import ConversationStore, Episode, Visit
from web_search import explicit_search

logger = logging.getLogger(__name__)

MAX_FACTS = 4
MAX_RECENT_TURNS = 3

# Pronouns point back into the CONVERSATION: "how long does IT rest?" means
# whatever we were just discussing.
ANAPHORIC_MARKERS = ("it", "its", "it's", "them", "they", "their")

# Pointing words point out into the WORLD: "is THIS done?" means the thing in
# front of the camera.
DEMONSTRATIVE_MARKERS = ("this", "these", "this one", "that one", "those")

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
    ("stop", r"\b(stop cooking|cancel (the )?recipe|forget (the )?recipe|"
             r"change (the )?(recipe|dish)|different recipe|something else to cook|"
             r"switch (the )?(dish|recipe)|(make|cook) something else)\b"),
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


def _clean(text: str) -> str:
    return re.sub(r"[^\w\s']", " ", (text or "").lower()).strip()


def detect_navigation(text: str) -> Optional[tuple]:
    """("next",) / ("previous",) / ("repeat",) / ("start",) / ("stop",) /
    ("goto", n) for a navigation request, else None.

    A doneness question ("is it done?") is not navigation, even though it
    contains "done".
    """
    if classify_facet(text) == "doneness":
        return None
    if explicit_search(text):
        return None
    if re.search(r"\bnext time\b", (text or "").lower()):
        return None
    cleaned = _clean(text)
    if cleaned in ("start", "start please"):
        return ("start",)
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


# Phrases that mean "I want to cook this one now", as opposed to asking about
# a recipe while cooking another.
SWITCH_PATTERN = re.compile(
    r"\b(let'?s (make|cook|do)|i'?m (making|cooking)|"
    r"i want to (make|cook|change)|"
    r"switch to|change to|instead|"
    r"change (the )?(dish|recipe)|switch (the )?(dish|recipe)|"
    r"(make|cook) something else|start)\b")

# Follow-ups that pick a dish the coach just named ("the one you mentioned").
RECIPE_ANAPHORA_PICK = re.compile(
    r"\b(the one (you|u) mentioned|that one|the .+ one)\b")

# Unlock the session with no replacement dish. Deliberately excludes bare
# "start" so "let's start" still only rewinds the current recipe's steps.
CLEAR_RECIPE_PATTERN = re.compile(
    r"\b(i want to change|change (the )?(dish|recipe)|switch (the )?(dish|recipe)|"
    r"(make|cook) something else|different recipe|something else to cook|"
    r"stop cooking|cancel (the )?recipe|forget (the )?recipe)\b")


def wants_recipe_switch(text: str) -> bool:
    """True when the cook is changing or confirming a dish, not just asking about one."""
    cleaned = _clean(text)
    return bool(SWITCH_PATTERN.search(cleaned) or RECIPE_ANAPHORA_PICK.search(cleaned))


def wants_recipe_clear(text: str) -> bool:
    """True when the cook wants to drop the current dish without naming a new one."""
    return bool(CLEAR_RECIPE_PATTERN.search(_clean(text)))


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


def is_demonstrative(text: str) -> bool:
    return _has_marker(text, DEMONSTRATIVE_MARKERS)


def is_follow_up(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in FOLLOW_UP_MARKERS)


# --- Language ----------------------------------------------------------------
# English only for now. The lever is kept so a second language is a config
# change plus content, not a code change.
DEFAULT_LANGUAGE = os.getenv("GUIDE_DEFAULT_LANGUAGE", "en").strip().lower()


def detect_language(text: str) -> str:
    return DEFAULT_LANGUAGE


class TurnContext:
    """Everything decided about one turn, plus the rendered prompt context."""

    __slots__ = ("visit", "episode", "subject", "subject_source", "facet", "language",
                 "facts", "text", "grounded", "message", "recipe", "steps", "nav",
                 "observation", "social")

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
    def current_step(self) -> Optional[Dict[str, Any]]:
        if not self.steps or self.step is None or self.step >= len(self.steps):
            return None
        return self.steps[self.step]

    def cooking_state(self) -> Dict[str, Any]:
        """What the phone shows on its step card."""
        return cooking_state(self.recipe.name if self.recipe else None,
                             self.step, self.steps or [])

    def __repr__(self):
        return (f"TurnContext(subject={self.subject_name!r} via {self.subject_source},"
                f" facet={self.facet}, nav={self.nav}, step={self.step},"
                f" facts={len(self.facts or [])},"
                f" camera={'yes' if self.observation else 'no'})")


class ContextBuilder:
    """Resolves a turn against the knowledge base and the conversation so far."""

    def __init__(self, knowledge: KnowledgeBase, store: ConversationStore):
        self.knowledge = knowledge
        self.store = store

    # -- recipe and step ---------------------------------------------------
    def active_recipe(self, visit: Visit) -> Optional[Subject]:
        if visit.recipe_id is None:
            return None
        return self.knowledge.subject_by_id(visit.recipe_id, source="recipe")

    def _recipe_from_recent(self, visit: Visit) -> Optional[Subject]:
        """Find a recipe named in recent chat that is not the one currently cooking."""
        current_id = visit.recipe_id
        for turn in reversed(self.store.recent_turns(visit.session_id, limit=6)):
            for text in (turn.get("assistant_text"), turn.get("user_text")):
                found = self.knowledge.resolve_recipe(text or "")
                if found is not None and found.id != current_id:
                    return found
        return None

    def _maybe_switch_recipe(self, visit: Visit, message: str,
                             recipe: Optional[Subject]) -> Optional[Subject]:
        named = self.knowledge.resolve_recipe(message)
        switch = wants_recipe_switch(message)
        # "the one you mentioned" / switch with no name: recover from recent chat.
        if named is None and switch and (
                detect_navigation(message) != ("start",) or RECIPE_ANAPHORA_PICK.search(_clean(message))):
            named = self._recipe_from_recent(visit)
        if named is None:
            # Explicit change/cancel with no target unlocks the session.
            # Do not clear on bare "start" or a failed "that one" anaphora.
            if recipe is not None and wants_recipe_clear(message):
                self.store.set_recipe(visit, None, None)
                return None
            return recipe
        if named == recipe:
            return recipe
        # With nothing on the go, naming a dish starts it. Mid-recipe, only an
        # explicit switch -- otherwise "is aglio olio spicy?" mid-cook would
        # throw away the cook's place.
        if recipe is None or switch:
            self.store.set_recipe(visit, named.id, named.name)
            return self.knowledge.subject_by_id(named.id, source="recipe")
        return recipe

    def _navigate(self, visit: Visit, nav: tuple, steps: List[Dict[str, Any]]):
        intent = nav[0]
        total = len(steps)
        current = visit.step
        if intent == "start":
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
            return
        if target != current:
            self.store.set_step(visit, target)

    # -- subject ----------------------------------------------------------
    def resolve_subject(self, message: str, episode: Optional[Episode],
                        recipe: Optional[Subject], observation=None):
        """Work out what this turn is about. See the module docstring for order."""
        named = self.knowledge.resolve_subject(message)
        if named is not None:
            return named, "named"

        carried = self._episode_subject(episode)
        if is_anaphoric(message) and carried is not None:
            return carried, "carried"

        if is_demonstrative(message) and observation is not None:
            for item in observation.items:
                seen = self.knowledge.subject_for_label(item)
                if seen is not None:
                    return seen, "camera"

        if is_follow_up(message) and carried is not None:
            return carried, "carried"

        if recipe is not None:
            return recipe, "recipe"

        if carried is not None and classify_facet(message) is not None:
            return carried, "carried"

        return None, "none"

    def _episode_subject(self, episode: Optional[Episode]) -> Optional[Subject]:
        if episode is None or episode.subject_id is None:
            return None
        return Subject(episode.subject_type, episode.subject_id,
                       episode.subject_name, source="episode")

    # -- build ------------------------------------------------------------
    def build(self, user_id: str, message: str, observation=None) -> TurnContext:
        visit = self.store.current_visit(user_id)
        language = visit.language or detect_language(message)

        previous_recipe_id = visit.recipe_id
        recipe = self._maybe_switch_recipe(visit, message, self.active_recipe(visit))

        nav = detect_navigation(message)
        # Choosing a dish always previews ingredients, even for "start curry".
        if recipe and recipe.id != previous_recipe_id:
            nav = None
        if nav and nav[0] == "stop":
            self.store.set_recipe(visit, None, None)
            recipe = None
        steps = self.knowledge.steps(recipe.id) if recipe else []
        if nav and recipe and steps:
            self._navigate(visit, nav, steps)

        active = self.store.active_episode(visit.session_id)
        subject, subject_source = self.resolve_subject(message, active, recipe, observation)
        facet = classify_facet(message)
        social = not nav and subject_source in ("none", "recipe") and is_social(message)

        episode = None
        if not social and not nav and subject is not None:
            episode = self.store.open_episode_for(
                visit.session_id, user_id, subject, opened_by=subject_source)

        # Navigation reads a step; the step is the content, so no facts.
        facts: List[Dict[str, Any]] = []
        if not social and not nav:
            delivered = self.store.delivered_fact_ids(visit.session_id)
            facts = self.knowledge.retrieve(subject, facet, query=message,
                                            exclude_ids=delivered, limit=MAX_FACTS,
                                            recipe=recipe)
            # Everything relevant has been said. Rather than answer with
            # nothing, let the ledger go and repeat the best material.
            if not facts and subject is not None and delivered:
                self.store.forget_delivered(visit.session_id)
                facts = self.knowledge.retrieve(subject, facet, query=message,
                                                limit=MAX_FACTS, recipe=recipe)

        turn = TurnContext(
            visit=visit, episode=episode, subject=subject, subject_source=subject_source,
            facet=facet, language=language, facts=facts, message=message,
            recipe=recipe, steps=steps, nav=nav, observation=observation, social=social,
            grounded=bool(social or nav or facts),
        )
        turn.text = self._render(turn)
        return turn

    # -- render -----------------------------------------------------------
    def _render(self, turn: TurnContext) -> str:
        visit = turn.visit
        lines: List[str] = []

        # --- COOK --------------------------------------------------------
        lines.append("[COOK]")
        if visit.cook_name:
            lines.append(f"Name: {visit.cook_name}.")
        if visit.skill_level:
            lines.append(f"Kitchen experience: {visit.skill_level}. Pitch explanations to it.")
        if visit.dietary:
            lines.append(f"Allergies or foods they avoid: {visit.dietary}. If anything "
                         f"in the recipe conflicts, say so.")
        lines.append(f"{visit.minutes_so_far} minutes into the session, "
                     f"{visit.turn_count} question{'s' if visit.turn_count != 1 else ''} so far.")
        lines.append("")

        # --- RECIPE ------------------------------------------------------
        lines.append("[RECIPE]")
        lines.extend(self._recipe_lines(turn))
        lines.append("")

        # --- NOW ---------------------------------------------------------
        lines.append("[NOW]")
        lines.extend(self._now_lines(turn))
        lines.append("")

        # --- RECENT ------------------------------------------------------
        recent = self.store.recent_turns(visit.session_id, MAX_RECENT_TURNS)
        if recent:
            lines.append("[RECENT]")
            for past in recent:
                if past.get("user_text"):
                    lines.append(f"Cook: {past['user_text']}")
                if past.get("assistant_text"):
                    reply = past["assistant_text"]
                    if len(reply) > 200:
                        reply = reply[:200].rstrip() + "..."
                    lines.append(f"Coach: {reply}")
            lines.append("")

        # --- FACTS -------------------------------------------------------
        lines.append("[FACTS]")
        if turn.social:
            lines.append("This is small talk, not a question. Reply warmly in one short "
                         "sentence. No facts needed.")
        elif turn.nav:
            lines.append("Navigation request: the step in [RECIPE] is the content. "
                         "No other facts needed.")
        elif turn.facts:
            lines.append("Use these for anything specific to this recipe. Do not invent "
                         "quantities, times or temperatures.")
            for fact in turn.facts:
                lines.append(f"- ({fact['facet']}) {fact['subject_name']}: {fact['text']}")
        else:
            # The grounding guard. Left implicit, the model invents recipe
            # numbers with total confidence.
            lines.append("The recipe notes do not cover this question.")
            lines.append("You may give brief general cooking advice and say that it is "
                         "general advice. Do NOT invent quantities, times or "
                         "temperatures for this recipe. On food safety, if unsure, "
                         "recommend the cautious option.")
        lines.append("")

        lines.append("[QUESTION]")
        lines.append(turn.message)
        return "\n".join(lines)

    def _recipe_lines(self, turn: TurnContext) -> List[str]:
        visit, recipe, steps = turn.visit, turn.recipe, turn.steps or []
        if recipe is None:
            lines = ["No recipe chosen yet."]
            if visit.dish:
                lines.append(f"They said they are cooking: {visit.dish}. There are no "
                             f"recipe notes for that.")
            available = [r["name"] for r in self.knowledge.recipes()]
            if available:
                lines.append(f"Recipes you can walk them through: {', '.join(available)}.")
            return lines

        total = len(steps)
        lines = [f"Cooking: {recipe.name} ({total} steps)."]
        if visit.step is None:
            lines.append("Not started yet. When they are ready, they say 'start'.")
            ingredients = self.knowledge.ingredients_text(recipe.id)
            if ingredients:
                lines.append(ingredients)
        elif visit.step >= total:
            lines.append(f"All {total} steps are finished.")
        else:
            step = steps[visit.step]
            lines.append(f"Current step {visit.step + 1} of {total}: {step['text']}")
            if step.get("done_when"):
                lines.append(f"Done when: {step['done_when']}")
            if step.get("timer_seconds"):
                lines.append(f"Timer for this step: about "
                             f"{spoken_duration(step['timer_seconds'])}.")
            if visit.step + 1 < total:
                lines.append(f"Next step after this: {steps[visit.step + 1]['text']}")
        return lines

    def _now_lines(self, turn: TurnContext) -> List[str]:
        lines: List[str] = []
        nav = turn.nav
        if nav:
            if nav[0] == "stop":
                lines.append("They stopped the recipe. Acknowledge briefly.")
            elif turn.recipe is None:
                lines.append("They want to move through recipe steps, but no recipe is "
                             "chosen. Ask which dish they are making.")
            elif turn.step is not None and turn.step >= len(turn.steps):
                lines.append("They finished the last step. Congratulate them briefly.")
            else:
                verb = {"next": "move to the next step", "previous": "go back a step",
                        "repeat": "hear the step again", "start": "start the recipe",
                        "goto": f"jump to step {nav[1] if len(nav) > 1 else ''}"}[nav[0]]
                lines.append(f"They asked to {verb}. Read the current step from [RECIPE] "
                             f"clearly and completely, quantities included.")
        elif turn.subject is not None:
            how = {
                "camera": "the camera sees it",
                "named": "they named it",
                "carried": "carried over from the last question",
                "recipe": "the recipe being cooked",
            }.get(turn.subject_source, "")
            lines.append(f"They are asking about {turn.subject.name}"
                         + (f" ({how})." if how else "."))
            if turn.episode is not None:
                covered = turn.episode.summary()
                if covered:
                    lines.append(covered)
        else:
            lines.append("A general kitchen question, not tied to a recipe.")
        if turn.facet and not nav:
            lines.append(f"They want to know about: {turn.facet}.")

        if turn.observation is not None:
            lines.extend(turn.observation.prompt_lines())
            lines.append("Use what the camera sees to answer, but describe it rather "
                         "than overclaim: a photo cannot show temperature or taste.")
        return lines

    # -- after the answer -------------------------------------------------
    def record(self, turn: TurnContext, response: str, intent: str = None,
               source: str = None):
        """Persist the turn and mark its facts as delivered."""
        self.store.record_turn(
            visit=turn.visit,
            episode=turn.episode,
            user_text=turn.message,
            assistant_text=response,
            language=turn.language,
            intent=intent or (turn.nav[0] if turn.nav else None),
            facet=turn.facet,
            fact_ids=turn.fact_ids,
            source=source,
            vision=turn.observation.summary if turn.observation else None,
        )


def cooking_state(recipe_name: Optional[str], step: Optional[int],
                  steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The recipe/step payload the phone renders, shared by every reply path."""
    current = steps[step] if step is not None and 0 <= step < len(steps) else {}
    return {
        "recipe": recipe_name,
        "step": step,
        "total_steps": len(steps),
        "step_text": current.get("text"),
        "done_when": current.get("done_when"),
        "timer_seconds": current.get("timer_seconds"),
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
