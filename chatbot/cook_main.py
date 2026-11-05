# cook_main.py - Orchestrator for the voice cooking coach
"""
Ties the pieces together for one turn of conversation:

    knowledge_base      what the coach knows: recipes, steps, glossary facts
    conversation_state  visit -> episode -> turn, the recipe/step pointer,
                        and the delivered-fact ledger
    vision              what the phone camera sees, via a pluggable VLM
    context_builder     resolves the turn and renders the layered prompt
    streaming_openai    the personas and the model call

One turn:

    onboarding answer?  -> store it, ask the next question (no model call)
    needs the camera?   -> VLM observation from the attached frame
    build context       -> recipe step, subject, facts, observation
    pick a persona      -> step | safety | help
    model call          -> reply
    record              -> turn, episode facets, delivered facts
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from context_builder import ContextBuilder, cooking_state, detect_navigation
from conversation_state import ConversationStore
from knowledge_base import KnowledgeBase, classify_facet
from vision import VisionProvider, VisionRequest, load_vision_provider, wants_vision

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CookingAssistant:
    """Local recipe knowledge + an optional chat model + an optional VLM."""

    def __init__(self, vision: VisionProvider = None):
        logger.info("Initializing cooking coach...")

        self.knowledge = KnowledgeBase()
        self.store = ConversationStore()
        self.context = ContextBuilder(self.knowledge, self.store)
        self.vision = vision or load_vision_provider()

        from enhanced_rag_openai import EnhancedRAGWithOpenAI
        self.model = EnhancedRAGWithOpenAI()

        logger.info("Cooking coach ready (%d facts, %d recipes)",
                    self.knowledge.fact_count(), len(self.knowledge.recipes()))

    # -- session ------------------------------------------------------------
    def start_visit(self, user_id: str) -> str:
        """The first thing to say when the phone connects."""
        question = self.store.start_visit(user_id)
        if question:
            return question
        visit = self.store.current_visit(user_id)
        name = f", {visit.cook_name}" if visit.cook_name else ""
        if visit.recipe_id is None:
            return f"Welcome back{name}! What are we cooking?"
        total = len(self.knowledge.steps(visit.recipe_id))
        if visit.step is None:
            return f"Welcome back{name}! {self._recipe_preview(visit.recipe_id, visit.recipe_name)}"
        if visit.step >= total:
            return f"Welcome back{name}! You finished {visit.recipe_name}."
        return (f"Welcome back{name}! You're on step {visit.step + 1} of {total} "
                f"of {visit.recipe_name}. Say repeat to hear it again.")

    def onboarding_state(self, user_id: str) -> Dict[str, Any]:
        return self.store.onboarding_state(user_id)

    def cooking_state(self, user_id: str) -> Dict[str, Any]:
        visit = self.store.current_visit(user_id)
        steps = self.knowledge.steps(visit.recipe_id) if visit.recipe_id else []
        state = cooking_state(visit.recipe_name, visit.step, steps)
        recipe = self.knowledge.recipe_details(visit.recipe_id) if visit.recipe_id else {}
        state.update(ingredients=recipe.get("ingredients", []), serves=recipe.get("serves"),
                     time_minutes=recipe.get("time_minutes"))
        return state

    def _recipe_preview(self, recipe_id: int, recipe_name: str) -> str:
        ingredients = self.knowledge.ingredients_text(recipe_id) or ""
        return (f"Let's make {recipe_name}.\n\n{ingredients}\n\n"
                "Gather these ingredients, then say start or tap Start cooking for step 1.")

    def _finish_onboarding(self, user_id: str) -> str:
        """Resolve 'what are you cooking today?' to a recipe, and greet."""
        visit = self.store.current_visit(user_id)
        name = f", {visit.cook_name}" if visit.cook_name else ""
        recipes = self.knowledge.recipes()
        available = ", ".join(r["name"] for r in recipes)
        recipe = self.knowledge.resolve_recipe(visit.dish) if visit.dish else None

        if recipe is not None:
            self.store.set_recipe(visit, recipe.id, recipe.name)
            return f"Great{name}! {self._recipe_preview(recipe.id, recipe.name)}"
        if visit.dish:
            return (f"Thanks{name}! I don't have a recipe for {visit.dish} yet, but I can "
                    f"still answer cooking questions. Recipes I can walk you through: "
                    f"{available}.")
        return (f"Thanks{name}! Recipes I can walk you through: {available}. "
                f"Which one would you like?")

    # -- one turn -----------------------------------------------------------
    async def process_message(self, message: str, user_id: str = "default_user",
                              image: Optional[tuple] = None, on_delta=None,
                              image_source: str = "camera") -> Dict[str, Any]:
        """Answer one utterance. `image` is (bytes, mime) from the phone, or None.

        Returns the reply plus the state the phone needs to update its screen.
        """
        try:
            onboarding = await asyncio.to_thread(
                self.store.submit_onboarding_answer, user_id, message)
            if onboarding["just_completed"]:
                reply = await asyncio.to_thread(self._finish_onboarding, user_id)
                return self._result(user_id, reply)
            if onboarding["message"] is not None:
                return self._result(user_id, onboarding["message"])

            uploaded_photo = image_source == "upload" and image is not None
            observation = await self._look(user_id, message, image, force=uploaded_photo)
            if uploaded_photo and observation is None:
                return self._result(user_id, "I couldn't analyze the uploaded photo. Check your vision model settings and try again.")

            previous_recipe_id = self.store.current_visit(user_id).recipe_id
            turn = await asyncio.to_thread(self.context.build, user_id, message, observation)
            logger.info("Resolved %r", turn)

            if turn.recipe and turn.recipe.id != previous_recipe_id and not uploaded_photo:
                reply = self._recipe_preview(turn.recipe.id, turn.recipe.name)
                await asyncio.to_thread(self.context.record, turn, reply, source="recipe-notes")
                return self._result(user_id, reply, observation=observation)

            context = {
                "prompt_context": turn.text,
                "language": turn.language,
                "query_type": self._persona(turn),
                "social": turn.social,
                "facet": turn.facet,
                "missing_notes": not turn.facts and not turn.observation,
            }
            reply = await self.model.process_query_with_openai(
                query=message, context=context, user_id=user_id, on_delta=on_delta)

            await asyncio.to_thread(self.context.record, turn, reply, source="model")
            return self._result(user_id, reply, cooking=turn.cooking_state(),
                                observation=observation, sources=context.get("sources"))

        except Exception as exc:
            logger.error("Processing error: %s", exc, exc_info=True)
            return self._result(user_id, "Sorry, something went wrong on my side. "
                                         "Could you say that again?")

    async def _look(self, user_id: str, message: str, image: Optional[tuple], force=False):
        """Ask the VLM about the attached frame, if this question needs eyes."""
        if image is None or not self.vision.available:
            return None
        if not force and (detect_navigation(message) or not wants_vision(
                message, classify_facet(message), self.vision.config.mode)):
            return None

        visit = self.store.current_visit(user_id)
        request = VisionRequest(question=message, recipe=visit.recipe_name)
        if visit.recipe_id is not None and visit.step is not None:
            steps = self.knowledge.steps(visit.recipe_id)
            if visit.step < len(steps):
                request.step_text = steps[visit.step]["text"]
                request.done_when = steps[visit.step].get("done_when")
        raw, mime = image
        return await self.vision.observe(raw, mime, request)

    @staticmethod
    def _persona(turn) -> str:
        if turn.nav:
            return "step"
        if turn.facet == "safety" or (turn.observation and turn.observation.hazards):
            return "safety"
        return "help"

    def _result(self, user_id, reply, cooking=None, observation=None, sources=None) -> Dict[str, Any]:
        return {
            "message": reply,
            "profile": self.store.onboarding_state(user_id),
            "cooking": {**self.cooking_state(user_id), **(cooking or {})},
            "vision": observation.to_dict() if observation else None,
            "sources": sources or [],
        }

    # -- reporting -----------------------------------------------------------
    def get_user_insights(self, user_id):
        return self.store.visit_summary(user_id)


async def test_cooking_session():
    """Walk a cook through onboarding and part of a recipe, printing each reply."""
    print("Testing the cooking coach")
    print("=" * 60)

    assistant = CookingAssistant()
    user_id = "test_cook"
    print(assistant.start_visit(user_id))
    for answer in ("Sam", "beginner", "none", "garlic pasta"):
        result = await assistant.process_message(answer, user_id)
        print(f"   profile: {answer!r} -> {result['message']}")

    script = [
        "What ingredients do I need?",
        "Let's start",
        "How salty should the water be?",
        "Next",
        "What does al dente mean?",
        "Next",
        "Can I use linguine instead?",
        "Repeat that",
        "The pan is smoking a lot, what do I do?",
        "Thanks!",
    ]
    for i, query in enumerate(script, 1):
        result = await assistant.process_message(query, user_id)
        cooking = result["cooking"]
        where = (f"step {cooking['step'] + 1}/{cooking['total_steps']}"
                 if cooking["step"] is not None else "not started")
        print(f"\n{i}. {query!r}   [{where}]")
        print(f"   -> {result['message']}")

    print("\n" + "=" * 60)
    import json
    print(json.dumps(assistant.get_user_insights(user_id), indent=2))


if __name__ == "__main__":
    asyncio.run(test_cooking_session())
