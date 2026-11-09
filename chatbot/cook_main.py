# cook_main.py - Orchestrator for the cooking coach
"""
Ties the pieces together. The chat model writes every reply; this module
gathers its context, gives it tools, and keeps the session state.

    knowledge_base      catalog recipes, steps and glossary facts
    conversation_state  visit -> episode -> turn, recipe/step pointer, profile,
                        uploaded recipes
    context_builder     resolves the turn and renders the layered prompt
    streaming_openai    system prompt, tool loop, streaming
    vision              camera frames: inline to the chat model, or a separate VLM
    recipe_import       uploaded recipe -> structured steps

One turn:

    short step command? -> move the pointer (instant, no tool round)
    camera on?          -> attach the latest frames (or a VLM observation)
    build context       -> profile, recipe + all steps, catalog, facts
    model call          -> streamed reply; tools may choose a recipe, move the
                           step, stop, or update the profile
    record              -> turn, episode facets, delivered facts

Other entry points: greet() for a new chat, import_recipe() for an upload,
open_saved_recipe() to cook an upload again, finish() for "how did it go?",
watch() for the periodic camera check while a step is in progress.

Tools can also hand the phone something to show: suggest_recipes adds idea
cards and start_timer starts a named timer. They come back in the reply as
``suggestions`` and ``timers``.
"""

import asyncio
import difflib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from context_builder import ContextBuilder, cooking_state, detect_navigation
from conversation_state import MAX_SERVINGS, RATINGS, ConversationStore
from knowledge_base import KnowledgeBase, classify_facet
from recipe_import import RecipeImportError, fetch_recipe_page, parse_recipe
from vision import VisionProvider, VisionRequest, load_vision_provider, to_data_url, wants_vision

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HISTORY_TURNS = 6
WATCH_INTERVAL_SECONDS = int(os.getenv("WATCH_INTERVAL_SECONDS", "20"))

GREETING_FALLBACK = ("Hi! I'm your cooking coach. Tell me what you'd like to cook, "
                     "ask me for ideas, or upload your own recipe.")

WATCH_PROMPT = """You are watching a home cook through their phone camera while \
they follow a recipe. You get the step they are on and the latest camera frames \
(oldest first). Decide whether to speak up.

Return JSON only:
{"status": "ok" | "unclear" | "issue" | "hazard" | "step_done",
 "message": "<what to say to the cook, or empty>",
 "seen": "<one short sentence describing what is visible>"}

- "hazard": something unsafe is visible (smoke, flames, boiling over, a knife near
  the counter edge, raw meat touching ready-to-eat food, fingers near a blade).
  message: one short, calm, urgent instruction.
- "issue": the cook is clearly doing the current step wrong, or the food is going
  wrong (burning, wrong ingredient, heat clearly too high, pieces far too big).
  message: one or two short sentences: what is wrong and what to do.
- "step_done": the frames clearly show the step's "done when" result.
  message: one short sentence saying it looks done and they can say next.
- "ok": normal progress. "unclear": food not visible, dark or blurry frames.
  message empty for both.
Only speak up when you are confident; when in doubt use "ok" or "unclear".
Do not repeat the previous alert unless the problem is clearly still there.
A picture cannot show temperature or taste; never claim it does."""

WATCH_STATUSES = ("ok", "unclear", "issue", "hazard", "step_done")


class CookingAssistant:
    """Local recipe knowledge + the chat model + camera frames."""

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
    def cooking_state(self, user_id: str) -> Dict[str, Any]:
        visit = self.store.current_visit(user_id)
        return cooking_state(self.context.active_recipe(visit), visit.step)

    @property
    def watch_available(self) -> bool:
        return bool(self.vision.available and self.vision.inline and self.model.available)

    async def greet(self, user_id: str) -> Optional[str]:
        """The opening message of a new chat, or None if this chat has started."""
        visit = self.store.current_visit(user_id)
        if visit.turn_count:
            return None
        recipe = self.context.active_recipe(visit)
        reply = None
        if self.model.available:
            from streaming_openai import LANGUAGE_INSTRUCTIONS
            prompt = "\n".join([
                LANGUAGE_INSTRUCTIONS.get(visit.language, LANGUAGE_INSTRUCTIONS["en"]), "",
                "[COOK]", *self.context.cook_lines(visit), "",
                "[RECIPE]", *self.context.recipe_lines(visit, recipe), "",
                "The cook has just opened the app. Greet them warmly in one or two short "
                "sentences (by name if you know it). Invite them to tell you what they'd "
                "like to cook, ask you for ideas, or upload their own recipe. If a recipe "
                "is in progress, say where they are instead. Do not list recipes."])
            reply = await self.model.complete([
                {"role": "system", "content": self.model.streaming_openai.system_prompt},
                {"role": "user", "content": prompt}], budget=600)
        reply = reply or GREETING_FALLBACK
        await asyncio.to_thread(self.store.record_turn, visit, None, None, reply,
                                source="greeting" if self.model.available else "fallback")
        return reply

    # -- tools the model can call ---------------------------------------------
    def run_tool(self, user_id: str, name: str, args: Dict[str, Any],
                 actions: Optional[Dict[str, list]] = None) -> Dict[str, Any]:
        """Run one tool. ``actions`` collects cards and timers for the phone."""
        actions = actions if actions is not None else {"suggestions": [], "timers": []}
        visit = self.store.current_visit(user_id)
        if name == "suggest_recipes":
            catalog = {r["id"]: r for r in self.knowledge.recipes()}
            cards, unknown = [], []
            for item in (args.get("recipes") or [])[:2]:
                item = item if isinstance(item, dict) else {"recipe_name": str(item)}
                subject = self.knowledge.resolve_recipe(str(item.get("recipe_name") or ""))
                if subject is None or any(c["name"] == subject.name for c in cards):
                    unknown.append(item.get("recipe_name"))
                    continue
                info = catalog.get(subject.id, {})
                cards.append({"name": subject.name, "why": " ".join(str(item.get("why") or "").split())[:120],
                              "time_minutes": info.get("time_minutes"), "difficulty": info.get("difficulty")})
            actions["suggestions"] = cards
            self.store.set_suggestions(visit, [c["name"] for c in cards])
            return {"ok": bool(cards), "shown": [c["name"] for c in cards],
                    **({"not_in_catalog": unknown} if unknown else {})}
        if name == "start_timer":
            label = " ".join(str(args.get("label") or "Timer").split())[:40] or "Timer"
            try:
                seconds = int(args.get("seconds"))
            except (TypeError, ValueError):
                return {"ok": False, "error": "seconds must be a whole number."}
            if not 5 <= seconds <= 6 * 60 * 60:
                return {"ok": False, "error": "Timers run from 5 seconds to 6 hours."}
            actions["timers"].append({"label": label, "seconds": seconds})
            return {"ok": True, "label": label, "seconds": seconds}
        if name == "scale_recipe":
            recipe = self.context.active_recipe(visit)
            if recipe is None:
                return {"ok": False, "error": "No recipe is active."}
            if not recipe.base_serves:
                return {"ok": False, "error": "This recipe doesn't say how many it serves, so it "
                                              "can't be scaled. Scale amounts in your reply instead."}
            try:
                servings = int(args.get("servings"))
                self.store.set_servings(visit, None if servings == recipe.base_serves else servings)
            except (TypeError, ValueError):
                return {"ok": False, "error": f"servings must be 1 to {MAX_SERVINGS}."}
            recipe = self.context.active_recipe(visit)
            return {"ok": True, "serves": recipe.serves, "originally_serves": recipe.base_serves,
                    "ingredients": recipe.ingredients}
        if name == "choose_recipe":
            wanted = str(args.get("recipe_name") or "")
            subject = self.knowledge.resolve_recipe(wanted)
            if subject is None:
                names = [r["name"] for r in self.knowledge.recipes()]
                close = difflib.get_close_matches(wanted, names, n=2, cutoff=0.3)
                return {"ok": False, "error": f"{wanted!r} is not in the catalog.",
                        "closest_catalog_recipes": close}
            self.store.set_recipe(visit, subject.id, subject.name)
            recipe = self.context.active_recipe(visit)
            # "The first one, for 4 people": choose and scale in one call.
            try:
                servings = int(args.get("servings") or 0)
            except (TypeError, ValueError):
                servings = 0
            if recipe.base_serves and 1 <= servings <= MAX_SERVINGS and servings != recipe.base_serves:
                self.store.set_servings(visit, servings)
                recipe = self.context.active_recipe(visit)
            return {"ok": True, "recipe": recipe.name, "serves": recipe.serves,
                    "originally_serves": recipe.base_serves,
                    "time_minutes": recipe.time_minutes, "ingredients": recipe.ingredients,
                    "steps": [s["text"] for s in recipe.steps],
                    "status": "Loaded, not started. Show the ingredients as bullets and a "
                              "short numbered overview of the steps, then ask if they're ready. "
                              "When they say they're ready, call go_to_step(1)."}
        if name == "go_to_step":
            recipe = self.context.active_recipe(visit)
            if recipe is None:
                return {"ok": False, "error": "No recipe is active. Choose one first."}
            try:
                number = int(args.get("step"))
            except (TypeError, ValueError):
                return {"ok": False, "error": "step must be a whole number."}
            total = len(recipe.steps)
            if number <= 0:
                self.store.set_step(visit, None)
                return {"ok": True, "step": 0, "status": "Back at the ingredient overview."}
            if number > total:
                self.store.set_step(visit, total)
                return {"ok": True, "step": total, "status": "All steps are finished."}
            self.store.set_step(visit, number - 1)
            step = recipe.steps[number - 1]
            return {"ok": True, "step": number, "of": total, "text": step["text"],
                    "done_when": step.get("done_when"), "timer_seconds": step.get("timer_seconds")}
        if name == "stop_recipe":
            self.store.set_recipe(visit, None, None)
            return {"ok": True, "status": "No recipe is active now."}
        if name == "update_cook_profile":
            values = {"cook_name": args.get("name"), "skill_level": args.get("skill_level"),
                      "dietary": args.get("dietary")}
            values = {k: str(v)[:limit] for (k, v), limit in
                      zip(values.items(), (80, 120, 1000)) if v not in (None, "")}
            if not values:
                return {"ok": False, "error": "Nothing to update."}
            return {"ok": True, "profile": self.store.update_profile(user_id, values)}
        return {"ok": False, "error": f"Unknown tool {name}."}

    # -- one turn -----------------------------------------------------------
    async def process_message(self, message: str, user_id: str = "default_user",
                              frames: Optional[List[tuple]] = None,
                              upload: Optional[tuple] = None, on_delta=None,
                              display_text: str = None, allow_nav: bool = True) -> Dict[str, Any]:
        """Answer one message.

        ``frames`` are the latest camera frames as (bytes, mime), oldest first;
        ``upload`` is a photo the cook attached on purpose. Returns the reply
        plus the state the phone needs to update its screen.
        """
        try:
            visit = self.store.current_visit(user_id)
            frames = frames or []
            nav_command = allow_nav and visit.has_recipe and detect_navigation(message)

            if upload is not None and not self.vision.available:
                return self._result(user_id, "Photo analysis is unavailable: no vision model "
                                             "is set up. Ask your question without the photo.")

            # Notes-only mode has no tools; still let a named dish start.
            if not self.model.available and not visit.has_recipe:
                named = self.knowledge.resolve_recipe(message)
                if named is not None:
                    self.store.set_recipe(visit, named.id, named.name)

            images, observation = [], None
            if self.vision.available and (upload is not None or (frames and not nav_command)):
                if self.vision.inline:
                    if upload is not None:
                        images = [(to_data_url(*upload), "auto")]
                    else:
                        images = [(to_data_url(*frame), "low") for frame in frames]
                else:
                    observation = await self._look(user_id, message, upload or frames[-1],
                                                   force=upload is not None)
                    if upload is not None and observation is None:
                        return self._result(user_id, "I couldn't analyze the uploaded photo. "
                                                     "Check your vision model settings and try again.")

            turn = await asyncio.to_thread(self.context.build, user_id, message, observation,
                                           len(images), allow_nav)
            logger.info("Resolved %r", turn)

            history = await asyncio.to_thread(self.store.recent_turns,
                                              turn.visit.session_id, HISTORY_TURNS)
            actions = {"suggestions": [], "timers": []}

            async def tool_handler(name, args):
                return await asyncio.to_thread(self.run_tool, user_id, name, args, actions)

            context = {
                "prompt_context": turn.text,
                "language": turn.language,
                "query_type": "step" if turn.nav else "help",
                "social": turn.social,
                "facet": turn.facet,
                "history": history,
                "images": images,
                "tools": None if turn.nav else _tools(),
                "tool_handler": tool_handler,
                "missing_notes": not turn.facts and not turn.observation,
            }
            reply = await self.model.process_query_with_openai(
                query=message, context=context, user_id=user_id, on_delta=on_delta)

            vision_note = "Looked at the uploaded photo." if upload is not None and images else None
            await asyncio.to_thread(self.context.record, turn, reply, None, "model",
                                    display_text, vision_note)
            return self._result(user_id, reply, observation=observation,
                                sources=context.get("sources"),
                                looked=len(images) if images and upload is None else 0,
                                actions=actions)

        except Exception as exc:
            logger.error("Processing error: %s", exc, exc_info=True)
            return self._result(user_id, "Sorry, something went wrong on my side. "
                                         "Could you say that again?")

    async def _look(self, user_id: str, message: str, image: tuple, force=False):
        """Ask a separate VLM about one frame, if this question needs eyes."""
        if not self.vision.available:
            return None
        if not force and (detect_navigation(message) or not wants_vision(
                message, classify_facet(message), self.vision.config.mode)):
            return None
        visit = self.store.current_visit(user_id)
        recipe = self.context.active_recipe(visit)
        request = VisionRequest(question=message, recipe=visit.recipe_name)
        if recipe is not None and visit.step is not None and visit.step < len(recipe.steps):
            request.step_text = recipe.steps[visit.step]["text"]
            request.done_when = recipe.steps[visit.step].get("done_when")
        raw, mime = image
        return await self.vision.observe(raw, mime, request)

    # -- uploaded recipes -------------------------------------------------------
    async def import_recipe(self, user_id: str, text: str = "", image: Optional[tuple] = None,
                            on_delta=None, url: str = "") -> Dict[str, Any]:
        """Parse an uploaded recipe (text, photo or web link), make it active, present it."""
        if not self.model.available:
            return self._result(user_id, "Reading an uploaded recipe needs the AI model. "
                                         "Add your API key in the launcher's settings.")
        try:
            if url:
                page = await fetch_recipe_page(url)
                text = f"{text}\n\n{page}".strip() if text else page
            recipe = await parse_recipe(self.model.streaming_openai.client, self.model.config,
                                        text, image)
        except RecipeImportError as exc:
            return self._result(user_id, str(exc))
        visit = self.store.current_visit(user_id)
        custom_id = await asyncio.to_thread(self.store.save_custom_recipe, visit, recipe)
        await asyncio.to_thread(self.store.set_recipe, visit, None, recipe["name"], custom_id)
        logger.info("Imported %r: %d steps", recipe["name"], len(recipe["steps"]))
        return await self.process_message(
            f"I just uploaded my own recipe for {recipe['name']}. Give me the overview.",
            user_id, on_delta=on_delta, display_text=f"Uploaded recipe: {recipe['name']}",
            allow_nav=False)

    async def open_saved_recipe(self, user_id: str, custom_id: Any, on_delta=None) -> Dict[str, Any]:
        """Cook one of this browser's uploaded recipes again."""
        data = self.store.custom_recipe(custom_id, user_id) if type(custom_id) is int else None
        if data is None:
            return self._result(user_id, "I couldn't find that saved recipe.")
        visit = self.store.current_visit(user_id)
        await asyncio.to_thread(self.store.set_recipe, visit, None, data["name"], custom_id)
        return await self.process_message(
            f"I want to cook my saved recipe {data['name']} again. Give me the overview.",
            user_id, on_delta=on_delta, display_text=f"Cook again: {data['name']}", allow_nav=False)

    # -- end of a recipe ------------------------------------------------------------
    def set_servings(self, user_id: str, servings: Any) -> Dict[str, Any]:
        """The phone's servings stepper: rescale without a model call."""
        visit = self.store.current_visit(user_id)
        recipe = self.context.active_recipe(visit)
        if recipe is None or not recipe.base_serves:
            raise ValueError("This recipe can't be scaled.")
        if type(servings) is not int:
            raise ValueError(f"Servings must be a whole number from 1 to {MAX_SERVINGS}.")
        self.store.set_servings(visit, None if servings == recipe.base_serves else servings)
        return self.cooking_state(user_id)

    async def finish(self, user_id: str, rating: Optional[str], note: str = "",
                     on_delta=None) -> Dict[str, Any]:
        """'How did it go?': remember the note for next time, then let the coach reply."""
        visit = self.store.current_visit(user_id)
        rating = rating if rating in RATINGS else None
        note = await asyncio.to_thread(self.store.save_feedback, visit, rating, note)
        words = {"loved": "I loved it", "ok": "It was OK", "struggled": "I struggled with it"}
        said = ". ".join(filter(None, [words.get(rating, "I've finished"),
                                        f"Note for next time: {note}" if note else ""]))
        return await self.process_message(
            f"I've finished cooking {visit.recipe_name or 'the dish'}. {said}. "
            f"(The note is already saved to my profile.)", user_id, on_delta=on_delta,
            display_text=said, allow_nav=False)

    # -- camera watch -------------------------------------------------------------
    async def watch(self, user_id: str, frames: List[tuple],
                    last_alert: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Look at the latest frames against the current step; maybe speak up.

        Returns None when there is nothing to check (no step in progress, no
        inline vision), else {"status", "message", "step"}.
        """
        if not self.watch_available or not frames:
            return None
        visit = self.store.current_visit(user_id)
        recipe = self.context.active_recipe(visit)
        if recipe is None or visit.step is None or visit.step >= len(recipe.steps):
            return None
        step = recipe.steps[visit.step]
        lines = [f"Recipe: {recipe.name}.",
                 f"Current step {visit.step + 1} of {len(recipe.steps)}: {step['text']}"]
        if step.get("done_when"):
            lines.append(f"Done when: {step['done_when']}")
        if visit.step + 1 < len(recipe.steps):
            lines.append(f"Next step: {recipe.steps[visit.step + 1]['text']}")
        if visit.dietary:
            lines.append(f"Allergies or foods they avoid: {visit.dietary}")
        lines.append(f"Previous alert: {last_alert}" if last_alert else "Previous alert: none.")
        from streaming_openai import LANGUAGE_INSTRUCTIONS
        if visit.language in LANGUAGE_INSTRUCTIONS and visit.language != "en":
            lines.append(f"Write \"message\" in this language: {LANGUAGE_INSTRUCTIONS[visit.language]}")
        content = [{"type": "text", "text": "\n".join(lines)}] + [
            {"type": "image_url", "image_url": {"url": to_data_url(*f), "detail": "low"}}
            for f in frames]
        reply = await self.model.complete([{"role": "system", "content": WATCH_PROMPT},
                                           {"role": "user", "content": content}],
                                          budget=700, json_mode=True)
        data = _json_object(reply)
        status = str(data.get("status") or "unclear").lower()
        status = status if status in WATCH_STATUSES else "unclear"
        message = " ".join(str(data.get("message") or "").split())[:400]
        seen = " ".join(str(data.get("seen") or "").split())[:300] or None
        logger.info("Watch %s (step %d): %s", status, visit.step + 1, seen)
        if status in ("issue", "hazard", "step_done") and message:
            await asyncio.to_thread(self.store.record_turn, visit, None, None, message,
                                    intent=status, source="camera-watch", vision=seen)
            return {"status": status, "message": message, "step": visit.step}
        return {"status": status, "message": "", "step": visit.step}

    # -- results -------------------------------------------------------------------
    def _result(self, user_id, reply, observation=None, sources=None, looked=0,
                actions=None) -> Dict[str, Any]:
        vision = observation.to_dict() if observation else None
        if vision is None and looked:
            vision = {"summary": None, "frames": looked, "hazards": []}
        actions = actions or {}
        return {
            "message": reply,
            "cooking": self.cooking_state(user_id),
            "vision": vision,
            "sources": sources or [],
            "suggestions": actions.get("suggestions") or [],
            "timers": actions.get("timers") or [],
        }

    # -- reporting -----------------------------------------------------------
    def get_user_insights(self, user_id):
        return self.store.visit_summary(user_id)


def _tools():
    from streaming_openai import TOOLS
    return TOOLS


def _json_object(text: Optional[str]) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


async def demo_session():
    """Walk through a short session against the configured model, printing replies."""
    assistant = CookingAssistant()
    user_id = "demo_cook"
    print("Coach:", await assistant.greet(user_id))
    for message in ("Any ideas for a quick dinner? I'm a beginner.", "the first one",
                    "start", "next", "what does al dente mean?"):
        result = await assistant.process_message(message, user_id)
        cooking = result["cooking"]
        where = (f"step {cooking['step'] + 1}/{cooking['total_steps']}"
                 if cooking["step"] is not None else "not started")
        print(f"\nCook: {message}   [{cooking['recipe']} · {where}]\nCoach: {result['message']}")


if __name__ == "__main__":
    asyncio.run(demo_session())
