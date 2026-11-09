# streaming_openai.py - Prompt assembly, tool loop and streaming calls
"""
One model call per turn (two when the model uses a tool):

    messages = [system prompt,
                last few exchanges as real user/assistant messages,
                user: language rule + context layers + camera frames]

The model can change the session through tools (choose_recipe, go_to_step,
stop_recipe, update_cook_profile). When it calls one, the tool runs, its
result is appended, and the model is called again to write the reply. Text is
streamed to the phone as it arrives in both rounds.
"""

import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, List

from api_client import create_async_client
from config import load_azure_openai_config

logger = logging.getLogger(__name__)

# Brevity is enforced by the system prompt, not the token budget: on gpt-5
# the budget also pays for reasoning, and a recipe overview with ingredients
# and numbered steps needs room.
REPLY_TOKEN_BUDGET = 1600
# One round of tool calls (possibly several in parallel), then a final reply
# without tools: a second round let the model "helpfully" start step 1 right
# after loading a recipe, before the cook said they were ready.
MAX_TOOL_ROUNDS = 1
HISTORY_TURNS = 6
FIRST_TOOLS = ("choose_recipe", "stop_recipe")

SYSTEM_PROMPT = """You are the cooking coach in a phone app, helping a home cook \
make a dish step by step. Your replies are shown in a chat and also read aloud, \
so the cook may have busy hands and be glancing at the screen.

HOW TO REPLY
- Sound like a calm, encouraging friend who cooks well. Be brief and practical:
  usually one to three short sentences.
- When you give instructions with more than one action, use a numbered list with
  one action per line ("1. ...", "2. ..."). Use "- " bullets for ingredients.
  No headings, tables, emojis or code blocks. **Bold** at most a word or two.
- When a recipe has just been chosen or uploaded: one short line, the ingredients
  as "- " bullets, then the steps as a numbered overview (a few words each), then
  ask if they are ready to start.
- When giving the current step: just that step with its quantities, then its
  "done when" cue or timer. Never paste the whole recipe again.
- When asked what to cook or for ideas: pick exactly 2 dishes from [CATALOG]
  that suit their profile and past notes, show them as cards (see ACTIONS), and
  in your reply give each a few words on why. Never list more than 2 recipes and
  never paste the catalog. Mention they can also upload their own recipe.
- Pitch explanations to their experience. If you don't know their experience or
  allergies yet, you may ask once, briefly, in passing; never run a questionnaire.
- Ask at most one question per reply, and only when it helps.

ACTIONS
You change the app by calling functions. They work silently: the cook only sees
the result (a recipe card, a step, a timer) and never sees function names. Never
write a function name, "called", or JSON in your reply; talk like a person. If
an action is needed, call the function itself; describing it does nothing.
- choose_recipe: as soon as the cook picks a catalog dish, including "the first
  one" or "that one" after your suggestions, or asks to switch dish. If the dish
  is not in the catalog, say so and offer the 2 closest catalog options or
  uploading their own recipe.
- go_to_step: when they say they finished a step, want to skip ahead, go back or
  jump to a step. Steps are numbered from 1; 0 means back to the ingredients.
  If the recipe has not started (Status: not started) and they say start, ready,
  let's go, or that they have everything, call go_to_step with step 1 and then
  give step 1. Don't show the overview again.
- stop_recipe: when they want to stop or cook something else without naming it.
- update_cook_profile: whenever the cook reveals their name, experience
  ("not confident at all" -> beginner), allergies or diet ("no allergies" ->
  none), even in passing, call it in that same turn.
- start_timer: when they ask for a timer, or say they are starting a step that
  has a fixed time in [RECIPE] ("timer for the noodles"). Use a short label.
- scale_recipe: when they want more or fewer servings ("make it for 4"). When
  they choose a dish and give a number of people in the same message, pass
  servings to choose_recipe instead.
- suggest_recipes: whenever you suggest dishes, with the same 2 you mention.
- "The first one" / "the second one" refer to the dishes you just suggested,
  in that order (see [RECIPE]), never to the order of [CATALOG].
- Never say you changed the recipe, step, timer or servings unless you called
  the function in this turn.

GROUNDING
- Quantities, times and temperatures for the recipe come only from [RECIPE] and
  [FACTS]. Never invent them. For anything else you may give general advice.
- Respect [COOK] allergies and diet: warn when a recipe or idea conflicts.

CAMERA
- Camera frames, when attached, are the latest frames from the cook's phone,
  oldest first. Use them to see what the cook is doing right now.
- For "is this right?", "how does this look?" and similar: compare what you see
  with the current step in [RECIPE]. Say plainly whether it looks right. If not,
  say what to change and why, as one or two short instructions.
- Describe only what is visible. A picture cannot show temperature, taste or
  whether meat is cooked inside: say what to check instead.
- If the frames are dark, blurry or don't show the food, say so and ask them to
  point the camera at the pan or board.
- If you see anything unsafe (smoke, flames, boiling over, a knife near the
  counter edge, raw meat touching ready-to-eat food), say that FIRST, as a short
  instruction."""

TOOLS = [
    {"type": "function", "function": {
        "name": "choose_recipe",
        "description": "Load a recipe from the catalog as the dish being cooked "
                       "(also to switch dish). Returns its ingredients and steps.",
        "parameters": {"type": "object", "properties": {
            "recipe_name": {"type": "string", "description": "Exact name from [CATALOG]."},
            "servings": {"type": "integer", "minimum": 1, "maximum": 24,
                         "description": "Only if they asked for a number of people or servings."}},
            "required": ["recipe_name"]}}},
    {"type": "function", "function": {
        "name": "go_to_step",
        "description": "Move to a step of the current recipe. 1 is the first step; "
                       "0 returns to the ingredient overview.",
        "parameters": {"type": "object", "properties": {
            "step": {"type": "integer", "minimum": 0}}, "required": ["step"]}}},
    {"type": "function", "function": {
        "name": "stop_recipe",
        "description": "Stop the current recipe so the cook can choose another.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "update_cook_profile",
        "description": "Remember what the cook said about themselves. Only include "
                       "fields they actually told you.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"},
            "skill_level": {"type": "string",
                            "description": "e.g. beginner, some experience, confident"},
            "dietary": {"type": "string",
                        "description": "Allergies and foods they avoid, or 'none'."}}}}},
    {"type": "function", "function": {
        "name": "suggest_recipes",
        "description": "Show the cook up to 2 catalog dishes as cards they can tap to "
                       "start. Call this whenever you suggest dishes.",
        "parameters": {"type": "object", "properties": {
            "recipes": {"type": "array", "maxItems": 2, "items": {
                "type": "object", "properties": {
                    "recipe_name": {"type": "string", "description": "Exact name from [CATALOG]."},
                    "why": {"type": "string", "description": "A few words on why it suits them."}},
                "required": ["recipe_name"]}}},
            "required": ["recipes"]}}},
    {"type": "function", "function": {
        "name": "start_timer",
        "description": "Start a named countdown timer on the cook's phone.",
        "parameters": {"type": "object", "properties": {
            "label": {"type": "string", "description": "Short name, e.g. 'Noodles'."},
            "seconds": {"type": "integer", "minimum": 5, "maximum": 21600}},
            "required": ["label", "seconds"]}}},
    {"type": "function", "function": {
        "name": "scale_recipe",
        "description": "Rescale the current recipe's quantities to a number of servings.",
        "parameters": {"type": "object", "properties": {
            "servings": {"type": "integer", "minimum": 1, "maximum": 24}},
            "required": ["servings"]}}},
]

LANGUAGE_INSTRUCTIONS = {
    "en": "Reply in English only.",
    "yue": ("Reply in Cantonese as people in Hong Kong write it (traditional Chinese "
            "characters, colloquial Cantonese). Keep quantities and recipe names clear."),
    "zh": "Reply in Mandarin, using simplified Chinese characters.",
}


class StreamingOpenAI:
    """Streaming HKUST GenAI or Azure client for the coach's replies."""

    def __init__(self):
        self.config = load_azure_openai_config()
        self.client = create_async_client(
            self.config,
            timeout=float(os.getenv("AZURE_OPENAI_TIMEOUT_SECONDS", "30")),
            max_retries=1,
        )
        self.system_prompt = SYSTEM_PROMPT

    # -- messages -------------------------------------------------------------
    @staticmethod
    def _language_instruction(context: Dict[str, Any]) -> str:
        return LANGUAGE_INSTRUCTIONS.get(context.get("language"), LANGUAGE_INSTRUCTIONS["en"])

    def build_messages(self, query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        for turn in context.get("history") or []:
            if turn.get("user_text"):
                messages.append({"role": "user", "content": turn["user_text"][:800]})
            if turn.get("assistant_text"):
                messages.append({"role": "assistant", "content": turn["assistant_text"][:1500]})
        text = "\n\n".join([self._language_instruction(context),
                            context.get("prompt_context") or f"[QUESTION]\n{query}"])
        images = context.get("images") or []
        if images:
            content: Any = [{"type": "text", "text": text}] + [
                {"type": "image_url", "image_url": {"url": url, "detail": detail}}
                for url, detail in images]
        else:
            content = text
        messages.append({"role": "user", "content": content})
        return messages

    # -- the streamed reply with tools ----------------------------------------
    async def stream_response(self, query: str, context: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream the reply, running any tools the model calls in between."""
        messages = self.build_messages(query, context)
        handler = context.get("tool_handler")
        tools = context.get("tools") if handler else None

        for round_number in range(MAX_TOOL_ROUNDS + 1):
            offer_tools = bool(tools) and round_number < MAX_TOOL_ROUNDS
            stream = await self.client.chat.completions.create(
                model=self.config.deployment_name,
                messages=messages,
                stream=True,
                **({"tools": tools} if offer_tools else {}),
                **self.config.completion_kwargs(REPLY_TOKEN_BUDGET),
            )
            text, calls = "", {}
            try:
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        text += delta.content
                        yield delta.content
                    for call in getattr(delta, "tool_calls", None) or []:
                        slot = calls.setdefault(call.index, {"id": "", "name": "", "arguments": ""})
                        slot["id"] = call.id or slot["id"]
                        if call.function is not None:
                            slot["name"] += call.function.name or ""
                            slot["arguments"] += call.function.arguments or ""
            finally:
                await stream.close()

            if not calls:
                logger.info("Reply streamed: %d chars after %d tool round(s)", len(text), round_number)
                return

            ordered = [calls[i] for i in sorted(calls)]
            messages.append({"role": "assistant", "content": text or None, "tool_calls": [
                {"id": c["id"] or f"call_{i}", "type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                for i, c in enumerate(ordered)]})
            # Recipe changes run first, so scaling or a step in the same round
            # applies to the recipe just chosen. Results keep the model's order.
            results = {}
            for i in sorted(range(len(ordered)), key=lambda i: ordered[i]["name"] not in FIRST_TOOLS):
                call = ordered[i]
                try:
                    arguments = json.loads(call["arguments"] or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                results[i] = await handler(call["name"], arguments if isinstance(arguments, dict) else {})
                logger.info("Tool %s(%s) -> %s", call["name"], arguments, str(results[i])[:120])
            for i, call in enumerate(ordered):
                messages.append({"role": "tool", "tool_call_id": call["id"] or f"call_{i}",
                                 "content": json.dumps(results[i])})
            if text and not text.endswith(("\n", " ")):
                yield "\n\n"

    # -- one-shot helpers -------------------------------------------------------
    async def complete(self, messages: List[Dict[str, Any]], budget: int = 800,
                       json_mode: bool = False) -> str:
        response = await self.client.chat.completions.create(
            model=self.config.deployment_name,
            messages=messages,
            **({"response_format": {"type": "json_object"}} if json_mode else {}),
            **self.config.completion_kwargs(budget),
        )
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        return ""
