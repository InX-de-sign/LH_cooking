# recipe_import.py - Turn an uploaded recipe into structured steps
"""
A cook can bring their own recipe: pasted text, a text file, or a photo of a
cookbook page or screenshot. The chat model reads it (JSON mode, with the
image attached when there is one) and returns the same shape as a catalog
recipe, so the step card, navigation, timers and camera checks all work on it
unchanged.

The model is told to keep the recipe's own quantities and wording, split
compound instructions into one action per step, and only add a "done when"
cue or timer where the recipe implies one.
"""

import base64
import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 20_000
MAX_STEPS = 40
MAX_INGREDIENTS = 60

SYSTEM_PROMPT = """You turn a home cook's recipe into structured cooking steps.
The recipe arrives as text, a photo of a page or screenshot, or both.

Return JSON only:
{"name": "<dish name>",
 "serves": <integer or null>,
 "time_minutes": <total minutes as an integer, or null>,
 "ingredients": ["<quantity + ingredient, exactly as written>", ...],
 "steps": [{"text": "<one clear action, keeping quantities, heat levels and times>",
            "done_when": "<visible cue that this step is finished, or null>",
            "timer_seconds": <integer if the step has a fixed wait, else null>}, ...],
 "error": null}

Rules:
- Keep the recipe's own quantities, temperatures and times. Never invent them.
- One action per step. Split long paragraphs into separate steps, in order.
- "done_when" only when the recipe states or clearly implies a cue
  (e.g. "until golden", "until the onions are soft").
- If there is no recipe in the input, return {"error": "<short reason>"} instead."""


class RecipeImportError(ValueError):
    """The upload could not be turned into a recipe."""


def _clean_text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _positive_int(value: Any, maximum: int) -> Optional[int]:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number if 0 < number <= maximum else None


def validate_recipe(data: Any) -> Dict[str, Any]:
    """Check and normalise the model's JSON. Raises RecipeImportError."""
    if not isinstance(data, dict):
        raise RecipeImportError("The recipe could not be read. Try pasting the text instead.")
    if data.get("error"):
        raise RecipeImportError(f"No recipe found: {_clean_text(data['error'], 200)}")
    name = _clean_text(data.get("name"), 120)
    steps = []
    for raw in (data.get("steps") or [])[:MAX_STEPS]:
        if isinstance(raw, str):
            raw = {"text": raw}
        if not isinstance(raw, dict):
            continue
        text = _clean_text(raw.get("text"), 600)
        if not text:
            continue
        steps.append({
            "text": text,
            "done_when": _clean_text(raw.get("done_when"), 200) or None,
            "timer_seconds": _positive_int(raw.get("timer_seconds"), 6 * 60 * 60),
        })
    ingredients = [_clean_text(i, 200) for i in (data.get("ingredients") or [])[:MAX_INGREDIENTS]]
    ingredients = [i for i in ingredients if i]
    if not name or not steps:
        raise RecipeImportError("I couldn't find a dish name and steps in that. "
                                "Try pasting the full recipe text.")
    return {
        "name": name,
        "serves": _positive_int(data.get("serves"), 100),
        "time_minutes": _positive_int(data.get("time_minutes"), 7 * 24 * 60),
        "ingredients": ingredients,
        "steps": steps,
    }


def _parse_json(text: str) -> Any:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


async def parse_recipe(client, config, text: str = "", image: Optional[tuple] = None) -> Dict[str, Any]:
    """Ask the chat model to structure an uploaded recipe.

    ``image`` is (bytes, mime) or None. Raises RecipeImportError on bad input
    or an unusable reply.
    """
    text = (text or "").strip()
    if len(text) > MAX_TEXT_CHARS:
        raise RecipeImportError(f"That recipe is too long. Keep it under {MAX_TEXT_CHARS:,} characters.")
    if not text and image is None:
        raise RecipeImportError("Paste a recipe, or choose a photo or text file of one.")

    content = [{"type": "text", "text": f"Recipe:\n{text}" if text else "Recipe: see the photo."}]
    if image is not None:
        raw, mime = image
        url = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
        content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})

    try:
        response = await client.chat.completions.create(
            model=config.deployment_name,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": content}],
            response_format={"type": "json_object"},
            **config.completion_kwargs(4000),
        )
    except Exception as exc:
        logger.warning("Recipe import call failed: %s", type(exc).__name__)
        raise RecipeImportError("I couldn't reach the AI model to read that recipe. "
                                "Please try again in a moment.") from exc
    reply = response.choices[0].message.content if response.choices else ""
    return validate_recipe(_parse_json(reply))
