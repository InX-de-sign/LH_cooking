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

import asyncio
import base64
import html
import ipaddress
import json
import logging
import re
import socket
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 20_000
MAX_PAGE_BYTES = 2_000_000
MAX_REDIRECTS = 3
FETCH_TIMEOUT_SECONDS = 10
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


# -- recipe links ------------------------------------------------------------------
# The server fetches a page the cook names, so the fetch must not become a way
# to reach the server's own network: http(s) on the standard ports only, every
# resolved address must be public, redirects are followed by hand and checked
# again, and the body is capped. (A DNS answer that changes between the check
# and the request could still slip through; acceptable for public recipe pages
# on a host with nothing private listening.)

async def _check_public(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise RecipeImportError("Use a normal web link starting with https://.")
    if parsed.username or parsed.password or parsed.port not in (None, 80, 443):
        raise RecipeImportError("That link can't be opened. Paste the recipe text instead.")
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(parsed.hostname, None, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise RecipeImportError("That website couldn't be found. Check the link.") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%")[0])
        if not address.is_global:
            raise RecipeImportError("That link points to a private address and can't be opened.")


def _walk_json_ld(node) -> List[dict]:
    """Every schema.org Recipe object in a JSON-LD document, however nested."""
    found = []
    if isinstance(node, list):
        for item in node:
            found += _walk_json_ld(item)
    elif isinstance(node, dict):
        kind = node.get("@type")
        kinds = kind if isinstance(kind, list) else [kind]
        if "Recipe" in kinds:
            found.append(node)
        for key in ("@graph", "mainEntity", "itemListElement"):
            if key in node:
                found += _walk_json_ld(node[key])
    return found


def _instruction_lines(node) -> List[str]:
    if isinstance(node, str):
        return [line for line in re.split(r"\n+", html.unescape(re.sub(r"<[^>]+>", " ", node))) if line.strip()]
    if isinstance(node, list):
        return [line for item in node for line in _instruction_lines(item)]
    if isinstance(node, dict):
        if node.get("itemListElement"):
            heading = [f"{node['name']}:"] if node.get("name") else []
            return heading + _instruction_lines(node["itemListElement"])
        return _instruction_lines(node.get("text") or node.get("name") or "")
    return []


def recipe_text_from_html(page: str) -> str:
    """Recipe text from a web page: schema.org JSON-LD if present, else the visible text."""
    for block in re.findall(r"<script[^>]+application/ld\+json[^>]*>(.*?)</script>", page, re.S | re.I):
        try:
            recipes = _walk_json_ld(json.loads(html.unescape(block.strip())))
        except (json.JSONDecodeError, ValueError):
            continue
        for recipe in recipes:
            ingredients = recipe.get("recipeIngredient") or recipe.get("ingredients") or []
            steps = _instruction_lines(recipe.get("recipeInstructions"))
            if not steps:
                continue
            serves = recipe.get("recipeYield")
            serves = serves[0] if isinstance(serves, list) and serves else serves
            parts = [str(recipe.get("name") or "Recipe")]
            if serves:
                parts.append(f"Serves: {serves}")
            parts.append("Ingredients:")
            parts += [f"- {html.unescape(str(i))}" for i in ingredients]
            parts.append("Method:")
            parts += [f"{n}. {line.strip()}" for n, line in enumerate(steps, 1)]
            return "\n".join(parts)[:MAX_TEXT_CHARS]
    text = re.sub(r"<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</\1>", " ", page, flags=re.S | re.I)
    text = re.sub(r"<(br|/p|/li|/h[1-6]|/div)[^>]*>", "\n", text, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)[:MAX_TEXT_CHARS]


async def fetch_recipe_page(url: str) -> str:
    """Download a recipe page and return its recipe text. Raises RecipeImportError."""
    import httpx
    url = (url or "").strip()
    if len(url) > 2000:
        raise RecipeImportError("That link is too long.")
    async with httpx.AsyncClient(timeout=FETCH_TIMEOUT_SECONDS, follow_redirects=False,
                                 headers={"User-Agent": "LH-Cooking-Coach/1.0 (recipe import)",
                                          "Accept": "text/html,text/plain;q=0.9"}) as client:
        for _ in range(MAX_REDIRECTS + 1):
            await _check_public(url)
            try:
                async with client.stream("GET", url) as response:
                    if response.is_redirect and response.headers.get("location"):
                        url = urljoin(url, response.headers["location"])
                        continue
                    if response.status_code >= 400:
                        raise RecipeImportError(f"That page couldn't be opened (HTTP {response.status_code}). "
                                                "Paste the recipe text instead.")
                    kind = response.headers.get("content-type", "")
                    if "html" not in kind and "text/plain" not in kind:
                        raise RecipeImportError("That link isn't a web page. Paste the recipe text instead.")
                    body = b""
                    async for chunk in response.aiter_bytes():
                        body += chunk
                        if len(body) > MAX_PAGE_BYTES:
                            break
                    page = body[:MAX_PAGE_BYTES].decode(response.encoding or "utf-8", "replace")
            except httpx.HTTPError as exc:
                raise RecipeImportError("That page didn't load. Paste the recipe text instead.") from exc
            text = recipe_text_from_html(page) if "html" in kind else page[:MAX_TEXT_CHARS]
            if len(text) < 40:
                raise RecipeImportError("I couldn't find a recipe on that page. Paste the text instead.")
            return text
    raise RecipeImportError("That link redirected too many times.")


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
