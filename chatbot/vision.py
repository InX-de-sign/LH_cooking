# vision.py - What the phone camera sees, via a vision-language model
"""
The camera half of the coach. The phone sends a JPEG frame with a question;
a VLM turns it into a short structured observation; context_builder puts that
observation into the [NOW] layer of the prompt next to the current recipe
step. The VLM never talks to the cook directly -- gpt-5-mini still writes
every reply, grounded in the recipe notes.

WHICH VLM
---------
Not decided here. Pick one with environment variables:

    VLM_PROVIDER=none                 no camera understanding (default)
    VLM_PROVIDER=openai_compatible    any OpenAI-style /chat/completions
                                      endpoint that accepts image_url parts:
                                      Tencent Hunyuan, Qwen-VL, OpenAI, a
                                      self-hosted vLLM, ...
        VLM_BASE_URL   e.g. https://api.hunyuan.cloud.tencent.com/v1
        VLM_API_KEY
        VLM_MODEL
    VLM_PROVIDER=azure                an image-capable HKUST GenAI model or
                                      direct Azure deployment, using the same
                                      endpoint and key as the chat model
        VLM_MODEL      deployment name (defaults to AZURE_DEPLOYMENT_NAME)

A provider that fits neither shape is one subclass of VisionProvider: take
image bytes and a VisionRequest, return an Observation (or None).

INLINE VISION (azure)
---------------------
When the VLM is the same image-capable chat model (VLM_PROVIDER=azure, e.g.
gpt-5-mini on HKUST), there is no separate observation step: the phone's
latest camera frames (or an uploaded photo) are attached straight to the
chat call, and the model that writes the reply looks at them itself. That is
one call instead of two, and the model sees the frames next to the recipe
step it is judging. Separate VLMs (openai_compatible) still produce a text
Observation first.

WHEN TO LOOK
------------
With a separate VLM, a call adds a second or more before the reply can start,
so by default it only looks when the question needs eyes -- see
wants_vision(). VLM_MODE=always sends every frame the phone attaches.

PRIVACY
-------
Frames come from inside someone's home. They are held in memory for one
request and never written to disk or logged; only the text observation is
stored with the turn.
"""

import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ALLOWED_MIME = ("image/jpeg", "image/png", "image/webp")
_DATA_URL = re.compile(r"^data:(?P<mime>image/[a-z]+);base64,(?P<data>.+)$", re.S)

# Questions that need eyes. Matched as words/phrases in the lowercased text.
# Generic words ("okay", "right", "what's") are deliberately absent: they fire
# on "okay, what's next?", which is step navigation and needs no picture.
VISION_MARKERS = (
    "this", "these", "those", "look", "looks", "see", "camera", "check",
    "is it done", "is it ready", "done yet", "ready yet", "hot enough",
    "burnt", "burning", "brown", "golden", "colour", "color", "raw",
    "cooked through", "too thick", "too thin", "smoking", "smoke", "bubbling",
)
VISION_FACETS = frozenset({"doneness", "safety"})


@dataclass
class VisionConfig:
    provider: str = "none"
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout_seconds: float = 10.0
    max_tokens: int = 300
    mode: str = "auto"               # auto | always
    max_image_bytes: int = 1_500_000


def load_vision_config() -> VisionConfig:
    return VisionConfig(
        provider=os.getenv("VLM_PROVIDER", "none").strip().lower(),
        model=os.getenv("VLM_MODEL") or None,
        base_url=os.getenv("VLM_BASE_URL") or None,
        api_key=os.getenv("VLM_API_KEY") or None,
        timeout_seconds=float(os.getenv("VLM_TIMEOUT_SECONDS", "10")),
        max_tokens=int(os.getenv("VLM_MAX_TOKENS", "300")),
        mode=os.getenv("VLM_MODE", "auto").strip().lower(),
        max_image_bytes=int(os.getenv("VLM_IMAGE_MAX_BYTES", "1500000")),
    )


@dataclass
class VisionRequest:
    """What the VLM is told alongside the picture."""
    question: str
    recipe: Optional[str] = None
    step_text: Optional[str] = None
    done_when: Optional[str] = None


@dataclass
class Observation:
    """One frame, described."""
    summary: str
    items: List[str] = field(default_factory=list)
    state: Optional[str] = None
    hazards: List[str] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    latency_ms: int = 0

    def prompt_lines(self) -> List[str]:
        lines = [f"The camera shows: {self.summary}"]
        if self.state:
            lines.append(f"How it looks: {self.state}")
        if self.hazards:
            lines.append(f"SAFETY - the camera sees: {'; '.join(self.hazards)}")
        return lines

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ImageError(ValueError):
    """The attached image could not be used."""


def decode_image(data_url: str, max_bytes: int) -> tuple:
    """Turn a browser data URL into (bytes, mime). Raises ImageError."""
    match = _DATA_URL.match((data_url or "").strip())
    if not match:
        raise ImageError("image must be a base64 data URL")
    mime = match.group("mime")
    if mime not in ALLOWED_MIME:
        raise ImageError(f"unsupported image type {mime}")
    try:
        raw = base64.b64decode(match.group("data"), validate=True)
    except ValueError as exc:
        raise ImageError("image is not valid base64") from exc
    if len(raw) > max_bytes:
        raise ImageError(f"image is {len(raw)} bytes; the limit is {max_bytes}")
    return raw, mime


MAX_FRAMES = 4


def decode_images(data_urls, max_bytes: int, limit: int = MAX_FRAMES) -> List[tuple]:
    """Decode a list of camera-frame data URLs, keeping the newest ``limit``.

    Raises ImageError if the list is malformed or any frame is invalid.
    """
    if not isinstance(data_urls, list):
        raise ImageError("images must be a list of data URLs")
    return [decode_image(url, max_bytes) for url in data_urls[-limit:]]


def to_data_url(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def wants_vision(message: str, facet: Optional[str], mode: str = "auto") -> bool:
    """Does this question need the camera?"""
    if mode == "always":
        return True
    if facet in VISION_FACETS:
        return True
    lowered = (message or "").lower()
    return any(re.search(rf"\b{re.escape(marker)}\b", lowered) for marker in VISION_MARKERS)


SYSTEM_PROMPT = """You are the eyes of a cooking coach. You see one photo from \
the cook's phone, pointed at what they are cooking. Describe only what is \
actually visible.

Reply with JSON only, no prose and no code fence:
{"summary": "<one plain sentence describing the scene>",
 "items": ["<visible foods and tools, common lowercase names, at most 8>"],
 "state": "<how cooked or prepared the main food looks, or null>",
 "hazards": ["<visible safety problems, e.g. smoke, flames, boiling over, \
knife near the counter edge, raw meat touching ready-to-eat food; empty if none>"]}

If the photo is too dark, blurry or does not show food, say so in summary and \
leave the lists empty. Never guess at what is not visible."""


def _user_text(request: VisionRequest) -> str:
    lines = [f"The cook asked: {request.question}"]
    if request.recipe:
        lines.append(f"They are making: {request.recipe}.")
    if request.step_text:
        lines.append(f"Current step: {request.step_text}")
    if request.done_when:
        lines.append(f"The step is done when: {request.done_when}")
    lines.append("Judge the state against that step if it is visible.")
    return "\n".join(lines)


def _parse(text: str) -> Optional[Dict[str, Any]]:
    """Pull the JSON object out of a model reply, tolerating fences and chatter."""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _as_list(value) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        value = [value]
    return [str(v).strip() for v in value if str(v).strip()][:8]


class VisionProvider:
    """Base class. Subclass and implement _complete() for a new VLM."""

    name = "none"
    # True when the chat model itself can take the frames (see INLINE VISION).
    inline = False

    def __init__(self, config: VisionConfig):
        self.config = config
        self.last_error = None

    @property
    def available(self) -> bool:
        return False

    async def observe(self, image: bytes, mime: str,
                      request: VisionRequest) -> Optional[Observation]:
        if not self.available:
            return None
        started = time.monotonic()
        self.last_error = None
        try:
            reply = await self._complete(image, mime, request)
        except Exception as exc:
            # A failed look must never cost the cook their answer.
            status = getattr(exc, "status_code", None)
            if status == 401:
                self.last_error = "HKUST rejected the API key for this vision endpoint (HTTP 401)."
            elif status == 403:
                self.last_error = "This API key cannot use the selected vision model (HTTP 403)."
            elif status == 404:
                self.last_error = "The selected vision endpoint or model was not found (HTTP 404)."
            elif status:
                self.last_error = f"The vision API returned HTTP {status}."
            elif type(exc).__name__ in ("APITimeoutError", "APIConnectionError"):
                self.last_error = "Could not connect to the vision API. Check the network and endpoint."
            else:
                self.last_error = "The vision API request failed. Check the model settings."
            logger.warning("VLM %s failed: %s", self.name, self.last_error)
            return None
        latency = int((time.monotonic() - started) * 1000)

        data = _parse(reply)
        if data is None:
            # Unstructured but non-empty is still worth having.
            summary = " ".join((reply or "").split())[:300]
            if not summary:
                self.last_error = "The vision model returned an empty reply. Try a larger output token budget."
                logger.warning("VLM %s returned nothing (%d ms)", self.name, latency)
                return None
            data = {"summary": summary}

        observation = Observation(
            summary=str(data.get("summary") or "").strip() or "nothing clear",
            items=_as_list(data.get("items")),
            state=(str(data["state"]).strip() or None) if data.get("state") else None,
            hazards=_as_list(data.get("hazards")),
            provider=self.name,
            model=self.config.model or "",
            latency_ms=latency,
        )
        logger.info("VLM %s (%d ms): %s", self.name, latency, observation.summary[:100])
        return observation

    async def _complete(self, image: bytes, mime: str, request: VisionRequest) -> str:
        raise NotImplementedError

    def _messages(self, image: bytes, mime: str, request: VisionRequest) -> list:
        data_url = f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": _user_text(request)},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ]

    def describe(self) -> Dict[str, Any]:
        return {"provider": self.name, "model": self.config.model,
                "available": self.available, "mode": self.config.mode,
                "inline": self.inline}


class NoVision(VisionProvider):
    name = "none"


class OpenAICompatibleVision(VisionProvider):
    """Any endpoint speaking OpenAI chat completions with image_url parts."""

    name = "openai_compatible"

    def __init__(self, config: VisionConfig):
        super().__init__(config)
        missing = [n for n, v in (("VLM_BASE_URL", config.base_url),
                                  ("VLM_API_KEY", config.api_key),
                                  ("VLM_MODEL", config.model)) if not v]
        if missing:
            logger.error("VLM_PROVIDER=openai_compatible needs %s - camera disabled",
                         ", ".join(missing))
            self.client = None
            return
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key,
                                  timeout=config.timeout_seconds, max_retries=0)

    @property
    def available(self) -> bool:
        return self.client is not None

    async def _complete(self, image, mime, request) -> str:
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=self._messages(image, mime, request),
            max_tokens=self.config.max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message.content if response.choices else ""


class AzureVision(VisionProvider):
    """An image-capable model on the configured HKUST or Azure endpoint."""

    name = "azure"

    @property
    def inline(self) -> bool:
        # Frames go straight into the chat call when the vision deployment is
        # the chat deployment; a different deployment keeps the two-step path.
        return self.available and self.config.model == load_chat_deployment()

    def __init__(self, config: VisionConfig):
        super().__init__(config)
        from config import load_azure_openai_config
        from api_client import create_async_client

        self.chat_config = load_azure_openai_config()
        self.client = None
        if not self.chat_config.configured:
            logger.error("VLM_PROVIDER=azure needs the chat-model slot (OPENAI_API_KEY, "
                         "AZURE_OPENAI_ENDPOINT) filled in - camera disabled")
            return
        # Reuse the chat config's parameter logic: a gpt-5 model takes
        # max_completion_tokens and temperature=1, a gpt-4o one does not.
        self.chat_config.deployment_name = config.model or self.chat_config.deployment_name
        config.model = self.chat_config.deployment_name
        self.client = create_async_client(
            self.chat_config,
            timeout=config.timeout_seconds,
            max_retries=0,
        )

    @property
    def available(self) -> bool:
        return self.client is not None

    async def _complete(self, image, mime, request) -> str:
        response = await self.client.chat.completions.create(
            model=self.chat_config.deployment_name,
            messages=self._messages(image, mime, request),
            **self.chat_config.completion_kwargs(self.config.max_tokens),
        )
        return response.choices[0].message.content if response.choices else ""


def load_chat_deployment() -> str:
    from config import load_azure_openai_config
    return load_azure_openai_config().deployment_name


PROVIDERS = {
    "none": NoVision,
    "openai_compatible": OpenAICompatibleVision,
    "azure": AzureVision,
}


def load_vision_provider(config: VisionConfig = None) -> VisionProvider:
    config = config or load_vision_config()
    cls = PROVIDERS.get(config.provider)
    if cls is None:
        logger.error("Unknown VLM_PROVIDER %r - camera disabled. Choose one of %s",
                     config.provider, ", ".join(PROVIDERS))
        cls = NoVision
    provider = cls(config)
    logger.info("Vision: %s", provider.describe())
    return provider
