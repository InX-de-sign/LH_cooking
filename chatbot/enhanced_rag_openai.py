# enhanced_rag_openai.py - the model layer
"""Turns a prepared prompt context into an answer from the chat model.

Retrieval lives in knowledge_base.py and prompt assembly in
streaming_openai.py. What is left here is the model call itself, plus a
notes-only fallback used when no model is configured or a call fails.

NO MODEL CONFIGURED
-------------------
With the chat-model slot in chatbot/.env left empty, the coach still runs:
steps are read verbatim and questions are answered with the retrieved recipe
notes. That is enough to test the phone app, navigation and retrieval before
anyone picks a model.
"""

import logging
import os
import re
from typing import Any, Dict, Optional

from config import load_azure_openai_config
from web_search import BuiltinWebSearch, SearchUnavailable, explicit_search, should_search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# A rendered [FACTS] line: "- (tip) al dente: Bite a piece: you want..."
_FACT_LINE = re.compile(r"^-\s*\((?P<facet>[a-z]+)\)\s*(?P<subject>[^:]+):\s*(?P<text>.+)$")


class EnhancedRAGWithOpenAI:
    """Azure OpenAI client for the cooking coach, or notes-only without one."""

    def __init__(self):
        self.config = load_azure_openai_config()
        self.openai_client = None
        self.streaming_openai = None
        self.web_search = BuiltinWebSearch()

        if not self.config.configured:
            logger.warning("No chat model configured (OPENAI_API_KEY is empty) - "
                           "replies come from the recipe notes only. Fill in the "
                           "chat-model slot in chatbot/.env to enable the model.")
            return

        from api_client import create_client
        self.openai_client = create_client(
            self.config,
            timeout=float(os.getenv("AZURE_OPENAI_TIMEOUT_SECONDS", "30")),
            max_retries=1,
        )
        # The personas live with the prompt builder that uses them.
        from streaming_openai import StreamingOpenAI
        self.streaming_openai = StreamingOpenAI()

    @property
    def available(self) -> bool:
        return self.openai_client is not None

    def describe(self) -> Dict[str, Any]:
        if not self.available:
            return {"configured": False, "mode": "recipe notes only", "web_search": self.web_search.describe()}
        from api_client import base_url
        return {
            "configured": True,
            "endpoint": base_url(self.config),
            "deployment": self.config.deployment_name,
            "api_version": self.config.api_version,
            "reasoning_model": self.config.is_reasoning_model,
            "parameters": self.config.completion_kwargs(),
            "web_search": self.web_search.describe(),
        }

    async def process_query_with_openai(self, query: str, context: Dict[str, Any],
                                        user_id: str, on_delta=None) -> str:
        """Answer, optionally delivering streamed text through an async callback."""
        context["sources"] = []
        notice = ""
        explicit = explicit_search(query)
        # Automatic lookups only when search is switched on; an explicit
        # "search online for..." is always tried.
        if should_search(query, context) and (explicit or getattr(self.web_search, "enabled", True)):
            try:
                result = await self.web_search.search(query, context=context.get("prompt_context"))
                context["sources"] = result.sources
                return result.text
            except SearchUnavailable as exc:
                logger.info("Web search unavailable (%s); answering from the model", exc)
                if not self.available:
                    return "Online search is unavailable. " + str(exc)
                if explicit:
                    # A model answer must never pass for an online lookup.
                    notice = ("I can't search the web from here, so this comes from my own cooking "
                              "knowledge, not online sources.\n\n")
                    # The note is already shown, so ask the model the question
                    # without "search online for", or it apologises a second time.
                    plain = re.sub(r"\b(please\s+)?(search|look\s*up|find|check)\s+(it\s+)?(online|on\s+the\s+web|the\s+web)"
                                   r"(\s+for)?\s*|\b(online|on\s+the\s+web)\b", "", query, flags=re.I).strip(" ,.") or query
                    rendered = context.get("prompt_context") or f"[QUESTION]\n{query}"
                    context["prompt_context"] = rendered.replace(f"[QUESTION]\n{query}", f"[QUESTION]\n{plain}") + (
                        "\n\n[NOTE]\nOnline search is not available and the cook has already been told. "
                        "Answer from your own knowledge without mentioning search.")
        if not self.available:
            return self._offline_fallback(query, context)
        parts = []
        if notice:
            parts.append(notice)
            if on_delta is not None:
                await on_delta(notice)
        try:
            async for delta in self.streaming_openai.stream_response(query, context):
                parts.append(delta)
                if on_delta is not None:
                    await on_delta(delta)
            reply = "".join(parts).strip()
            if not reply or reply == notice.strip():
                return notice + self._offline_fallback(query, context, failed=True)
            return reply
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            if parts and on_delta is not None:
                # Do not replace words the cook has already heard with a new answer.
                suffix = " My connection was interrupted. Please ask me to repeat that."
                await on_delta(suffix)
                return "".join(parts).strip() + suffix
            return self._offline_fallback(query, context, failed=True)

    async def complete(self, messages, budget: int = 800, json_mode: bool = False) -> Optional[str]:
        """One non-streamed call (greetings, camera checks). None on failure."""
        if not self.available:
            return None
        try:
            return await self.streaming_openai.complete(messages, budget, json_mode) or None
        except Exception as e:
            logger.error(f"Chat model call failed: {type(e).__name__}: {e}")
            return None

    # -- notes only -----------------------------------------------------------
    def _offline_fallback(self, query: str, context: Dict[str, Any],
                          failed: bool = False) -> str:
        """Answer from what ContextBuilder already rendered, without the model.

        The current step and the retrieved facts were read from knowledge.db
        moments earlier, so reading them back out is grounded by construction.
        Never a hardcoded cooking claim.
        """
        step = self._current_step_from_context(context)
        if context.get('query_type') == 'step':
            rendered = context.get('prompt_context') or ""
            if step:
                return step[0].upper() + step[1:]
            if "steps are finished" in rendered:
                return "That was the last step. Enjoy your meal!"
            if "No recipe chosen yet" in rendered:
                return "Which dish are you making? Tell me and we'll go step by step."

        if context.get('social'):
            return "Happy to help. Tell me what you'd like to cook, or say next for the next step."

        # "Is this done?" mid-recipe: the current step's own cue beats any
        # doneness fact retrieval found elsewhere in the recipe.
        if context.get('facet') == 'doneness':
            rendered = context.get('prompt_context') or ""
            for line in rendered.splitlines():
                if line.startswith("Done when: "):
                    return f"For this step, it's done when: {line[len('Done when: '):]}"

        facts = self._facts_from_context(context)
        if facts:
            return " ".join(fact["text"] for fact in facts[:2])

        if failed:
            return "I can't reach my AI model right now. Please ask me again in a moment."
        return "I don't have notes on that yet."

    @staticmethod
    def _current_step_from_context(context: Dict[str, Any]) -> Optional[str]:
        rendered = context.get('prompt_context') or ""
        for line in rendered.splitlines():
            if line.startswith("Current step "):
                return line[len("Current "):]
        return None

    @staticmethod
    def _facts_from_context(context: Dict[str, Any]) -> list:
        """Pull the [FACTS] block back out of the rendered prompt context."""
        rendered = context.get('prompt_context') or ""
        facts = []
        in_facts = False
        for line in rendered.splitlines():
            stripped = line.strip()
            if stripped.startswith("["):
                in_facts = stripped.startswith("[FACTS]")
                continue
            if not in_facts:
                continue
            match = _FACT_LINE.match(stripped)
            if match:
                facts.append(match.groupdict())
        return facts
