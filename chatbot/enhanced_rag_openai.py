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

import asyncio
import logging
import os
import re
from typing import Any, Dict, Optional

from config import load_azure_openai_config
from web_search import BuiltinWebSearch, SearchUnavailable, should_search

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
        if should_search(query, context):
            try:
                result = await self.web_search.search(query, context=context.get("prompt_context"))
                context["sources"] = result.sources
                return result.text
            except SearchUnavailable as exc:
                # A model-only answer must never be presented as an online lookup.
                return "Online search is unavailable. " + str(exc)
        if not self.available:
            return self._offline_fallback(query, context)
        parts = []
        try:
            if on_delta is not None:
                async for delta in self.streaming_openai.stream_response(query, context):
                    parts.append(delta)
                    await on_delta(delta)
                reply = "".join(parts).strip()
                return reply or self._offline_fallback(query, context, failed=True)

            from streaming_openai import DEFAULT_PERSONA
            prompts = self.streaming_openai.system_prompts
            system_prompt = prompts.get(context.get('query_type', DEFAULT_PERSONA),
                                        prompts[DEFAULT_PERSONA])
            user_prompt = self.streaming_openai._build_short_prompt(query, context)

            response = await self._call_openai_api(system_prompt, user_prompt)
            return response or self._offline_fallback(query, context, failed=True)
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            if parts:
                # Do not replace words the cook has already heard with a new answer.
                suffix = " My connection was interrupted. Please ask me to repeat that."
                await on_delta(suffix)
                return "".join(parts).strip() + suffix
            return self._offline_fallback(query, context, failed=True)

    async def _call_openai_api(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=self.config.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                **self.config.completion_kwargs()
            )
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            return None
        except Exception as e:
            logger.error(f"Chat model call failed: {e}")
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
            if "All " in rendered and "steps are finished" in rendered:
                return "That was the last step. Enjoy your meal!"
            if "No recipe chosen yet" in rendered:
                return "Which dish are you making? Tell me and we'll go step by step."

        if context.get('social'):
            return "Happy to help. Say next when you're ready for the next step."

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
