# streaming_openai.py - Prompt assembly and streaming calls for the cooking coach
import logging
import os
from typing import AsyncGenerator, Dict, Any

from api_client import create_async_client
from config import load_azure_openai_config

logger = logging.getLogger(__name__)

DEFAULT_PERSONA = "help"


class StreamingOpenAI:
    """Streaming HKUST GenAI or Azure client for real-time text generation."""

    def __init__(self):
        self.config = load_azure_openai_config()
        self.client = create_async_client(
            self.config,
            timeout=float(os.getenv("AZURE_OPENAI_TIMEOUT_SECONDS", "30")),
            max_retries=1,
        )

        # The user prompt arrives as labelled layers -- [COOK], [RECIPE],
        # [NOW], [RECENT], [FACTS] -- built by context_builder. Every persona
        # shares the same grounding contract; only the job differs.
        grounding = """
            GROUNDING - this matters more than style:
                - Quantities, times and temperatures for this recipe come ONLY
                  from [RECIPE] and [FACTS]. Never make one up.
                - If [FACTS] says the notes do not cover the question, you may
                  give brief general cooking advice and say it is general.
                - [NOW] says what they are asking about and what the camera
                  sees. A photo cannot show temperature, taste or whether meat
                  is cooked inside; never claim it does.
                - If [NOW] reports a SAFETY problem, say that first.
                - [RECENT] is what was already said. Do not repeat it."""

        voice = """
            VOICE - every reply is spoken aloud to someone with busy hands:
                - Calm, warm and encouraging, like a friend who cooks well.
                - Plain sentences only. No lists, markdown, emojis or symbols.
                - Say numbers the way you would speak them ("half a teaspoon").
                - Pitch the explanation to [COOK] experience."""

        self.system_prompts = {
            'step': f"""
                You are a voice cooking coach guiding someone through a recipe
                at home. They just asked to move through the steps.
                {grounding}
                {voice}
                JOB:
                    - Read the current step from [RECIPE] completely, keeping
                      every quantity and time.
                    - Then add at most one short sentence: the "done when" cue,
                      or a timer if the step has one.
                    - Never skip ahead or merge steps.""",

            'help': f"""
                You are a voice cooking coach helping someone cook at home.
                {grounding}
                {voice}
                JOB:
                    - Answer the question directly in one or two short
                      sentences, at most 40 words.
                    - If it relates to the current step in [RECIPE], tie the
                      answer back to that step.
                    - Do not end with a question unless you genuinely need an
                      answer to help them.""",

            'safety': f"""
                You are a voice cooking coach, and this is a safety moment.
                {grounding}
                {voice}
                JOB:
                    - Lead with the single most important action, as a short
                      instruction ("Turn off the heat now.").
                    - Then one sentence of why or what next. Nothing else.
                    - For fire that will not go out, or a serious injury,
                      tell them to call the emergency services.""",
        }

    async def stream_response(self, query: str, context: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream the model's reply chunk by chunk."""
        persona = context.get('query_type', DEFAULT_PERSONA)
        system_prompt = self.system_prompts.get(persona, self.system_prompts[DEFAULT_PERSONA])
        user_prompt = self._build_short_prompt(query, context)

        # Brevity is enforced by the system prompt, not the token budget. On
        # gpt-5 the budget also pays for reasoning tokens, so a small one
        # returns an empty string rather than a short one.
        stream = await self.client.chat.completions.create(
            model=self.config.deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            **self.config.completion_kwargs()
        )

        accumulated = ""
        try:
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    accumulated += content
                    yield content
        finally:
            await stream.close()
        logger.info("Streaming completed: %d chars", len(accumulated))

    def _language_instruction(self, context: Dict[str, Any]) -> str:
        """Name the target language outright rather than asking the model to infer it."""
        if context.get("language") == "zh":
            return "Reply in Mandarin, using simplified Chinese characters."
        return "Reply in English only."

    def _build_short_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """The language instruction plus the context_builder layers."""
        parts = [self._language_instruction(context)]
        rendered = context.get("prompt_context")
        if rendered:
            parts.append(rendered)
        else:
            parts.append(f"[QUESTION]\n{query}")
        parts.append("Reply now, following the JOB in your instructions.")
        return "\n\n".join(parts)
