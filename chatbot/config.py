# config.py - Configuration for the chat model
#
# The VLM is configured separately, in vision.py, so it can be swapped
# without touching the chat model.
import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Settings come from ONE file: chatbot/.env (copy chatbot/.env.example). It
# holds the two slots -- the chat model and the VLM -- and nothing else.
#
# The repository-root .env is deliberately NOT read. In this fork it is a
# leftover from PAL_OP holding an expired key, and loading it as a fallback
# meant an empty slot silently picked up that dead key instead of running in
# notes-only mode.
#
# A value already exported in the real environment (docker-compose, the
# Lighthouse .env.lighthouse) outranks the file. Placeholder and empty values
# are treated as unset.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PLACEHOLDER_PREFIXES = ("your_", "replace_with_", "<")


_SECRET_NAMES = ("OPENAI_API_KEY", "VLM_API_KEY")


def _drop_placeholders():
    """Unset any secret still holding a .env.example placeholder.

    Done between files, not after both: a placeholder left in place would
    satisfy load_dotenv's override=False and block the real value in the next
    file from ever being read.
    """
    # An empty value counts as unset too: docker-compose exports `KEY=` for a
    # variable it could not fill, and that would block the .env file the same way.
    for name in _SECRET_NAMES:
        value = os.environ.get(name)
        if value is not None and (not value.strip() or value.startswith(_PLACEHOLDER_PREFIXES)):
            del os.environ[name]


def _load_env_files():
    _drop_placeholders()
    path = os.path.join(_HERE, ".env")
    if os.path.exists(path):
        load_dotenv(path, override=False)
        _drop_placeholders()


_load_env_files()

@dataclass
class AzureOpenAIConfig:
    """Configuration for Azure OpenAI (HKUST) integration"""
    
    # API Configuration
    api_key: str = None
    azure_endpoint: str = "https://hkust.azure-api.net"
    api_version: str = "2024-10-21"
    deployment_name: str = "gpt-5-mini"

    # Completion budget. On gpt-5 this is `max_completion_tokens` and it must
    # also cover the model's hidden reasoning tokens -- a budget sized for the
    # visible answer alone comes back as an EMPTY string with
    # finish_reason="length". See completion_kwargs() below.
    max_tokens: int = 400
    temperature: float = 1.0

    # gpt-5 only. "minimal" spends zero reasoning tokens, which is what a
    # real-time voice bot wants: measured against the HKUST proxy it answers a
    # one-sentence question in ~1.4 s versus ~2.1 s for gpt-4o-mini.
    reasoning_effort: str = "minimal"
    verbosity: str = "low"

    @property
    def configured(self) -> bool:
        """False until someone fills in the chat-model slot in chatbot/.env."""
        return bool(self.api_key and self.azure_endpoint and self.deployment_name)

    @property
    def is_reasoning_model(self) -> bool:
        """True for gpt-5 / o-series deployments, which take a different param set."""
        name = (self.deployment_name or "").lower()
        return name.startswith(("gpt-5", "o1", "o3", "o4"))

    def completion_kwargs(self, max_output_tokens: int = None) -> dict:
        """
        Build the model-specific half of a chat.completions.create() call.

        gpt-5 differs from gpt-4o in three ways that all fail loudly or, worse,
        silently:
          - `max_tokens` is rejected as a parameter name; it is
            `max_completion_tokens`, and it is a budget for reasoning +
            visible output, not visible output alone.
          - `temperature` must be 1 (anything else is a 400 from the proxy).
          - `reasoning_effort` / `verbosity` are accepted and are the only
            useful latency knobs.
        """
        budget = max_output_tokens or self.max_tokens

        if not self.is_reasoning_model:
            return {"max_tokens": budget, "temperature": self.temperature}

        kwargs = {
            "max_completion_tokens": max(budget, MIN_REASONING_BUDGET),
            "temperature": 1,
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.verbosity:
            kwargs["verbosity"] = self.verbosity
        return kwargs


# Floor for gpt-5's completion budget. Even at reasoning_effort="minimal" the
# model needs headroom above the visible answer; below roughly this figure
# short-answer prompts start returning "" instead of a sentence.
MIN_REASONING_BUDGET = 300


def load_azure_openai_config():
    """Load Azure OpenAI configuration"""
    
    config = AzureOpenAIConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://hkust.azure-api.net"),
        # Direct Azure resources use this version. The HKUST GenAI v1 route
        # does not take an api-version query parameter.
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-5-mini"),
        max_tokens=int(os.getenv("AZURE_MAX_COMPLETION_TOKENS", "400")),
        reasoning_effort=os.getenv("AZURE_REASONING_EFFORT", "minimal"),
        verbosity=os.getenv("AZURE_VERBOSITY", "low"),
    )
    
    # No key is a supported state, not an error: the coach then runs on its
    # recipe notes alone, so the app can be tried before a model is chosen.
    return config

def load_openai_config():
    """Legacy function name - redirects to load_azure_openai_config"""
    return load_azure_openai_config()
