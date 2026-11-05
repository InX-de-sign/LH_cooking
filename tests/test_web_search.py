"""Responses request contract and honest failure handling; no live API calls."""
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chatbot"))
with patch("dotenv.load_dotenv"):
    from config import AzureOpenAIConfig
from enhanced_rag_openai import EnhancedRAGWithOpenAI
from web_search import BuiltinWebSearch, SearchResult, SearchUnavailable, default_responses_base, should_search


def answer():
    return {"status": "completed", "output": [
        {"type": "web_search_call", "status": "completed"},
        {"type": "message", "content": [{"type": "output_text", "text": "Cook tofu. \ue200cite\ue202turn0search0\ue201",
            "annotations": [
                {"type": "url_citation", "url": "https://recipes.example/tofu", "title": "Tofu recipe"},
                {"type": "url_citation", "url": "https://recipes.example/tofu", "title": "Duplicate"},
                {"type": "url_citation", "url": "javascript:alert(1)", "title": "Unsafe"},
                {"type": "url_citation", "url": "https://user:secret@example.com/", "title": "Credentials"},
            ]}]}]}


class WebSearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.requests = []
        self.payload = answer()
        self.status = 200
        self.client_class = httpx.AsyncClient
        self.config_patch = patch("web_search.load_azure_openai_config", return_value=AzureOpenAIConfig(api_key="test-only"))
        self.env_patch = patch.dict(os.environ, {"WEB_SEARCH_ENABLED": "true", "AZURE_RESPONSES_BASE_URL": "",
                                                 "WEB_SEARCH_TIMEOUT_SECONDS": "60"})
        self.config_patch.start()
        self.env_patch.start()
        self.addCleanup(self.config_patch.stop)
        self.addCleanup(self.env_patch.stop)
        self.search = BuiltinWebSearch()

    def handle(self, request):
        self.requests.append(request)
        return httpx.Response(self.status, json=self.payload)

    def transport(self):
        return patch("web_search.httpx.AsyncClient", side_effect=lambda **kwargs:
                     self.client_class(transport=httpx.MockTransport(self.handle), **kwargs))

    async def test_required_tool_reasoning_context_and_safe_sources(self):
        with self.transport():
            result = await self.search.search("Find a tofu recipe online", context="Allergies: peanuts")
        request = self.requests[0]
        self.assertEqual(str(request.url), "https://hkust.azure-api.net/hkust-genai/v1/responses")
        self.assertEqual(request.headers["api-key"], "test-only")
        body = json.loads(request.content)
        self.assertEqual(body["tools"], [{"type": "web_search"}])
        self.assertEqual(body["tool_choice"], "required")
        self.assertEqual(body["reasoning"], {"effort": "low"})
        self.assertFalse(body["store"])
        self.assertIn("Allergies: peanuts", body["input"])
        self.assertEqual(result.text, "Cook tofu.")
        self.assertEqual(result.sources, [{"url": "https://recipes.example/tofu", "title": "Tofu recipe"}])
        self.assertEqual(self.search.describe()["last_status"], "verified")

    async def test_plain_model_text_is_not_an_online_search(self):
        self.payload["output"].pop(0)
        with self.transport(), self.assertRaisesRegex(SearchUnavailable, "without a completed web search"):
            await self.search.search("Find a recipe")

    async def test_citations_and_completion_are_required(self):
        cases = [answer(), answer()]
        cases[0]["output"][1]["content"][0]["annotations"] = []
        cases[1]["status"] = "incomplete"
        for payload in cases:
            self.payload = payload
            with self.transport(), self.assertRaises(SearchUnavailable):
                await self.search.search("Find a recipe", force=True)

    async def test_gateway_failure_is_redacted_and_cached(self):
        self.status = 404
        self.payload = {"error": "provider may echo test-only secret request data"}
        with self.transport():
            for _ in range(2):
                with self.assertRaisesRegex(SearchUnavailable, "HTTP 404") as error:
                    await self.search.search("Find a recipe")
                self.assertNotIn("test-only", str(error.exception))
            self.assertEqual(len(self.requests), 1)
            with self.assertRaises(SearchUnavailable):
                await self.search.search("Find a recipe", force=True)
            self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.search.describe()["last_status"], "unavailable")

    async def test_missing_key_disabled_search_and_invalid_url_do_not_send(self):
        self.search.enabled = False
        with self.assertRaisesRegex(SearchUnavailable, "disabled"):
            await self.search.search("Find a recipe")
        self.search.enabled = True
        self.search.chat_config.api_key = None
        with self.assertRaisesRegex(SearchUnavailable, "API key"):
            await self.search.search("Find a recipe")
        self.search.chat_config.api_key = "test-only"
        for base in ("http://example.com/v1", "https://user:key@example.com/v1", "https://example.com/chat"):
            with self.assertRaisesRegex(SearchUnavailable, "HTTPS Responses base URL"):
                await self.search.search("Find a recipe", base_url=base)

    async def test_malformed_payload_is_unavailable(self):
        for payload in ([], {"output": "invalid"}, {"output": [None]}):
            self.payload = payload
            with self.transport(), self.assertRaises(SearchUnavailable):
                await self.search.search("Find a recipe", force=True)

    async def test_search_reply_carries_sources_and_failure_does_not_fall_back(self):
        model = EnhancedRAGWithOpenAI.__new__(EnhancedRAGWithOpenAI)
        model.web_search = SimpleNamespace(search=AsyncMock(return_value=SearchResult("Cook tofu.", [{"url": "https://example.com"}])))
        context = {"prompt_context": "Allergies: peanuts"}
        reply = await model.process_query_with_openai("Search online for tofu", context, "test")
        self.assertEqual(reply, "Cook tofu.")
        self.assertEqual(context["sources"], [{"url": "https://example.com"}])
        model.web_search.search.side_effect = SearchUnavailable("HTTP 404")
        reply = await model.process_query_with_openai("Search online for tofu", context, "test")
        self.assertIn("Online search is unavailable", reply)
        self.assertEqual(context["sources"], [])

    def test_search_routing_preserves_steps_and_immediate_safety(self):
        self.assertFalse(should_search("next", {"query_type": "step"}))
        self.assertFalse(should_search("What now? The pan is on fire", {"query_type": "safety", "missing_notes": True}))
        self.assertFalse(should_search("thanks", {"social": True}))
        self.assertTrue(should_search("Search online for safe ingredient storage", {"query_type": "safety"}))
        self.assertTrue(should_search("How do I cook tofu?", {"missing_notes": True}))
        self.assertFalse(should_search("How do I cook tofu?", {"missing_notes": False}))

    def test_documented_base_urls(self):
        self.assertEqual(default_responses_base("https://hkust.azure-api.net/"), "https://hkust.azure-api.net/hkust-genai/v1")
        self.assertEqual(default_responses_base("https://example.openai.azure.com/"), "https://example.openai.azure.com/openai/v1")
        self.assertEqual(default_responses_base("https://example.com/custom/v1"), "https://example.com/custom/v1")


if __name__ == "__main__":
    unittest.main()
