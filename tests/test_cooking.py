"""Offline regression checks: python -m unittest discover -s tests -v"""
import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chatbot"))
TEMP = tempfile.TemporaryDirectory()
os.environ["MEMORY_DB_PATH"] = str(Path(TEMP.name) / "memory.db")
os.environ["KNOWLEDGE_DB_PATH"] = str(Path(TEMP.name) / "knowledge.db")
os.environ["VLM_PROVIDER"] = "none"
os.environ.pop("OPENAI_API_KEY", None)
# Never load a developer's real credentials into offline tests.
with patch("dotenv.load_dotenv"):
    import config
from config import AzureOpenAIConfig
from enhanced_rag_openai import EnhancedRAGWithOpenAI
from context_builder import detect_navigation
from fastapi.testclient import TestClient
import cook_api
from vision import AzureVision, VisionConfig, VisionProvider, VisionRequest, Observation, decode_image, ImageError


def tearDownModule():
    cook_api.assistant.knowledge.close()
    cook_api.assistant.store.close()
    TEMP.cleanup()


class ModelTests(unittest.IsolatedAsyncioTestCase):
    def test_hkust_client_uses_genai_route_and_header(self):
        from api_client import create_async_client
        cfg = AzureOpenAIConfig(api_key="test-only")
        with patch("api_client.AsyncOpenAI") as constructor:
            create_async_client(cfg, timeout=10, max_retries=0)
        self.assertEqual(constructor.call_args.kwargs["base_url"],
                         "https://hkust.azure-api.net/hkust-genai/v1")
        self.assertEqual(constructor.call_args.kwargs["default_headers"],
                         {"api-key": "test-only"})

    def test_direct_azure_resource_keeps_azure_client(self):
        from api_client import create_async_client
        cfg = AzureOpenAIConfig(api_key="test-only", azure_endpoint="https://example.openai.azure.com")
        with patch("api_client.AsyncAzureOpenAI") as constructor:
            create_async_client(cfg, timeout=10, max_retries=0)
        self.assertEqual(constructor.call_args.kwargs["azure_endpoint"], cfg.azure_endpoint)

    async def test_real_sdk_stream_parsing_and_hkust_request(self):
        import httpx
        from openai import AsyncOpenAI
        from streaming_openai import StreamingOpenAI

        def handle(request):
            self.assertIn('/hkust-genai/v1/chat/completions', str(request.url))
            self.assertEqual(request.headers['api-key'], 'test-only')
            body = json.loads(request.content)
            self.assertEqual(body['model'], 'gpt-5-mini')
            self.assertTrue(body['stream'])
            self.assertIn('max_completion_tokens', body)
            events = [
                {"choices": []},
                {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
                {"choices": [{"index": 0, "delta": {"content": "Stir. "}}]},
                {"choices": [{"index": 0, "delta": {"content": "Serve."}}]},
            ]
            content = ''.join('data: ' + json.dumps(event) + '\n\n' for event in events)
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content=content + 'data: [DONE]\n\n')

        async with AsyncOpenAI(
            api_key='test-only', base_url='https://hkust.azure-api.net/hkust-genai/v1',
            default_headers={'api-key': 'test-only'},
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        ) as client:
            with patch('streaming_openai.create_async_client', return_value=client):
                streamer = StreamingOpenAI()
                chunks = [chunk async for chunk in streamer.stream_response('help', {})]
            self.assertEqual(chunks, ['Stir. ', 'Serve.'])

    def model(self, tokens, fail=False):
        model = EnhancedRAGWithOpenAI()
        model.openai_client = object()

        async def stream(*args):
            for token in tokens:
                yield token
            if fail:
                raise RuntimeError("simulated network failure")
        model.streaming_openai = SimpleNamespace(stream_response=stream)
        return model

    async def test_stream_delivers_before_completion(self):
        deltas = []
        model = self.model(["First sentence. ", "Second sentence."])

        async def receive(delta):
            deltas.append(delta)
        reply = await model.process_query_with_openai("help", {}, "test", on_delta=receive)
        self.assertEqual(reply, "".join(deltas))
        self.assertEqual(len(deltas), 2)

    async def test_partial_failure_keeps_delivered_words(self):
        deltas = []

        async def receive(delta):
            deltas.append(delta)
        reply = await self.model(["Turn off the heat."], fail=True).process_query_with_openai(
            "help", {}, "test", on_delta=receive)
        self.assertEqual(reply, "".join(deltas))
        self.assertIn("interrupted", reply)

    async def test_empty_and_failed_streams_use_notes(self):
        context = {"query_type": "step", "prompt_context": "Current step 1: Test step."}
        for fail in (False, True):
            reply = await self.model([], fail=fail).process_query_with_openai(
                "next", context, "test", on_delta=AsyncMock())
            self.assertIn("Test step", reply)

    async def test_azure_vision_shares_key_and_supports_reasoning_parameters(self):
        chat_config = AzureOpenAIConfig(api_key="test-only", deployment_name="gpt-5-mini")
        complete = AsyncMock(return_value=SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content='{"summary":"a tomato"}'))]))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
        with patch("config.load_azure_openai_config", return_value=chat_config), \
                patch("api_client.create_async_client", return_value=client) as constructor:
            provider = AzureVision(VisionConfig(provider="azure", model="gpt-5-mini"))
            result = await provider.observe(b"test-image", "image/jpeg", VisionRequest("is this done?"))
        self.assertEqual(result.summary, "a tomato")
        self.assertEqual(constructor.call_args.args[0].api_key, "test-only")
        kwargs = complete.call_args.kwargs
        self.assertIn("max_completion_tokens", kwargs)
        self.assertNotIn("max_tokens", kwargs)
        self.assertEqual(kwargs["messages"][1]["content"][1]["type"], "image_url")

    async def test_vision_reports_auth_failure_without_exposing_credentials(self):
        class FailingVision(VisionProvider):
            name = "azure"

            @property
            def available(self):
                return True

            async def _complete(self, image, mime, request):
                error = RuntimeError("secret-key-must-not-appear")
                error.status_code = 401
                raise error

        provider = FailingVision(VisionConfig(provider="azure"))
        self.assertIsNone(await provider.observe(b"test", "image/jpeg", VisionRequest("look")))
        self.assertIn("HTTP 401", provider.last_error)
        self.assertNotIn("secret-key-must-not-appear", provider.last_error)


class AppTests(unittest.TestCase):
    def test_page_chat_ids_start_fresh_keep_profile_and_resume_only_same_chat(self):
        with TestClient(cook_api.app) as client:
            with client.websocket_connect("/ws?client_id=new-chat-regression&chat_id=first") as first:
                first.receive_json()
                for answer in ("Sam", "beginner", "Peanuts", "garlic pasta", "start"):
                    first.send_json({"type": "ask", "message": answer})
                    first.receive_json()
                    final = first.receive_json()
                self.assertEqual(final["cooking"]["step"], 0)
                with client.websocket_connect("/ws?client_id=new-chat-regression&chat_id=second") as second:
                    fresh = second.receive_json()
                    self.assertEqual(fresh["profile"]["step"], "dish")
                    self.assertIsNone(fresh["cooking"]["recipe"])
                    self.assertIsNone(fresh["cooking"]["step"])
                    second.send_json({"type": "profile_get", "request_id": "profile"})
                    self.assertEqual(second.receive_json()["profile"]["dietary"], "Peanuts")
                    first.send_json({"type": "ask", "message": "Next"})
                    first.receive_json()
                    self.assertEqual(first.receive_json()["cooking"]["step"], 1, 'another tab cannot switch this chat')
                    second.send_json({"type": "ask", "message": "tomato eggs"})
                    second.receive_json()
                    self.assertTrue(second.receive_json()["profile"]["complete"])
                    second.send_json({"type": "history_list", "request_id": "history"})
                    self.assertEqual(len(second.receive_json()["sessions"]), 2)
            with client.websocket_connect("/ws?client_id=new-chat-regression&chat_id=first") as first:
                self.assertEqual(first.receive_json()["cooking"]["step"], 1, 'same-page reconnect retains progress')
            with client.websocket_connect("/ws?client_id=new-chat-regression&chat_id=reopened") as reopened:
                self.assertEqual(reopened.receive_json()["profile"]["step"], "dish")

    def test_profile_and_history_websocket_without_chat_or_api_calls(self):
        with TestClient(cook_api.app) as client:
            with client.websocket_connect("/ws?client_id=settings-regression") as ws:
                ws.receive_json()
                ws.send_json({"type": "profile_save", "request_id": "save", "profile": {
                    "cook_name": "Alex", "skill_level": "beginner", "dietary": "Peanut allergy; vegetarian"}})
                saved = ws.receive_json()
                self.assertEqual(saved["type"], "settings_result")
                self.assertEqual(saved["profile"]["cook_name"], "Alex")
                self.assertEqual(saved["onboarding"]["step"], "dish")
                ws.send_json({"type": "ask", "message": "garlic pasta"})
                ws.receive_json()
                self.assertTrue(ws.receive_json()["profile"]["complete"])
                with patch.object(cook_api.assistant.model, "process_query_with_openai",
                                  AsyncMock(return_value="Use a peanut-free option.")) as model:
                    ws.send_json({"type": "ask", "message": "What ingredients do I need?"})
                    ws.receive_json()
                    ws.receive_json()
                    self.assertIn("Peanut allergy; vegetarian", model.call_args.kwargs["context"]["prompt_context"])
                    ws.send_json({"type": "profile_get", "request_id": "get"})
                    self.assertEqual(ws.receive_json()["profile"]["dietary"], "Peanut allergy; vegetarian")
                    ws.send_json({"type": "history_list", "request_id": "list"})
                    history = ws.receive_json()
                    self.assertEqual(len(history["sessions"]), 1)
                    session = history["sessions"][0]["session_id"]
                    ws.send_json({"type": "history_detail", "request_id": "detail", "session_id": session})
                    detail = ws.receive_json()
                    self.assertEqual(detail["turns"][0]["assistant_text"], "Use a peanut-free option.")
                    self.assertEqual(model.await_count, 1)
                ws.send_json({"type": "profile_save", "request_id": "invalid", "profile": {"dietary": []}})
                self.assertIn("error", ws.receive_json())
                ws.send_json({"type": "ping"})
                self.assertEqual(ws.receive_json()["type"], "pong")
            with client.websocket_connect("/ws?client_id=another-browser") as ws:
                ws.receive_json()
                ws.send_json({"type": "history_detail", "request_id": "denied", "session_id": session})
                self.assertEqual(ws.receive_json()["error"], "History entry not found.")

    def test_uploaded_photo_is_analyzed_for_ordinary_prompt_and_not_reused(self):
        provider = SimpleNamespace(available=True, config=VisionConfig(), observe=AsyncMock(
            return_value=Observation(summary="a tomato", items=["tomato"])))
        with patch.object(cook_api.assistant, "vision", provider), TestClient(cook_api.app) as client:
            with client.websocket_connect("/ws?client_id=photo-regression") as ws:
                ws.receive_json()
                for answer in ("Sam", "beginner", "none", "garlic pasta"):
                    ws.send_json({"type": "ask", "message": answer})
                    ws.receive_json()
                    ws.receive_json()
                with patch.object(cook_api.assistant.model, "process_query_with_openai",
                                  AsyncMock(return_value="I can see a tomato.")) as model:
                    ws.send_json({"type": "ask", "message": "List the ingredients", "image_source": "upload",
                                  "image": "data:image/jpeg;base64,YWJj"})
                    self.assertEqual(ws.receive_json()["type"], "thinking")
                    result = ws.receive_json()
                    self.assertEqual(result["vision"]["summary"], "a tomato")
                    self.assertIn("a tomato", model.call_args.kwargs["context"]["prompt_context"])
                    self.assertEqual(provider.observe.call_args.args[0], b"abc")
                    self.assertEqual(provider.observe.await_count, 1)
                    ws.send_json({"type": "ask", "message": "Next"})
                    ws.receive_json()
                    self.assertIsNone(ws.receive_json()["vision"])
                    self.assertEqual(provider.observe.await_count, 1)
                    provider.observe.return_value = None
                    ws.send_json({"type": "ask", "message": "List the ingredients", "image_source": "upload",
                                  "image": "data:image/jpeg;base64,YWJj"})
                    ws.receive_json()
                    self.assertIn("couldn't analyze", ws.receive_json()["message"])
                    self.assertEqual(model.await_count, 2, 'failed photo analysis does not produce a model-only answer')
                    ws.send_json({"type": "ask", "message": "Analyze", "image_source": "upload",
                                  "image": "data:image/jpeg;base64,!!"})
                    ws.receive_json()
                    failed = ws.receive_json()
                    self.assertIn("could not be attached", failed["message"])
                    self.assertIn("image_error", failed)
                    self.assertEqual(provider.observe.await_count, 2)

    def test_onboarding_navigation_and_streaming_websocket(self):
        with TestClient(cook_api.app) as client:
            with client.websocket_connect("/ws?client_id=offline-regression") as ws:
                self.assertEqual(ws.receive_json()["type"], "system")
                for answer in ("Sam", "beginner", "none", "garlic pasta"):
                    ws.send_json({"type": "ask", "message": answer})
                    self.assertEqual(ws.receive_json()["type"], "thinking")
                    response = ws.receive_json()
                    self.assertEqual(response["type"], "response")
                self.assertTrue(response["profile"]["complete"])
                ws.send_json({"type": "ask", "message": "Let's start"})
                ws.receive_json()
                response = ws.receive_json()
                self.assertEqual(response["cooking"]["step"], 0)
                self.assertIn("boil", response["message"].lower())

                async def fake_reply(query, context, user_id, on_delta=None):
                    await on_delta("One. ")
                    await on_delta("Two.")
                    return "One. Two."
                with patch.object(cook_api.assistant.model, "process_query_with_openai", fake_reply):
                    ws.send_json({"type": "ask", "message": "Next"})
                    self.assertEqual(ws.receive_json()["type"], "thinking")
                    self.assertEqual(ws.receive_json(), {"type": "delta", "text": "One. "})
                    self.assertEqual(ws.receive_json(), {"type": "delta", "text": "Two."})
                    final = ws.receive_json()
                    self.assertEqual(final["message"], "One. Two.")
                    self.assertEqual(final["cooking"]["step"], 1)
            # Session state survives a reconnect without duplicate step movement.
            with client.websocket_connect("/ws?client_id=offline-regression") as ws:
                self.assertEqual(ws.receive_json()["cooking"]["step"], 1)
                from web_search import SearchResult
                sources = [{"url": "https://recipes.example/tofu", "title": "Tofu recipe"}]
                with patch.object(cook_api.assistant.model.web_search, "search",
                                  AsyncMock(return_value=SearchResult("Cook tofu.", sources))):
                    ws.send_json({"type": "ask", "message": "Search online for my next recipe"})
                    self.assertEqual(ws.receive_json()["type"], "thinking")
                    result = ws.receive_json()
                    self.assertEqual(result["sources"], sources)
                    self.assertEqual(result["cooking"]["step"], 1)

    def test_health_recipes_and_safe_chicken_retrieval(self):
        with TestClient(cook_api.app) as client:
            self.assertFalse(client.get("/health").json()["model"]["configured"])
            recipes = client.get("/api/recipes").json()["recipes"]
            self.assertEqual(len(recipes), 50)
            import yaml
            source_recipes = yaml.safe_load((ROOT / "chatbot" / "recipes.yaml").read_text(encoding="utf-8"))
            for source in source_recipes:
                with self.subTest(recipe=source["name"]):
                    recipe = next(r for r in recipes if r["name"] == source["name"])
                    self.assertEqual(recipe["ingredients"], source["ingredients"])
                    self.assertTrue(recipe["ingredients"])
                    self.assertGreater(recipe["steps"], 0)
                    self.assertGreater(recipe["serves"], 0)
                    self.assertGreater(recipe["time_minutes"], 0)
                    for alias in [source["name"], *source.get("aliases", [])]:
                        resolved = cook_api.assistant.knowledge.resolve_recipe(alias)
                        self.assertIsNotNone(resolved, alias)
                        self.assertEqual(resolved.id, recipe["id"], alias)
            shrimp = next(r for r in recipes if r["name"] == "Spicy Shrimp Glass Noodles")
            self.assertEqual(shrimp["steps"], 5)
            self.assertEqual(cook_api.assistant.knowledge.resolve_recipe("chinese fried spicy rice noodle and shrimp").id, shrimp["id"])
            ingredients = cook_api.assistant.knowledge.ingredients_text(shrimp["id"])
            self.assertIn("glass noodles", ingredients)
            self.assertIn("shrimp", ingredients)
            reply = client.get("/api/retrieve", params={"q": "is chicken cooked through?"}).json()
            texts = " ".join(f["text"] for f in reply["facts"])
            self.assertIn("74 degrees Celsius", texts)
            self.assertNotIn("juices run clear", texts)
            self.assertEqual(client.post("/api/vision/test", json={}).status_code, 503)

    def test_recipe_selection_previews_ingredients_before_start_and_on_switch(self):
        with TestClient(cook_api.app) as client:
            with client.websocket_connect("/ws?client_id=ingredients-preview") as ws:
                ws.receive_json()
                for answer in ("Sam", "beginner", "none", "garlic pasta"):
                    ws.send_json({"type": "ask", "message": answer})
                    ws.receive_json()
                    result = ws.receive_json()
                self.assertIsNone(result["cooking"]["step"])
                self.assertEqual(result["cooking"]["serves"], 2)
                self.assertEqual(result["cooking"]["time_minutes"], 20)
                for ingredient in result["cooking"]["ingredients"]:
                    self.assertIn(ingredient, result["message"])
                ws.send_json({"type": "ask", "message": "start"})
                ws.receive_json()
                self.assertEqual(ws.receive_json()["cooking"]["step"], 0)

                for command, recipe_name in (("start chicken curry", "Chicken Curry"),
                                              ("switch to pad thai", "Shrimp Pad Thai"),
                                              ("Let's make banana bread", "Banana Bread")):
                    with self.subTest(command=command), patch.object(
                            cook_api.assistant.model, "process_query_with_openai", AsyncMock()) as model:
                        ws.send_json({"type": "ask", "message": command})
                        ws.receive_json()
                        result = ws.receive_json()
                        self.assertEqual(result["cooking"]["recipe"], recipe_name)
                        self.assertIsNone(result["cooking"]["step"])
                        self.assertIsNone(result["cooking"]["step_text"])
                        self.assertTrue(result["cooking"]["ingredients"])
                        for ingredient in result["cooking"]["ingredients"]:
                            self.assertIn(ingredient, result["message"])
                        model.assert_not_awaited()
            with client.websocket_connect("/ws?client_id=ingredients-preview") as ws:
                resumed = ws.receive_json()
                self.assertIsNone(resumed["cooking"]["step"])
                self.assertIn("3 medium ripe bananas", resumed["message"])
                ws.send_json({"type": "ask", "message": "start"})
                ws.receive_json()
                started = ws.receive_json()
                self.assertEqual(started["cooking"]["step"], 0)
                self.assertEqual(started["cooking"]["ingredients"], resumed["cooking"]["ingredients"])

    def test_navigation_questions_do_not_advance(self):
        self.assertIsNone(detect_navigation("is it done?"))
        self.assertIsNone(detect_navigation("what do I do next time?"))
        self.assertIsNone(detect_navigation("Search online for my next recipe"))
        self.assertEqual(detect_navigation("next"), ("next",))
        self.assertEqual(detect_navigation("start"), ("start",))
        self.assertEqual(detect_navigation("back"), ("previous",))
        self.assertEqual(detect_navigation("step 4"), ("goto", 4))

    def test_image_validation(self):
        for image in ("invalid", "data:image/gif;base64,YQ==", "data:image/jpeg;base64,!!"):
            with self.assertRaises(ImageError):
                decode_image(image, 100)
        with self.assertRaises(ImageError):
            decode_image("data:image/jpeg;base64,YWI=", 1)

    def test_completion_parameters_and_placeholder_keys(self):
        self.assertIn("max_completion_tokens", AzureOpenAIConfig().completion_kwargs())
        self.assertIn("max_tokens", AzureOpenAIConfig(deployment_name="gpt-4o-mini").completion_kwargs())
        with patch.dict(os.environ, {"OPENAI_API_KEY": "replace_with_key"}):
            config._drop_placeholders()
            self.assertNotIn("OPENAI_API_KEY", os.environ)


class RecipeSwitchTests(unittest.IsolatedAsyncioTestCase):
    """Mid-cook dish changes must update visit state, not only the chat reply."""

    @classmethod
    def setUpClass(cls):
        from knowledge_base import KnowledgeBase, build_knowledge_db
        build_knowledge_db(os.environ["KNOWLEDGE_DB_PATH"])
        cook_api.assistant.knowledge.close()
        kb = KnowledgeBase(os.environ["KNOWLEDGE_DB_PATH"], auto_build=False)
        cook_api.assistant.knowledge = kb
        cook_api.assistant.context.knowledge = kb

    def _pasta_visit(self, user_id: str):
        import time
        store = cook_api.assistant.store
        visit = store.current_visit(user_id)
        store._update_visit(
            visit.session_id,
            onboarding_step="complete",
            onboarding_started_at=time.time(),
            onboarding_completed_at=time.time(),
            cook_name="Sam",
            skill_level="beginner",
            dietary="none",
        )
        visit = store.current_visit(user_id)
        pasta = cook_api.assistant.knowledge.resolve_recipe("broccoli garlic pasta")
        self.assertIsNotNone(pasta)
        store.set_recipe(visit, pasta.id, pasta.name)
        store.set_step(visit, 0)
        return store.current_visit(user_id), pasta

    def test_change_dish_to_stir_fry_shrimp_switches_recipe(self):
        user = "switch-shrimp"
        visit, pasta = self._pasta_visit(user)
        shrimp = cook_api.assistant.knowledge.resolve_recipe("stir-fry shrimp")
        self.assertIsNotNone(shrimp)
        self.assertEqual(shrimp.name, "Garlic Butter Shrimp")

        turn = cook_api.assistant.context.build(
            user, "i want to change the dish to make to stir-fry shrimp")
        self.assertEqual(turn.recipe.id, shrimp.id)
        self.assertNotEqual(turn.recipe.id, pasta.id)
        self.assertIsNone(turn.visit.step)

    def test_change_recipe_without_name_clears_active_dish(self):
        user = "switch-clear"
        self._pasta_visit(user)
        self.assertEqual(detect_navigation("change recipe"), ("stop",))
        turn = cook_api.assistant.context.build(user, "change recipe")
        self.assertIsNone(turn.recipe)
        self.assertIsNone(turn.visit.recipe_id)

    def test_question_about_another_dish_does_not_steal_session(self):
        user = "switch-keep"
        visit, pasta = self._pasta_visit(user)
        turn = cook_api.assistant.context.build(user, "is broccoli spicy?")
        self.assertEqual(turn.recipe.id, pasta.id)
        self.assertEqual(turn.visit.recipe_id, pasta.id)

    def test_the_one_you_mentioned_uses_recent_chat(self):
        user = "switch-anaphora"
        visit, pasta = self._pasta_visit(user)
        cook_api.assistant.store.record_turn(
            visit, None,
            user_text="maybe shrimp?",
            assistant_text="We can switch to a simple garlic-butter shrimp recipe.",
            intent="help",
        )
        turn = cook_api.assistant.context.build(user, "the one u mentioned")
        self.assertIsNotNone(turn.recipe)
        self.assertEqual(turn.recipe.name, "Garlic Butter Shrimp")
        self.assertNotEqual(turn.recipe.id, pasta.id)

    async def test_process_message_returns_preview_after_switch(self):
        user = "switch-preview"
        self._pasta_visit(user)
        result = await cook_api.assistant.process_message(
            "i want to change the dish to stir-fry shrimp", user_id=user)
        self.assertEqual(result["cooking"]["recipe"], "Garlic Butter Shrimp")
        self.assertIn("shrimp", result["message"].lower())


if __name__ == "__main__":
    unittest.main()
