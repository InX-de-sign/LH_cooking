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
from context_builder import detect_navigation, ingredient_core
from fastapi.testclient import TestClient
import cook_api
import cook_main
from cook_main import GREETING_FALLBACK
from recipe_import import RecipeImportError, validate_recipe
from vision import (AzureVision, VisionConfig, VisionProvider, VisionRequest, Observation,
                    decode_image, decode_images, ImageError)

FRAME = "data:image/jpeg;base64,YWJj"   # b"abc"


def tearDownModule():
    cook_api.assistant.knowledge.close()
    cook_api.assistant.store.close()
    TEMP.cleanup()


def open_chat(ws, new=True):
    """Read the connect handshake: state, then a greeting for a new chat."""
    system = ws.receive_json()
    assert system["type"] == "system", system
    if new:
        assert ws.receive_json()["type"] == "thinking"
        greeting = ws.receive_json()
        assert greeting["type"] == "response", greeting
        system["greeting"] = greeting["message"]
    return system


def ask(ws, message, **fields):
    ws.send_json({"type": "ask", "message": message, **fields})
    assert ws.receive_json()["type"] == "thinking"
    return ws.receive_json()


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

    @staticmethod
    def sse(events):
        import httpx
        content = ''.join('data: ' + json.dumps(event) + '\n\n' for event in events)
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=content + 'data: [DONE]\n\n')

    async def streamer_with(self, handle):
        import httpx
        from openai import AsyncOpenAI
        from streaming_openai import StreamingOpenAI
        client = AsyncOpenAI(
            api_key='test-only', base_url='https://hkust.azure-api.net/hkust-genai/v1',
            default_headers={'api-key': 'test-only'},
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        with patch('streaming_openai.create_async_client', return_value=client):
            return StreamingOpenAI(), client

    async def test_real_sdk_stream_parsing_and_hkust_request(self):
        def handle(request):
            self.assertIn('/hkust-genai/v1/chat/completions', str(request.url))
            self.assertEqual(request.headers['api-key'], 'test-only')
            body = json.loads(request.content)
            self.assertEqual(body['model'], 'gpt-5-mini')
            self.assertTrue(body['stream'])
            self.assertIn('max_completion_tokens', body)
            self.assertNotIn('tools', body, 'no tools without a handler')
            return self.sse([
                {"choices": []},
                {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
                {"choices": [{"index": 0, "delta": {"content": "Stir. "}}]},
                {"choices": [{"index": 0, "delta": {"content": "Serve."}}]},
            ])
        streamer, client = await self.streamer_with(handle)
        async with client:
            chunks = [chunk async for chunk in streamer.stream_response('help', {})]
        self.assertEqual(chunks, ['Stir. ', 'Serve.'])

    async def test_tool_call_runs_then_final_reply_without_tools(self):
        """The model picks a recipe through a tool; the result feeds a second, tool-free call."""
        bodies = []

        def handle(request):
            body = json.loads(request.content)
            bodies.append(body)
            if len(bodies) == 1:
                return self.sse([
                    {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                        "function": {"name": "choose_recipe", "arguments": ""}}]}}]},
                    {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0,
                        "function": {"arguments": "{\"recipe_name\": \"Mapo Tofu\"}"}}]}}]},
                ])
            return self.sse([{"choices": [{"index": 0, "delta": {"content": "Mapo Tofu it is."}}]}])

        calls = []

        async def tool_handler(name, args):
            calls.append((name, args))
            return {"ok": True, "recipe": "Mapo Tofu"}

        from streaming_openai import TOOLS
        streamer, client = await self.streamer_with(handle)
        async with client:
            chunks = [c async for c in streamer.stream_response('the first one', {
                "tools": TOOLS, "tool_handler": tool_handler,
                "history": [{"user_text": "ideas?", "assistant_text": "Try Mapo Tofu or Egg Fried Rice."}],
                "images": [(FRAME, "low")]})]
        self.assertEqual(calls, [("choose_recipe", {"recipe_name": "Mapo Tofu"})])
        self.assertEqual("".join(chunks), "Mapo Tofu it is.")
        self.assertIn("tools", bodies[0])
        self.assertNotIn("tools", bodies[1], "only one tool round, then a plain reply")
        roles = [m["role"] for m in bodies[1]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "user", "assistant", "tool"])
        self.assertEqual(json.loads(bodies[1]["messages"][-1]["content"])["recipe"], "Mapo Tofu")
        frame_part = bodies[0]["messages"][3]["content"][1]
        self.assertEqual(frame_part["image_url"], {"url": FRAME, "detail": "low"})
        self.assertIn("2 dishes", bodies[0]["messages"][0]["content"], "the prompt caps suggestions at two")

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
        context = {"query_type": "step", "prompt_context": "Current step 1 of 3: Test step."}
        for fail in (False, True):
            reply = await self.model([], fail=fail).process_query_with_openai(
                "next", context, "test", on_delta=AsyncMock())
            self.assertIn("Test step", reply)

    async def test_azure_vision_shares_key_supports_reasoning_parameters_and_is_inline(self):
        chat_config = AzureOpenAIConfig(api_key="test-only", deployment_name="gpt-5-mini")
        complete = AsyncMock(return_value=SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content='{"summary":"a tomato"}'))]))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
        with patch("config.load_azure_openai_config", return_value=chat_config), \
                patch("api_client.create_async_client", return_value=client) as constructor:
            provider = AzureVision(VisionConfig(provider="azure", model="gpt-5-mini"))
            result = await provider.observe(b"test-image", "image/jpeg", VisionRequest("is this done?"))
            self.assertTrue(provider.inline, "same deployment as chat: frames go straight to the chat call")
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
    def test_new_chat_greets_without_questionnaire_and_reconnect_resumes(self):
        with TestClient(cook_api.app) as client:
            with client.websocket_connect("/ws?client_id=new-chat-regression&chat_id=first") as first:
                hello = open_chat(first)
                self.assertEqual(hello["greeting"], GREETING_FALLBACK)
                self.assertIsNone(hello["cooking"]["recipe"])
                self.assertNotIn("profile", hello, "no setup questionnaire state")
                # Without a model there are no tools, but naming a dish still starts it.
                chosen = ask(first, "garlic pasta")
                self.assertEqual(chosen["cooking"]["recipe"], "Spaghetti Aglio e Olio")
                self.assertIsNone(chosen["cooking"]["step"])
                self.assertEqual(len(chosen["cooking"]["steps"]), chosen["cooking"]["total_steps"])
                self.assertEqual(ask(first, "ok I'm ready")["cooking"]["step"], 0, "ready starts a recipe that hasn't begun")
                self.assertEqual(ask(first, "next")["cooking"]["step"], 1)
                self.assertEqual(ask(first, "ready")["cooking"]["step"], 1, "mid-recipe 'ready' never rewinds")
                self.assertEqual(ask(first, "start")["cooking"]["step"], 0, "start is an explicit restart")
                first.send_json({"type": "profile_save", "request_id": "save", "profile": {
                    "cook_name": "Sam", "skill_level": "beginner", "dietary": "Peanuts"}})
                self.assertEqual(first.receive_json()["profile"]["dietary"], "Peanuts")
                with client.websocket_connect("/ws?client_id=new-chat-regression&chat_id=second") as second:
                    fresh = open_chat(second)
                    self.assertIsNone(fresh["cooking"]["recipe"])
                    second.send_json({"type": "profile_get", "request_id": "profile"})
                    self.assertEqual(second.receive_json()["profile"]["dietary"], "Peanuts")
                    self.assertEqual(ask(first, "Next")["cooking"]["step"], 1, 'another tab cannot switch this chat')
                    self.assertEqual(ask(second, "tomato eggs")["cooking"]["recipe"], "Tomato and Egg Stir-fry")
                    second.send_json({"type": "history_list", "request_id": "history"})
                    self.assertEqual(len(second.receive_json()["sessions"]), 2)
            with client.websocket_connect("/ws?client_id=new-chat-regression&chat_id=first") as first:
                resumed = open_chat(first, new=False)
                self.assertEqual(resumed["cooking"]["step"], 1, 'same-page reconnect retains progress, no second greeting')
                first.send_json({"type": "ping"})
                self.assertEqual(first.receive_json()["type"], "pong")
            with client.websocket_connect("/ws?client_id=new-chat-regression&chat_id=reopened") as reopened:
                self.assertIsNone(open_chat(reopened)["cooking"]["recipe"])

    def test_profile_and_history_websocket(self):
        with TestClient(cook_api.app) as client:
            with client.websocket_connect("/ws?client_id=settings-regression") as ws:
                open_chat(ws)
                ws.send_json({"type": "profile_save", "request_id": "save", "profile": {
                    "cook_name": "Alex", "skill_level": "beginner", "dietary": "Peanut allergy; vegetarian"}})
                saved = ws.receive_json()
                self.assertEqual(saved["type"], "settings_result")
                self.assertEqual(saved["profile"]["cook_name"], "Alex")
                ask(ws, "garlic pasta")
                with patch.object(cook_api.assistant.model, "process_query_with_openai",
                                  AsyncMock(return_value="Use a peanut-free option.")) as model:
                    ask(ws, "What ingredients do I need?")
                    context = model.call_args.kwargs["context"]
                    self.assertIn("Peanut allergy; vegetarian", context["prompt_context"])
                    self.assertIn("1. ", context["prompt_context"], "the model sees every numbered step")
                    self.assertNotIn("[CATALOG]", context["prompt_context"], "no catalog once a dish is chosen")
                    self.assertTrue(context["history"], "recent exchanges go to the model as chat history")
                    ws.send_json({"type": "profile_get", "request_id": "get"})
                    self.assertEqual(ws.receive_json()["profile"]["dietary"], "Peanut allergy; vegetarian")
                    ws.send_json({"type": "history_list", "request_id": "list"})
                    history = ws.receive_json()
                    self.assertEqual(len(history["sessions"]), 1)
                    session = history["sessions"][0]["session_id"]
                    ws.send_json({"type": "history_detail", "request_id": "detail", "session_id": session})
                    detail = ws.receive_json()
                    self.assertEqual(detail["turns"][0]["assistant_text"], GREETING_FALLBACK)
                    self.assertEqual(detail["turns"][-1]["assistant_text"], "Use a peanut-free option.")
                    self.assertEqual(model.await_count, 1)
                ws.send_json({"type": "profile_save", "request_id": "invalid", "profile": {"dietary": []}})
                self.assertIn("error", ws.receive_json())
            with client.websocket_connect("/ws?client_id=another-browser") as ws:
                open_chat(ws)
                ws.send_json({"type": "history_detail", "request_id": "denied", "session_id": session})
                self.assertEqual(ws.receive_json()["error"], "History entry not found.")

    def test_uploaded_photo_with_separate_vlm_is_analyzed_and_not_reused(self):
        provider = SimpleNamespace(available=True, inline=False, config=VisionConfig(), observe=AsyncMock(
            return_value=Observation(summary="a tomato", items=["tomato"])))
        with patch.object(cook_api.assistant, "vision", provider), TestClient(cook_api.app) as client:
            with client.websocket_connect("/ws?client_id=photo-regression") as ws:
                open_chat(ws)
                ask(ws, "garlic pasta")
                with patch.object(cook_api.assistant.model, "process_query_with_openai",
                                  AsyncMock(return_value="I can see a tomato.")) as model:
                    result = ask(ws, "List the ingredients", image_source="upload", image=FRAME)
                    self.assertEqual(result["vision"]["summary"], "a tomato")
                    self.assertIn("a tomato", model.call_args.kwargs["context"]["prompt_context"])
                    self.assertEqual(provider.observe.call_args.args[0], b"abc")
                    self.assertEqual(provider.observe.await_count, 1)
                    self.assertIsNone(ask(ws, "Next")["vision"])
                    self.assertEqual(provider.observe.await_count, 1)
                    provider.observe.return_value = None
                    self.assertIn("couldn't analyze", ask(ws, "List the ingredients",
                                                         image_source="upload", image=FRAME)["message"])
                    self.assertEqual(model.await_count, 2, 'failed photo analysis does not produce a model-only answer')
                    failed = ask(ws, "Analyze", image_source="upload", image="data:image/jpeg;base64,!!")
                    self.assertIn("could not be attached", failed["message"])
                    self.assertIn("image_error", failed)
                    self.assertEqual(provider.observe.await_count, 2)

    def test_inline_vision_sends_camera_frames_to_the_chat_model(self):
        provider = SimpleNamespace(available=True, inline=True, config=VisionConfig(), observe=AsyncMock())
        with patch.object(cook_api.assistant, "vision", provider), TestClient(cook_api.app) as client:
            with client.websocket_connect("/ws?client_id=inline-frames") as ws:
                open_chat(ws)
                ask(ws, "garlic pasta")
                ask(ws, "start")
                with patch.object(cook_api.assistant.model, "process_query_with_openai",
                                  AsyncMock(return_value="Looks right.")) as model:
                    result = ask(ws, "Am I doing this right?", images=[FRAME] * 3)
                    context = model.call_args.kwargs["context"]
                    self.assertEqual([detail for _, detail in context["images"]], ["low"] * 3)
                    self.assertTrue(context["images"][0][0].startswith("data:image/jpeg;base64,"))
                    self.assertIn("3 live camera frames are attached", context["prompt_context"])
                    self.assertEqual(result["vision"]["frames"], 3)
                    ask(ws, "Next", images=[FRAME] * 3)
                    self.assertEqual(model.call_args.kwargs["context"]["images"], [],
                                     "a step command does not spend tokens on frames")
                    ask(ws, "What's in this?", image_source="upload", image=FRAME)
                    self.assertEqual(model.call_args.kwargs["context"]["images"][0][1], "auto")
                    provider.observe.assert_not_awaited()
                ws.send_json({"type": "history_list", "request_id": "h"})
                session = ws.receive_json()["sessions"][0]["session_id"]
                ws.send_json({"type": "history_detail", "request_id": "d", "session_id": session})
                visions = [t["vision"] for t in ws.receive_json()["turns"] if t["vision"]]
                self.assertIn("Looked at 3 camera frames.", visions)

    def test_navigation_and_streaming_websocket(self):
        with TestClient(cook_api.app) as client:
            with client.websocket_connect("/ws?client_id=offline-regression") as ws:
                open_chat(ws)
                ask(ws, "garlic pasta")
                response = ask(ws, "Let's start")
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
                self.assertEqual(open_chat(ws, new=False)["cooking"]["step"], 1)
                from web_search import SearchResult
                sources = [{"url": "https://recipes.example/tofu", "title": "Tofu recipe"}]
                with patch.object(cook_api.assistant.model.web_search, "search",
                                  AsyncMock(return_value=SearchResult("Cook tofu.", sources))):
                    result = ask(ws, "Search online for my next recipe")
                    self.assertEqual(result["sources"], sources)
                    self.assertEqual(result["cooking"]["step"], 1)

    def test_watch_alerts_are_rate_limited_and_step_done_nudges_once(self):
        watch = AsyncMock(return_value={"status": "issue", "message": "The heat looks too high.", "step": 0})
        with patch.object(cook_api.assistant, "watch", watch), TestClient(cook_api.app) as client:
            with client.websocket_connect("/ws?client_id=watch-regression") as ws:
                open_chat(ws)
                ws.send_json({"type": "watch", "images": [FRAME, FRAME]})
                alert = ws.receive_json()
                self.assertEqual((alert["type"], alert["level"]), ("alert", "issue"))
                self.assertEqual(alert["message"], "The heat looks too high.")
                self.assertEqual(len(watch.call_args.args[1]), 2)
                ws.send_json({"type": "watch", "images": [FRAME]})   # too soon: ignored
                ws.send_json({"type": "ping"})
                self.assertEqual(ws.receive_json()["type"], "pong")
                self.assertEqual(watch.await_count, 1)
                watch.return_value = {"status": "step_done", "message": "Looks done, say next.", "step": 0}
                with patch.object(cook_api, "WATCH_MIN_SECONDS", 0):
                    ws.send_json({"type": "watch", "images": [FRAME]})
                    self.assertEqual(ws.receive_json()["level"], "step_done")
                    ws.send_json({"type": "watch", "images": [FRAME]})   # same step: no second nudge
                    ws.send_json({"type": "watch", "images": "not-a-list"})
                    ws.send_json({"type": "ping"})
                    self.assertEqual(ws.receive_json()["type"], "pong")
                    self.assertEqual(watch.call_args.args[2], "The heat looks too high.",
                                     "the model is told what it already said")

    def test_recipe_upload_becomes_the_active_recipe(self):
        parsed = {"name": "Grandma's Noodles", "serves": 2, "time_minutes": 10,
                  "ingredients": ["200 g egg noodles", "2 tbsp butter"],
                  "steps": [{"text": "Boil the noodles for 3 minutes.", "done_when": None, "timer_seconds": 180},
                            {"text": "Toss with butter.", "done_when": "noodles are glossy", "timer_seconds": None}]}
        model = cook_api.assistant.model
        with patch.object(model, "openai_client", object()), \
                patch.object(model, "streaming_openai", SimpleNamespace(client=None)), \
                patch("cook_main.parse_recipe", AsyncMock(return_value=parsed)) as parse, \
                patch.object(model, "process_query_with_openai", AsyncMock(return_value="Here's your recipe.")) as reply, \
                patch.object(cook_api.assistant, "greet", AsyncMock(return_value="Hi!")), \
                TestClient(cook_api.app) as client:
            with client.websocket_connect("/ws?client_id=upload-regression") as ws:
                open_chat(ws)
                ws.send_json({"type": "recipe_upload", "text": "Grandma's noodles: boil, toss with butter."})
                self.assertEqual(ws.receive_json()["type"], "thinking")
                result = ws.receive_json()
                self.assertEqual(parse.call_args.args[2], "Grandma's noodles: boil, toss with butter.")
                cooking = result["cooking"]
                self.assertEqual((cooking["recipe"], cooking["custom"], cooking["step"]), ("Grandma's Noodles", True, None))
                self.assertEqual(cooking["steps"], ["Boil the noodles for 3 minutes.", "Toss with butter."])
                self.assertIn("uploaded by the cook", reply.call_args.kwargs["context"]["prompt_context"])
                started = ask(ws, "start")
                self.assertEqual((started["cooking"]["step"], started["cooking"]["timer_seconds"]), (0, 180))
                self.assertEqual(ask(ws, "next")["cooking"]["done_when"], "noodles are glossy")
                ws.send_json({"type": "recipe_upload", "text": "x" * 20_001})
                ws.receive_json()
                self.assertIn("could not be used", ws.receive_json()["message"])
                self.assertEqual(parse.await_count, 1)

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

    def test_navigation_fast_path_is_only_for_short_commands(self):
        self.assertIsNone(detect_navigation("is it done?"))
        self.assertIsNone(detect_navigation("what do I do next time?"))
        self.assertIsNone(detect_navigation("Search online for my next recipe"))
        self.assertIsNone(detect_navigation("I'm done chopping the onions, does this look right?"),
                          "a longer message is the model's call, with the camera")
        self.assertIsNone(detect_navigation("change recipe"), "stopping a recipe is a model tool now")
        self.assertIsNone(detect_navigation("give me suggestions"))
        self.assertEqual(detect_navigation("next"), ("next",))
        self.assertEqual(detect_navigation("what's next?"), ("next",))
        self.assertEqual(detect_navigation("start"), ("start",))
        for go in ("ok start", "yes let's start", "let's begin", "start cooking now"):
            self.assertEqual(detect_navigation(go), ("start",), go)
        for go in ("I'm ready", "ok let's go", "ready", "yes do it"):
            self.assertEqual(detect_navigation(go), ("ready",), go)
        self.assertIsNone(detect_navigation("how do I start the sauce"))
        self.assertEqual(detect_navigation("back"), ("previous",))
        self.assertEqual(detect_navigation("step 4"), ("goto", 4))

    def test_image_validation(self):
        for image in ("invalid", "data:image/gif;base64,YQ==", "data:image/jpeg;base64,!!"):
            with self.assertRaises(ImageError):
                decode_image(image, 100)
        with self.assertRaises(ImageError):
            decode_image("data:image/jpeg;base64,YWI=", 1)
        self.assertEqual(len(decode_images([FRAME] * 6, 100)), 4, "only the newest frames are kept")
        with self.assertRaises(ImageError):
            decode_images(FRAME, 100)

    def test_completion_parameters_and_placeholder_keys(self):
        self.assertIn("max_completion_tokens", AzureOpenAIConfig().completion_kwargs())
        self.assertIn("max_tokens", AzureOpenAIConfig(deployment_name="gpt-4o-mini").completion_kwargs())
        with patch.dict(os.environ, {"OPENAI_API_KEY": "replace_with_key"}):
            config._drop_placeholders()
            self.assertNotIn("OPENAI_API_KEY", os.environ)


class AssistantTests(unittest.IsolatedAsyncioTestCase):
    """The model's tools, the catalog it suggests from, uploads and the camera watch."""

    def setUp(self):
        self.assistant = cook_api.assistant
        self.user = f"assistant-{self._testMethodName}"

    def visit(self):
        return self.assistant.store.current_visit(self.user)

    def test_tools_choose_move_stop_and_remember(self):
        run = lambda tool, **args: self.assistant.run_tool(self.user, tool, args)
        missing = run("choose_recipe", recipe_name="Beef Wellington")
        self.assertFalse(missing["ok"])
        self.assertLessEqual(len(missing["closest_catalog_recipes"]), 2)
        chosen = run("choose_recipe", recipe_name="Mapo Tofu")
        self.assertTrue(chosen["ok"])
        self.assertTrue(chosen["ingredients"] and chosen["steps"])
        self.assertIsNone(self.visit().step)
        self.assertEqual(run("go_to_step", step=2)["text"], chosen["steps"][1])
        self.assertEqual(self.visit().step, 1)
        self.assertEqual(run("go_to_step", step=99)["status"], "All steps are finished.")
        self.assertEqual(run("go_to_step", step=0)["step"], 0)
        self.assertIsNone(self.visit().step)
        self.assertFalse(run("go_to_step", step="two")["ok"])
        self.assertTrue(run("update_cook_profile", name="Kit", dietary="no pork")["ok"])
        self.assertEqual((self.visit().cook_name, self.visit().dietary), ("Kit", "no pork"))
        self.assertTrue(run("stop_recipe")["ok"])
        self.assertIsNone(self.visit().recipe_id)
        self.assertFalse(run("go_to_step", step=1)["ok"])
        self.assertFalse(run("explode")["ok"])

    def test_catalog_only_while_choosing_and_compact(self):
        context = self.assistant.context
        turn = context.build(self.user, "what should I cook?")
        self.assertIn("[CATALOG]", turn.text)
        self.assertIn("never list them all", turn.text)
        self.assertIn("Kitchen experience: not known yet", turn.text)
        self.assertEqual(ingredient_core("200 grams spaghetti, dried"), "spaghetti")
        self.assertEqual(ingredient_core("2 tablespoons light soy sauce"), "light soy sauce")
        self.assistant.run_tool(self.user, "choose_recipe", {"recipe_name": "Egg Fried Rice"})
        turn = context.build(self.user, "how long do I fry the rice?")
        self.assertNotIn("[CATALOG]", turn.text)
        self.assertIn("Status: not started yet", turn.text)

    async def test_greeting_is_once_per_chat(self):
        self.assertEqual(await self.assistant.greet(self.user), GREETING_FALLBACK)
        self.assertIsNone(await self.assistant.greet(self.user))

    def test_recipe_import_validation(self):
        good = validate_recipe({"name": " Toast ", "serves": "2", "ingredients": ["bread", ""],
                                "steps": ["Toast the bread.", {"text": "Butter it.", "timer_seconds": "-5"}]})
        self.assertEqual(good["name"], "Toast")
        self.assertEqual(good["serves"], 2)
        self.assertEqual(good["ingredients"], ["bread"])
        self.assertEqual([s["text"] for s in good["steps"]], ["Toast the bread.", "Butter it."])
        self.assertIsNone(good["steps"][1]["timer_seconds"])
        for bad in (None, {"error": "this is a shopping list"}, {"name": "Toast", "steps": []},
                    {"steps": ["Do it."]}):
            with self.assertRaises(RecipeImportError):
                validate_recipe(bad)

    async def test_watch_speaks_only_about_problems_during_a_step(self):
        model = self.assistant.model
        provider = SimpleNamespace(available=True, inline=True, config=VisionConfig())
        frames = [(b"abc", "image/jpeg")]
        with patch.object(self.assistant, "vision", provider), patch.object(model, "openai_client", object()):
            with patch.object(model, "complete", AsyncMock()) as complete:
                self.assertIsNone(await self.assistant.watch(self.user, frames), "nothing to check before a step")
                self.assistant.run_tool(self.user, "choose_recipe", {"recipe_name": "Egg Fried Rice"})
                self.assistant.run_tool(self.user, "go_to_step", {"step": 1})
                complete.return_value = '{"status": "ok", "message": "Looks good", "seen": "rice in a wok"}'
                quiet = await self.assistant.watch(self.user, frames)
                self.assertEqual((quiet["status"], quiet["message"]), ("ok", ""))
                complete.return_value = '{"status": "hazard", "message": "Pan is smoking. Turn the heat down.", "seen": "smoke"}'
                turns = self.visit().turn_count
                alert = await self.assistant.watch(self.user, frames, last_alert="earlier note")
                self.assertEqual(alert, {"status": "hazard", "message": "Pan is smoking. Turn the heat down.", "step": 0})
                self.assertEqual(self.visit().turn_count, turns + 1, "alerts become part of the conversation")
                sent = complete.call_args.args[0][1]["content"]
                self.assertIn("Current step 1", sent[0]["text"])
                self.assertIn("earlier note", sent[0]["text"])
                self.assertEqual(sent[1]["image_url"]["detail"], "low")
                complete.return_value = "not json"
                self.assertEqual((await self.assistant.watch(self.user, frames))["status"], "unclear")


if __name__ == "__main__":
    unittest.main()
