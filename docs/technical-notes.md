# LH_cooking - technical notes

A voice cooking coach for people cooking at home. The cook props up their
phone, opens a web page, and talks to it: "let's start", "next", "is this
done?", "can I use linguine instead?". The coach walks them through a recipe
step by step, answers questions grounded in recipe notes, and can look through
the phone camera with a vision-language model (VLM).

Forked from PAL_OP (the Ocean Park guide) on 2026-11-02. The retrieval,
conversation-memory and grounding design carried over; the zoo content, the
YOLO/VGG16 vision service, the ESP32 firmware and the server-side speech
stack did not. This repository starts from a single commit; PAL_OP's history
stays in PAL_OP.

---

## How it fits together

```
phone browser (chatbot/static/index.html)
  camera  -> JPEG frame, attached to each question
  mic     -> Web Speech API -> text
  speaker <- speechSynthesis
      |  wss://<domain>/ws   {"type": "ask", "message": "...", "image": "data:image/jpeg;base64,..."}
      v
chatbot (FastAPI, cook_api.py)
  cook_main.py        one turn: onboarding -> look -> context -> model -> record
  vision.py           VLM provider (none | openai_compatible | azure)
  context_builder.py  step navigation, subject resolution, layered prompt
  knowledge_base.py   recipes.yaml + glossary.yaml -> knowledge.db
  conversation_state  memory.db: session profile, recipe + step, turns, ledger
  streaming_openai.py personas: step | help | safety  -> gpt-5-mini (Azure)
      ^
Caddy (HTTPS) on Tencent Cloud Lighthouse
```

Speech runs in the browser, so the server only ever handles text and an
optional frame. A phone only allows camera and microphone access over HTTPS
(localhost is the one exception), which is why production sits behind Caddy
with a real domain.

### One turn

1. **Onboarding** (first four answers of a session, no model call): name,
   kitchen experience, allergies / foods avoided, and what they're cooking.
   The dish answer is resolved to a recipe if one matches. Selection returns
   an ingredient preview with quantities and serving size before step 1, without
   a model call. Saying `start` or tapping **Start cooking** begins the tutorial.
   Switching to another recipe also returns its preview first, including for
   commands such as `start chicken curry`. The cooking payload always includes
   `ingredients`, `serves` and `time_minutes`, including after reconnecting.
   The saved recipe picker searches names, aliases and ingredient text across
   50 curated recipes. Ingredient lists are indexed in `recipe_ingredients`;
   YAML remains the source of truth.
2. **Navigation** is decided by pattern, not by the model: `start`, `next`,
   `go back`, `repeat`, `step 4`, `stop cooking`. It moves the step pointer
   before the prompt is built. "Is it done?" is a doneness question, not
   "next".
3. **Look** -- only if a VLM is configured, the phone attached a frame, and
   the question needs eyes ("is this done?", "does this look right?", a
   safety question). The observation goes into the prompt; it never answers
   the cook directly.
4. **Context** is rendered as labelled layers: `[COOK] [RECIPE] [NOW]
   [RECENT] [FACTS] [QUESTION]`. `[RECIPE]` carries the current step and its
   "done when" cue, so the model and the camera judge against the same
   target.
5. **Persona**: `step` for navigation, `safety` for a safety facet or a
   hazard the camera saw, `help` otherwise. Model text streams as WebSocket
   `delta` events; one final `response` carries the full text and updated state.
   The browser queues complete sentences for speech as they arrive, without
   speaking the final response a second time. Interrupted streams retain the
   text already delivered and append an interruption notice.

The recipe and step pointer live on the session, not on the current topic,
so "what does emulsify mean?" in the middle of step 6 doesn't lose the
cook's place.

## The two slots

Everything model-related lives in **one file, `chatbot/.env`** (template:
`chatbot/.env.example`). Whoever picks the models fills in two slots and
restarts the server; no code changes.

| Slot | What it does | Variables |
|---|---|---|
| 1. Chat model | writes every reply | `OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_DEPLOYMENT_NAME` |
| 2. Vision model | describes the camera frame when a question needs it | `VLM_PROVIDER`, `VLM_MODEL` (+ `VLM_BASE_URL`, `VLM_API_KEY` for non-Azure) |

Both on the university's Azure: fill slot 1, then set `VLM_PROVIDER=azure`
and `VLM_MODEL=<an image-capable deployment>`. The VLM reuses slot 1's
endpoint and key.

**Both slots can stay empty.** The coach then runs in *notes-only* mode:
onboarding, step navigation, timers and retrieval all work, and replies are
the recipe notes read back verbatim instead of model-written answers. Good
enough to test the app before any model is chosen.

`GET /health` shows configuration: `model.configured`, the deployment, and the
vision provider. It does not prove connectivity. Run
`python scripts/check_setup.py --live --image food.jpg` from the repository root
to test actual chat streaming and vision access; this uses API quota.

The coach's standing instructions (tone, length, grounding rules, the
`step` / `help` / `safety` personas) are plain text in
`chatbot/streaming_openai.py`, in `system_prompts`. They don't depend on which
model is chosen.

> A variable set in the OS environment outranks `chatbot/.env`. On the
> development PC an old `OPENAI_API_KEY` from PAL_OP is set as a Windows user
> variable; while it exists, the coach uses that (expired) key instead of
> notes-only mode. Remove it under *System Properties > Environment
> Variables* if PAL_OP no longer needs it.
>
> The repository-root `.env` is a PAL_OP leftover and is not read by anything.

## Running it

### Locally (recommended for testing)

```bash
cd chatbot
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -r requirements.txt
python cook_api.py
```

Open <http://localhost:8000> in a desktop browser (Chrome or Edge for voice
input). Camera and microphone work on `localhost` without HTTPS, so a laptop
webcam stands in for the phone. A phone on the same Wi-Fi can open the page
but won't get camera or mic over plain `http://`; that needs HTTPS, which the
Lighthouse deployment provides.

### Docker

```bash
make up        # docker compose up -d, coach on :8000
make test      # scripted cooking session against the real model
```

### Tencent Cloud Lighthouse (later)

Not deployed yet. When it is, the same two slots go in `.env.lighthouse`
instead of `chatbot/.env`:

```bash
cp .env.lighthouse.example .env.lighthouse         # DOMAIN, plus the two slots
make lighthouse-up
```

- Point the domain's DNS at the Lighthouse public IP and open ports 80/443 in
  the Lighthouse firewall; Caddy fetches the certificate itself.
- A **mainland China** region requires ICP filing for a domain served from it.
  Hong Kong and overseas regions do not.
- The browser's speech recognition uses Google's service on Chrome/Android
  and Apple's on Safari/iOS. Chrome's will not work for users inside mainland
  China; if that audience matters, add Tencent ASR server-side.
- Container names are prefixed `lh_cooking_`, so this stack can share a host
  with PAL_OP, but only one Caddy can own ports 80/443.

## The vision model

Chosen by environment variable; nothing else changes:

| `VLM_PROVIDER` | Needs | Use for |
|---|---|---|
| `none` (default) | -- | no camera understanding; the preview still shows |
| `openai_compatible` | `VLM_BASE_URL`, `VLM_API_KEY`, `VLM_MODEL` | Tencent Hunyuan, Qwen-VL, OpenAI, a self-hosted vLLM -- anything speaking OpenAI chat completions with `image_url` parts |
| `azure` | `VLM_MODEL` (deployment; defaults to the chat one) | an image-capable deployment on the HKUST Azure endpoint |

`VLM_MODE=auto` (default) only calls the VLM when the question needs it;
`always` sends every frame. The VLM is asked for JSON:
`{summary, items[], state, hazards[]}`; a reply that isn't JSON is still
used as a plain summary.

Try a provider without the app:

```bash
curl -X POST localhost:8000/api/vision/test -H 'content-type: application/json' \
     -d @frame.json      # {"image": "data:image/jpeg;base64,...", "question": "is this done?"}
```

A provider that fits neither shape is one subclass of `VisionProvider` in
`chatbot/vision.py`: implement `_complete(image, mime, request)` and register
it in `PROVIDERS`.

Frames are held in memory for one request and never written to disk or
logged. Only the text observation is stored with the turn.

## The cooking knowledge

| File | Role |
|---|---|
| `chatbot/recipes.yaml` | recipes: ingredients, ordered steps with `done_when` and `timer_seconds`, and facts |
| `chatbot/glossary.yaml` | techniques, ingredients and kitchen topics (safety, storage) |
| `chatbot/knowledge.db` | **built** from the two YAML files; gitignored, rebuilt in the Docker build |
| `chatbot/memory.db` | runtime: session profile, recipe + step, turns, delivered-fact ledger |

Each fact is one finished sentence with a facet: `overview ingredients
technique doneness heat timing substitution safety equipment storage tip`.
A recipe's `techniques:` list links it to glossary entries, so a technique
question asked mid-recipe reaches the technique's own facts.

There are five recipes. Ingredient/step consistency has been reviewed and key
chicken, egg, leftover and fire guidance corrected with cited safety sources.
The salad is adapted from USDA's Nutrition.gov recipe. House recipes have not
been kitchen-tested; original facts and glossary notes still marked
`starter:unverified` need further review. Timers and photos do not establish
safe internal temperatures.

After editing either YAML file:

```bash
cd chatbot && python knowledge_base.py --rebuild
python knowledge_base.py --ask "can I use linguine instead" --recipe "aglio olio"
```

Debug endpoints: `GET /health` (live model + vision provider),
`/api/recipes`, `/api/knowledge` (facts per facet), `/api/context/{client_id}`,
`/api/retrieve?q=...&recipe=...` (retrieval without a model call).

## The chat model

Unchanged from PAL_OP: **gpt-5-mini** on the HKUST Azure proxy, set by
`AZURE_DEPLOYMENT_NAME`. `config.completion_kwargs()` emits the right
parameters for gpt-5 vs gpt-4o; never pass `max_tokens` or `temperature` at
a call site. `python scripts/check_model.py` verifies a deployment and exits
non-zero on failure. gpt-5 gotchas: `temperature` must be 1, the api-version
must stay `2024-10-21`, and a small completion budget returns an empty string
rather than an error.

## Validation and remaining checks

The browser header has Profile and History dialogs. A `profiles` table stores
name, experience and dietary needs for the browser's existing random user ID.
Upgrading an existing database seeds it from the latest completed visit; edits
update the active visit without changing its recipe or step. New visits copy
the profile and start onboarding at today's dish.

Profile/history commands use the existing WebSocket: `profile_get`,
`profile_save`, `history_list`, and `history_detail`, with a `request_id` for
matching `settings_result` replies. History queries use the connected user's ID
and return paginated session summaries or chronological turns. They do not call
the model or enter the chat/speech queue. History retains text and stored vision
descriptions, not photo bytes. Clearing browser storage changes its identity.

The WebSocket also receives a `chat_id`, generated in memory on each page load
or New chat. It binds the store's current-visit lookup through a task-local
ContextVar, which propagates through `asyncio.to_thread`. Each page/chat has
separate recipe state, recent turns and delivered facts while sharing the
browser's profile/history identity. Reconnects reuse the chat ID; reopening,
refreshing, or restoring an exited page creates a fresh one. The browser clears
drafts, attachments, timers, camera/voice activity and pending requests when
starting over, and ignores replies from the old socket. Existing clients that
omit `chat_id` retain the legacy idle-session behavior.

`chatbot/web_search.py` sends a Responses API request with a required
`web_search` tool. Search requires both a completed `web_search_call` and usable
URL citations; plain model text never counts as a successful lookup. Search
errors are sanitized and cached for five minutes in a running browser service
to avoid repeated failing requests. Desktop queries start a fresh worker and
can retry immediately. Source links travel in the WebSocket `sources` field.
Search receives the current cooking context, including dietary requirements.
Step navigation and immediate safety responses continue through the usual path.

`WEB_SEARCH_ENABLED`, `AZURE_RESPONSES_BASE_URL` and
`WEB_SEARCH_TIMEOUT_SECONDS` propagate through local launches and Lighthouse
bundles. A custom base reuses the existing key and the `api-key` header.
Live checks on 2026-11-03 returned HTTP 404 at both documented v1 Responses
route candidates on the configured HKUST gateway. Search is implemented but
needs provider-enabled access before a real result can be verified.

`python -m unittest discover -s tests -v` runs offline regression tests with
temporary databases and mocked model calls. No developer API keys are loaded.
`node tests/test_browser.cjs` checks streamed sentence speech using browser
stubs, when Node.js is available.

- Azure still needs the real key and account-permitted deployment names.
- Real phone, microphone, camera, speech and HTTPS behavior need testing on a
  deployed domain; see `tencent-lighthouse-deployment.md` for acceptance checks.
- Kitchen testing and review of remaining unverified glossary content are open.
- Older HTML documents under `docs/` describe the zoo system and are historical.
