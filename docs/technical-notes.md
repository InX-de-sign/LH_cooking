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
  camera  -> rolling buffer of JPEG frames (one every ~1.2 s, last 3 kept)
  mic     -> Web Speech API -> text
  speaker <- speechSynthesis
      |  wss://<domain>/ws   ask {message, images:[frames]} | recipe_upload | watch
      v
chatbot (FastAPI, cook_api.py)
  cook_main.py        greet, one turn, tools, recipe upload, camera watch
  streaming_openai.py system prompt + tool loop -> gpt-5-mini (HKUST GenAI)
  context_builder.py  step fast path, subject resolution, layered prompt
  recipe_import.py    uploaded text/photo -> structured recipe (JSON mode)
  vision.py           frames inline to the chat model, or a separate VLM
  knowledge_base.py   recipes.yaml + glossary.yaml -> knowledge.db
  conversation_state  memory.db: profile, recipe + step, uploads, turns, ledger
      ^
Caddy (HTTPS) on Tencent Cloud Lighthouse
```

Speech runs in the browser, so the server only ever handles text and images.
A phone only allows camera and microphone access over HTTPS (localhost is the
one exception), which is why production sits behind Caddy with a real domain.

### One turn

The model drives the conversation. There is no scripted questionnaire and no
canned reply: every message the cook sees is written by the model, except the
notes-only fallbacks when no model is configured.

1. **Greeting**: a new chat gets one model-written greeting inviting the cook
   to name a dish, ask for ideas, or upload a recipe. It is stored as the first
   turn, so a reconnect does not greet again.
2. **Step fast path**: short commands (six words or fewer) are matched by
   pattern and move the step pointer before the prompt is built: `next`,
   `back`, `repeat`, `step 4`, `start` / `ok start` (restart at step 1), and
   `ready` / `let's go` (only starts a recipe that has not begun). Longer or
   fuzzier messages ("I've finished chopping, does this look right?") go to the
   model. "Is it done?" is a doneness question, not "next".
3. **Camera**: when the camera is on, each question carries the last three
   frames (oldest first). With `VLM_PROVIDER=azure` and the same deployment as
   chat, the frames go straight into the chat call at `detail: low`, so the
   model that answers also looks, next to the current step. A photo the cook
   attaches on purpose goes at `detail: auto`. Step commands skip the frames.
   A separate VLM (`openai_compatible`) still writes a text observation first.
4. **Context** layers: `[COOK]` (profile or "not known yet"), `[RECIPE]` (all
   ingredients, every numbered step, the current step with its "done when"
   cue and timer), `[CATALOG]` (only while choosing: one compact line per
   recipe), `[NOW]`, `[FACTS]`, `[QUESTION]`. The last six exchanges are sent
   as real chat messages.
5. **Model + tools**: one streamed call with four tools: `choose_recipe`,
   `go_to_step`, `stop_recipe`, `update_cook_profile`. If the model calls one,
   it runs, its result is appended, and a second tool-free call writes the
   reply (one tool round only, so it never starts step 1 on its own after
   loading a recipe). The system prompt sets the format: short replies,
   numbered lists for multi-action instructions, `- ` bullets for ingredients,
   at most 2 suggestions, never the whole catalog.
6. **Streaming and speech**: text streams as WebSocket `delta` events; one
   final `response` carries the full text and updated state. The browser
   renders lists as real lists. Speech reads complete lines and sentences as
   they arrive, reads `1. Boil` as "1, Boil", and replaces runs of three or more
   bullets (ingredient lists) with "The list is on your screen."

### Uploaded recipes

**Upload recipe** takes pasted text, a `.txt`/`.md` file, or a photo or
screenshot. `recipe_import.py` asks the chat model (JSON mode, image attached)
for `name, serves, time_minutes, ingredients[], steps[{text, done_when,
timer_seconds}]`, keeping the recipe's own quantities and splitting long
paragraphs into one action per step. The result is validated, stored in
`custom_recipes` (private to the browser's user ID), and becomes the active
recipe via `visits.recipe_custom_id`. From then on the step card, step
commands, timers, camera questions and watch mode work exactly as for a
catalog recipe.

### Mobile workflow

The page follows the cook through four stages, set on `<body data-stage>` from
the cooking state: **choose** (no recipe) → **prep** (recipe, step `null`) →
**cook** (step in range) → **done** (step past the end). Each stage shows its
own panel; Cook makes the step text large, shows Back / Repeat / Next and a
swipe area (left = next, right = back; tap = talk), floats the camera as a
thumbnail, and collapses the chat to the last coach message (CSS `:has()`).
Entering Cook turns on spoken replies, and hands-free if the microphone was
already allowed; leaving restores the saved settings. Step commands from
buttons and swipes never attach camera frames.

More model tools feed the phone directly: `suggest_recipes` (idea cards; the
two names are stored on the visit so "the first one" is unambiguous),
`start_timer` (named timers), `scale_recipe` (and `servings` on
`choose_recipe`). Scaling is deterministic (`scaling.py`): ingredient lines
scale their leading quantity; step text scales only amounts followed by a
measuring unit, never times or temperatures. Feedback at Done is stored in
`feedback`, and the note is appended to `profiles.notes`, which appears in
`[COOK]` on every later turn.

Language (`en`, `yue`, `zh`) is chosen in the menu, sent on connect (`lang`)
and with each message, stored on the profile, and used for the reply
instruction, the greeting, watch alerts, browser speech recognition and the
speaking voice.

Links are fetched by `recipe_import.fetch_recipe_page`: http(s) on standard
ports only, every resolved address must be public (checked again on each
redirect), 2 MB cap; schema.org Recipe JSON-LD is used when present. The app
manifest declares a share target; `/share-target` holds the shared text, link
or image for ten minutes under a random token and the page opens it in the
Upload dialog. `sw.js` caches only the page shell and icons.

### Coach watches

While the camera is on and a step is in progress, the phone sends the latest
frames every `WATCH_INTERVAL_SECONDS` (default 20) unless the cook is talking,
listening or waiting for a reply. The server rate-limits to half that interval.
The model answers in JSON with `ok | unclear | issue | hazard | step_done`.
Only `issue` and `hazard` (once per new problem) and `step_done` (once per
step) reach the phone, as an `alert` bubble that is spoken and vibrates. Alerts
are stored as turns, so later answers know what was already said. The toggle
**Coach watches** turns it off.

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
naming a catalog dish selects it, step navigation, timers and retrieval all
work, and replies are the recipe notes read back verbatim instead of
model-written answers. Suggestions, uploads and camera checks need the model.

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
the profile. The model fills it in from conversation with
`update_cook_profile` (only the fields the cook mentioned).

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
