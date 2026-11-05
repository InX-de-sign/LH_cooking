# LH_cooking

A voice cooking coach for people cooking at home. The cook props up a phone,
opens the web page, and talks to it: "let's start", "next", "is this done?",
"can I use linguine instead?". The coach walks them through a recipe step by
step and can look through the phone camera.

## How it works now

```
Phone browser (chatbot/static/index.html)
  mic -> speech-to-text in the browser
  camera frame or uploaded photo + question text
        |
        v
Server (chatbot/cook_api.py, Python/FastAPI)
  1. recipe notes  -> current step + relevant facts   (recipes.yaml, glossary.yaml)
  2. VISION SLOT   -> describes an uploaded photo, or camera frame for visual questions
  3. CHAT SLOT     -> streams the spoken reply
        |
        v
Phone speaks the reply (text-to-speech in the browser)
```

Already working, tested locally:

- **New chat** (top right): start a fresh conversation for a different dish.
  Leaving and reopening or refreshing the page also starts a fresh chat.
  Your saved profile and past conversations in History are kept. Temporary
  connection drops within the open page resume the same chat.
- First-run questions (name, experience, allergies, what are you cooking)
- **Profile** (top right): edit your name, cooking experience, and allergies or
  dietary needs. Changes apply to the next cooking reply and are remembered in
  future sessions in the same browser. A new session asks what you are cooking.
- **History** (top right): browse previous sessions and read their questions,
  replies, and photo descriptions. Load more sessions or older messages as needed.
  Opening history leaves the current recipe, step and unsent message in place.
- Step navigation by voice or buttons: start, next, back, repeat, "step 4"
- Recipe questions answered from the notes (ingredients, substitutions, safety...)
- 50 saved recipes with ingredient quantities and guided steps, including
  curries, pasta, breakfasts, stir-fries, soups and vegetarian dishes, plus a photo-inspired
  spicy shrimp glass-noodle dish. The photo does not reveal the exact seasoning;
  the recipe uses optional chilli and has not been kitchen-tested. Restart a
  running local app after recipe edits so it can refresh its recipe index.
- Recipe selection automatically shows the full ingredient list and serving size
  before the tutorial. Gather the ingredients, then say **start** or tap
  **Start cooking**. Ingredients remain available in a collapsible list during cooking.
  Changing dishes opens the new ingredient preview first. You can also switch
  recipes mid-cook (for example “change the dish to stir-fry shrimp”); the left
  panel updates and shows the new ingredient preview. The recipe picker can
  search dish names, common aliases and ingredients. These are curated familiar
  dishes, not a measured ranking of search popularity; new recipes are house
  drafts and have not been kitchen-tested.
- Built-in API web search integration for recipes and ingredients, with source
  links. **The HKUST GenAI API does not support online search**, so this only
  works with other providers that expose Responses + `web_search` (see below).
- Step timers, hands-free mode, camera preview
- Paperclip button to the left of the browser chat input: choose a JPEG, PNG or
  WebP photo, preview it, then send it with your question. Remove clears the
  attachment; use the paperclip again to replace it. Each photo is sent with one message only. Finish the
  setup questions first. Sending a photo alone asks what is visible in it.
- Runs with **no keys at all**: replies are then the recipe notes read back
  word for word

## Task status

- [ ] **Use online search with a non-HKUST API**: HKUST GenAI has no Responses /
      `web_search` route (HTTP 404 on 2026-11-04). Point chat/search at another
      provider that supports built-in web search, or wait until HKUST adds it.
- [ ] **Kitchen-test recipes**: times are estimates; original house notes and
      glossary material still marked `starter:unverified` need practical review.
- [ ] **Deploy and test a real phone over HTTPS**: needs your Lighthouse server
      and domain; see [deployment guide](docs/tencent-lighthouse-deployment.md).

## Where to fill the slots

Everything goes in **one file: `chatbot/.env`**. Copy `chatbot/.env.example`
to `chatbot/.env` and fill it in. No code changes needed.
The local `chatbot/.env` has already been prepared in this workspace. Put your
HKUST key after `OPENAI_API_KEY=` there. Never paste it into HTML or source code.

### Slot 1: chat model (writes the replies)

```
OPENAI_API_KEY=<HKUST GenAI API key>
AZURE_OPENAI_ENDPOINT=https://hkust.azure-api.net
AZURE_DEPLOYMENT_NAME=gpt-5-mini
AZURE_OPENAI_API_VERSION=2024-10-21
```

The API version is used only for direct Azure resources. For HKUST, the app
automatically uses `https://hkust.azure-api.net/hkust-genai/v1` with the
`api-key` header.

### Slot 2: vision model (looks at the camera)

Using a model on the same HKUST GenAI account (reuses slot 1's key):

```
VLM_PROVIDER=azure
VLM_MODEL=<image-capable model, or leave blank to reuse the chat model>
```

Your configured `gpt-5-mini` was verified with image input on the current
HKUST GenAI route. Leaving `VLM_MODEL` empty reuses the chat model. Restart
the browser service after editing `.env`; the desktop launcher uses saved
settings on its next request.

Using a model somewhere else (any OpenAI-style API):

```
VLM_PROVIDER=openai_compatible
VLM_BASE_URL=<API base URL>
VLM_API_KEY=<key>
VLM_MODEL=<model name>
```

Leave `VLM_PROVIDER=none` to run without vision.

### Built-in online search

Open **Recipe search** in the Python launcher, enter a recipe or ingredient
question, then click **Search online**. In the browser coach, finish onboarding
and ask, for example, **"Search online for a tofu recipe"** or **"Look up cumin
substitutes online"**. Unsupported cooking questions can also trigger search
when local notes are missing. Step commands keep using the saved recipes.

Search uses the provider's Responses API with its built-in `web_search` tool,
reusing `OPENAI_API_KEY` and `AZURE_DEPLOYMENT_NAME`. It requires a completed
search call and citation links before displaying a result as an online answer.
Results have clickable sources; new recipes are not automatically imported into
the saved step-by-step recipe library. Search replies arrive after completion;
ordinary chat replies still stream.

**HKUST GenAI does not support online search.** Live checks returned HTTP 404
from both `https://hkust.azure-api.net/hkust-genai/v1/responses` and
`https://hkust.azure-api.net/openai/v1/responses`. Chat and vision work on
HKUST; Recipe search / “search online” only works when you configure a
**different** API that exposes the Responses API with a built-in `web_search`
tool (for example a direct Azure OpenAI resource with web search enabled).
With HKUST alone, the app reports that search is unavailable instead of
faking citations from model knowledge.

Settings are in **Azure settings**, or `chatbot/.env`:

```dotenv
WEB_SEARCH_ENABLED=true
# Leave blank for the gateway default; set only to a provider-confirmed base.
AZURE_RESPONSES_BASE_URL=
WEB_SEARCH_TIMEOUT_SECONDS=60
```

A custom Responses base must use HTTPS and end in `/v1`. The HKUST default is
`https://hkust.azure-api.net/hkust-genai/v1`; direct Azure resources use
`https://RESOURCE.openai.azure.com/openai/v1`. This integration uses the
`api-key` header. Do not enter a different provider's URL unless that provider
is intended to receive your configured key. Restart the browser service after
changing settings. Desktop searches use saved settings immediately.

To test just search (uses API quota):

```powershell
.\.venv\Scripts\python.exe scripts/check_search.py --query "Find a tofu recipe online"
```

The setup follows the [HKUST integration guide](https://itso.hkust.edu.hk/services/general-it-services/generative-ai-tools/hkust-genai-api-service/integration-ai-tools)
and [Azure web search documentation](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/web-search).
Gateway compatibility must be verified separately from ordinary chat.

## Python desktop app (local launcher and cloud deployment)

Double-click **`Start Cooking Coach.pyw`** on Windows, or run from this folder:

```powershell
.\.venv\Scripts\python.exe app.py
```

The launcher also opens with `python app.py`; it needs Python 3.10+ with Tkinter.

1. **Azure settings**: enter your HKUST key, save, and optionally test chat or
   a photo. Existing `chatbot/.env` settings load automatically; the key is masked.
2. **Local app**: click **Start & open coach**. The browser opens after the
   service is ready. **Stop** stops only the process started by the launcher.
   With no key, recipe-notes mode still works. Closing the launcher stops it.
3. **Camera / photo**: open your computer camera (0 is the default), capture a
   frame, or upload a JPEG, PNG or WebP photo. Preview it, enter a question and
   click **Analyze photo** to see the vision model's description. Camera preview
   stays local; only Analyze sends the selected image to your configured model.
   Camera access stops after capture, when you leave this tab, or on app close.
4. **Recipe search**: look up recipes or ingredients online and open source links.
   HKUST GenAI does not support this; use another API with Responses / `web_search`.
5. **Lighthouse**: later, enter your domain, server address, SSH username and
   optional key-file path. **Check server** checks SSH/Docker; **Deploy** uploads
   the application and saved Azure settings, builds containers, and checks HTTPS.
   You need to create the server, install Docker, configure DNS and trust the
   SSH host first. See [the deployment guide](docs/tencent-lighthouse-deployment.md).

**Install / repair dependencies** creates `.venv` and installs
`requirements-desktop.txt`: service dependencies plus Pillow and OpenCV for
native photo preview and webcam capture. Reopen the launcher after installation.
Uploaded photos are oriented, resized and stripped of metadata before analysis;
captured images stay in memory. API tests, photo analysis and search use quota.
The **Prepare bundle** action writes `.deployment/lh-cooking.tar.gz`, including
your API key for the server. Keep that archive private; `.deployment/` is ignored
by git. No cloud server has been provisioned or deployed automatically.

## Run it locally from a terminal

```
cd chatbot
python -m venv .venv
.venv\Scripts\activate          (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python cook_api.py
```

Open http://localhost:8000 in Chrome or Edge.

The workspace also has a prepared `.venv`. From the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe chatbot/cook_api.py
```

## Check your setup

- http://localhost:8000/health shows configured models. `configured` and
  `available` mean settings exist, not that a live Azure request has succeeded.
- To test the vision model on a single photo without the app, send a POST to
  `/api/vision/test` with `{"image": "data:image/jpeg;base64,...", "question": "is this done?"}`.
- `python scripts/check_model.py` checks that the chat model in slot 1 answers.

From the repository root using the prepared environment:

```powershell
# Settings only; never prints the key. Exit 2 means the chat slot is empty.
.\.venv\Scripts\python.exe scripts/check_setup.py
# Calls Azure and uses API quota; checks chat streaming and optionally a photo.
.\.venv\Scripts\python.exe scripts/check_setup.py --live
.\.venv\Scripts\python.exe scripts/check_setup.py --live --image C:\path\to\food.jpg
# Offline checks, with fake model responses; no API quota needed.
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

JPEG/PNG/WebP photos must be smaller than 1.5 MB. A 401 usually indicates a key
problem; a 403 can indicate deployment access; for 404 check endpoint, deployment
name and API version. A successful live chat test does not verify vision unless
you also supply `--image` and see its PASS result.

The browser photo picker accepts source files up to 20 MB and resizes/re-encodes
them locally to fit the 1.5 MB API limit. The preview is local until you send the
message. An explicitly uploaded photo takes priority over a live camera frame
and always requests vision analysis for that prompt. A configured vision model
is required. If preparation or sending is blocked, the draft remains available.

## Good to know

- An `OPENAI_API_KEY` set in your computer's environment variables overrides
  `chatbot/.env`. If the server uses the wrong key, check there.
- `chatbot/.env` is never committed to git. Don't put keys anywhere else.
  Production uses the separately ignored `.env.lighthouse` on the server.
- Camera frames stay in request memory; text observations and conversation
  profiles are saved in SQLite. Text observations may also appear in server logs.
- Profiles and history use this browser's saved identity on this server. Another
  browser or clearing browser storage creates a different identity. Photo files
  themselves are not stored in history.
- The coach's instructions (tone, length, safety rules) are in
  `chatbot/streaming_openai.py` under `system_prompts`.
- More detail: `docs/technical-notes.md`. The other files in `docs/` describe
  the older zoo project this was forked from.
