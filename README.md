# LH_cooking

A voice cooking coach for people cooking at home. The cook props up a phone,
opens the web page, and talks to it: "let's start", "next", "is this done?",
"can I use linguine instead?". The coach walks them through a recipe step by
step and can look through the phone camera.

## How it works now

```
Phone browser (chatbot/static/index.html)
  mic -> speech-to-text in the browser
  camera frame + question text
        |
        v
Server (chatbot/cook_api.py, Python/FastAPI)
  1. recipe notes  -> current step + relevant facts   (recipes.yaml, glossary.yaml)
  2. VISION SLOT   -> describes the camera frame      (only for "is this done?"-type questions)
  3. CHAT SLOT     -> writes the spoken reply
        |
        v
Phone speaks the reply (text-to-speech in the browser)
```

Already working, tested locally:

- First-run questions (name, experience, allergies, what are you cooking)
- Step navigation by voice or buttons: start, next, back, repeat, "step 4"
- Recipe questions answered from the notes (ingredients, substitutions, safety...)
- Step timers, hands-free mode, camera preview
- Runs with **no keys at all**: replies are then the recipe notes read back
  word for word

## What's missing

- [ ] **Chat model**: none connected yet. Fill in slot 1.
- [ ] **Vision model**: none connected yet. Fill in slot 2.
- [ ] **Recipes**: only 3 starter recipes, written as placeholders and marked
      `starter:unverified`. They need checking and more need adding
      (`chatbot/recipes.yaml`).
- [ ] **Phone testing over the internet**: phones only allow camera and mic on
      HTTPS, so testing on a real phone needs the Tencent Cloud deployment
      (not set up yet). On a laptop, `localhost` works fine.
- [ ] Replies are spoken only after the whole answer arrives (no streaming yet).

## Where to fill the slots

Everything goes in **one file: `chatbot/.env`**. Copy `chatbot/.env.example`
to `chatbot/.env` and fill it in. No code changes needed.

### Slot 1: chat model (writes the replies)

```
OPENAI_API_KEY=<Azure subscription key>
AZURE_OPENAI_ENDPOINT=https://hkust.azure-api.net
AZURE_DEPLOYMENT_NAME=gpt-5-mini        # or another deployment, e.g. gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-10-21     # keep this for the HKUST endpoint
```

### Slot 2: vision model (looks at the camera)

Using a model on the same Azure account (reuses slot 1's key):

```
VLM_PROVIDER=azure
VLM_MODEL=<name of an Azure deployment that accepts images, e.g. gpt-4o>
```

Using a model somewhere else (any OpenAI-style API):

```
VLM_PROVIDER=openai_compatible
VLM_BASE_URL=<API base URL>
VLM_API_KEY=<key>
VLM_MODEL=<model name>
```

Leave `VLM_PROVIDER=none` to run without vision.

## Run it locally

```
cd chatbot
python -m venv .venv
.venv\Scripts\activate          (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python cook_api.py
```

Open http://localhost:8000 in Chrome or Edge.

## Check your setup

- http://localhost:8000/health shows which chat model and vision model are
  live (`"configured": false` means slot 1 is still empty).
- To test the vision model on a single photo without the app, send a POST to
  `/api/vision/test` with `{"image": "data:image/jpeg;base64,...", "question": "is this done?"}`.
- `python scripts/check_model.py` checks that the chat model in slot 1 answers.

## Good to know

- An `OPENAI_API_KEY` set in your computer's environment variables overrides
  `chatbot/.env`. If the server uses the wrong key, check there.
- `chatbot/.env` is never committed to git. Don't put keys anywhere else.
- The coach's instructions (tone, length, safety rules) are in
  `chatbot/streaming_openai.py` under `system_prompts`.
- More detail: `docs/technical-notes.md`. The other files in `docs/` describe
  the older zoo project this was forked from.
