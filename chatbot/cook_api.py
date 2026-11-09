# cook_api.py - FastAPI entry point for the cooking coach
"""
Serves the phone web app and the one WebSocket it talks to.

The phone does speech both ways in the browser (Web Speech API for listening,
speechSynthesis for speaking), so the server only ever sees text plus images:

    phone -> {"type": "ask", "message": "is this right?", "images": [<latest camera frames>]}
    phone -> {"type": "ask", "message": "...", "image": <photo>, "image_source": "upload"}
    phone -> {"type": "recipe_upload", "text": "...", "image": <photo of a recipe>}
    phone -> {"type": "watch", "images": [...]}     periodic camera check while cooking
    phone <- {"type": "thinking"} / {"type": "delta", "text": "..."} /
             {"type": "response", "message": "...", "cooking": {...}, "vision": {...}}
    phone <- {"type": "alert", "level": "issue|hazard|step_done", "message": "..."}

Camera and microphone access in a mobile browser require HTTPS (localhost is
the only exception), which is why production runs behind Caddy.
"""

import asyncio
import logging
import os
import sqlite3
import time
import uuid

from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from cook_main import WATCH_INTERVAL_SECONDS, CookingAssistant
from vision import ImageError, VisionRequest, decode_image, decode_images

# The phone asks every WATCH_INTERVAL_SECONDS; the server refuses anything
# faster so a misbehaving tab cannot burn the API quota.
WATCH_MIN_SECONDS = max(5, WATCH_INTERVAL_SECONDS // 2)
MAX_UPLOAD_TEXT = 20_000

app = FastAPI(title="LH Cooking Coach API")

assistant = CookingAssistant()

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
if os.path.isdir(STATIC):
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC, "index.html"))


def _session_payload(session_id: str) -> dict:
    return {
        "type": "system",
        "client_id": session_id,
        "cooking": assistant.cooking_state(session_id),
        "vision_available": assistant.vision.available,
        "watch_available": assistant.watch_available,
        "watch_interval": WATCH_INTERVAL_SECONDS,
    }


async def _open_chat(websocket: WebSocket, session_id: str):
    """Send the session state, then a greeting if this chat has not started."""
    await websocket.send_json(await asyncio.to_thread(_session_payload, session_id))
    visit = await asyncio.to_thread(assistant.store.current_visit, session_id)
    if visit.turn_count == 0:
        await websocket.send_json({"type": "thinking"})
        greeting = await assistant.greet(session_id)
        if greeting:
            await websocket.send_json({"type": "response", "message": greeting,
                                       "cooking": assistant.cooking_state(session_id)})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, client_id: str = Query(None), chat_id: str = Query(None)):
    """The phone's conversation channel.

    client_id identifies the saved profile/history. chat_id is generated for
    each page load or New chat; reconnects within that page reuse it.
    """
    await websocket.accept()
    session_id = (client_id or "").strip()[:64] or f"web_{uuid.uuid4().hex[:12]}"
    chat_token = assistant.store.bind_chat((chat_id or "").strip()[:64] or None)
    logger.info("Phone connected: %s", session_id)

    max_bytes = assistant.vision.config.max_image_bytes
    last_watch = 0.0
    last_alert = None
    nudged_steps = set()

    async def send_delta(text):
        await websocket.send_json({"type": "delta", "text": text})

    try:
        await _open_chat(websocket, session_id)

        while True:
            data = await websocket.receive_json()
            kind = data.get("type", "ask")

            if kind == "connect":
                requested = (data.get("client_id") or "").strip()[:64]
                if requested and requested != session_id:
                    session_id = requested
                    await _open_chat(websocket, session_id)
                continue
            if kind == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            if kind in ("profile_get", "profile_save", "history_list", "history_detail"):
                request_id = data.get("request_id")
                if not isinstance(request_id, str) or not 1 <= len(request_id) <= 64:
                    continue
                try:
                    if kind == "profile_get":
                        result = {"profile": await asyncio.to_thread(assistant.store.get_profile, session_id)}
                    elif kind == "profile_save":
                        profile = await asyncio.to_thread(assistant.store.save_profile, session_id, data.get("profile"))
                        result = {"profile": profile, "cooking": assistant.cooking_state(session_id)}
                    elif kind == "history_list":
                        result = await asyncio.to_thread(assistant.store.history, session_id, data.get("offset", 0))
                    else:
                        result = await asyncio.to_thread(assistant.store.history_detail, session_id,
                                                        data.get("session_id"), data.get("offset", 0))
                    await websocket.send_json({"type": "settings_result", "request_id": request_id, **result})
                except ValueError as exc:
                    await websocket.send_json({"type": "settings_result", "request_id": request_id, "error": str(exc)})
                continue

            if kind == "watch":
                # Periodic camera check while a step is in progress. Silent
                # unless the model sees a problem or a finished step.
                now = time.monotonic()
                if now - last_watch < WATCH_MIN_SECONDS:
                    continue
                last_watch = now
                try:
                    frames = decode_images(data.get("images"), max_bytes)
                except ImageError as exc:
                    logger.warning("Watch frames from %s rejected: %s", session_id, exc)
                    continue
                result = await assistant.watch(session_id, frames, last_alert)
                if not result or not result["message"]:
                    continue
                if result["status"] == "step_done":
                    if result["step"] in nudged_steps:
                        continue
                    nudged_steps.add(result["step"])
                else:
                    last_alert = result["message"]
                await websocket.send_json({"type": "alert", "level": result["status"],
                                           "message": result["message"],
                                           "cooking": assistant.cooking_state(session_id)})
                continue

            if kind == "recipe_upload":
                text = data.get("text") if isinstance(data.get("text"), str) else ""
                image, image_error = None, None
                if data.get("image"):
                    try:
                        image = decode_image(data["image"], max_bytes)
                    except ImageError as exc:
                        image_error = str(exc)
                await websocket.send_json({"type": "thinking"})
                if image_error or len(text) > MAX_UPLOAD_TEXT:
                    await websocket.send_json({"type": "response", **assistant._result(
                        session_id, "That upload could not be used. Paste up to 20,000 characters "
                                    "of recipe text, or choose a JPEG, PNG or WebP photo."),
                        "image_error": image_error})
                    continue
                logger.info("Recipe upload from %s: %d chars, photo: %s",
                            session_id, len(text), "yes" if image else "no")
                result = await assistant.import_recipe(session_id, text, image, on_delta=send_delta)
                await websocket.send_json({"type": "response", **result})
                continue

            message = (data.get("message") or "").strip()
            if not message:
                continue

            upload, frames, image_error = None, [], None
            try:
                if data.get("image_source") == "upload":
                    if not data.get("image"):
                        raise ImageError("No photo was received.")
                    upload = decode_image(data["image"], max_bytes)
                elif data.get("images"):
                    frames = decode_images(data["images"], max_bytes)
                elif data.get("image"):  # older pages send one camera frame
                    frames = [decode_image(data["image"], max_bytes)]
            except ImageError as exc:
                image_error = str(exc)
                logger.warning("Image from %s rejected: %s", session_id, exc)

            logger.info("Question from %s: %r (frames: %d, photo: %s)", session_id, message[:80],
                        len(frames), "yes" if upload else "no")
            await websocket.send_json({"type": "thinking"})

            if image_error and data.get("image_source") == "upload":
                await websocket.send_json({"type": "response", **assistant._result(
                    session_id, "The photo could not be attached. Choose another JPEG, PNG or WebP photo and try again."),
                    "image_error": image_error})
                continue

            result = await assistant.process_message(
                message, session_id, frames=frames, upload=upload, on_delta=send_delta)
            payload = {"type": "response", **result}
            if image_error:
                payload["image_error"] = image_error
            await websocket.send_json(payload)

    except WebSocketDisconnect:
        logger.info("Phone disconnected: %s", session_id)
    except Exception as exc:
        logger.error("WebSocket error for %s: %s", session_id, exc, exc_info=True)
    finally:
        assistant.store.reset_chat_binding(chat_token)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "LH Cooking Coach",
        # Which models are actually live. Without this the only way to confirm
        # a model switch reached a running server is to grep its logs.
        "model": assistant.model.describe(),
        "vision": assistant.vision.describe(),
        "knowledge": {
            "facts": assistant.knowledge.fact_count(),
            "recipes": len(assistant.knowledge.recipes()),
            "database": assistant.knowledge.db_path,
        },
    }


@app.get("/api/recipes")
async def list_recipes():
    """Every recipe the coach can walk through, with its step and fact counts."""
    return {"recipes": assistant.knowledge.recipes()}


@app.get("/api/recipes/{recipe_id}/steps")
async def recipe_steps(recipe_id: int):
    return {"steps": assistant.knowledge.steps(recipe_id)}


@app.get("/api/knowledge")
async def knowledge_stats():
    """What the coach can retrieve, broken down by facet and subject kind.

    A thin facet is the signal that the YAML needs filling in, not that
    retrieval is broken.
    """
    conn = sqlite3.connect(assistant.knowledge.db_path)
    try:
        return {
            "database": assistant.knowledge.db_path,
            "subjects_by_kind": dict(conn.execute(
                "SELECT kind, COUNT(*) FROM subjects GROUP BY kind")),
            "facts_by_facet": dict(conn.execute(
                "SELECT facet, COUNT(*) FROM facts GROUP BY facet ORDER BY 2 DESC")),
            "aliases": conn.execute("SELECT COUNT(*) FROM aliases").fetchone()[0],
        }
    finally:
        conn.close()


@app.get("/api/context/{user_id}")
async def session_context(user_id: str):
    """One cook's session: profile, recipe and step, episodes, facts told."""
    return assistant.get_user_insights(user_id)


@app.get("/api/retrieve")
async def retrieve_preview(q: str, recipe: str = None):
    """Dry-run retrieval for a question, without calling the model.

    The fastest way to tell a retrieval problem from a prompting one.
    `recipe` names the recipe to treat as being cooked.
    """
    from knowledge_base import classify_facet
    kb = assistant.knowledge
    cooking = kb.resolve_recipe(recipe) if recipe else None
    subject = kb.resolve_subject(q) or cooking
    facet = classify_facet(q)
    facts = kb.retrieve(subject, facet, query=q, limit=6, recipe=cooking)
    return {
        "query": q,
        "recipe": cooking.name if cooking else None,
        "subject": subject.name if subject else None,
        "subject_kind": subject.type if subject else None,
        "facet": facet,
        "facts": [{"facet": f["facet"], "subject": f["subject_name"],
                   "text": f["text"]} for f in facts],
    }


@app.post("/api/vision/test")
async def vision_test(request: Request):
    """Run the configured VLM on one image, outside a conversation.

    For trying a provider before wiring it into the app:

        {"image": "data:image/jpeg;base64,...", "question": "is this done?",
         "recipe": "Spaghetti Aglio e Olio", "step_text": "...", "done_when": "..."}
    """
    body = await request.json()
    if not assistant.vision.available:
        return JSONResponse(status_code=503, content={
            "error": "no VLM configured",
            "vision": assistant.vision.describe(),
        })
    try:
        raw, mime = decode_image(body.get("image"), assistant.vision.config.max_image_bytes)
    except ImageError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    observation = await assistant.vision.observe(raw, mime, VisionRequest(
        question=body.get("question") or "What do you see?",
        recipe=body.get("recipe"),
        step_text=body.get("step_text"),
        done_when=body.get("done_when"),
    ))
    if observation is None:
        return JSONResponse(status_code=502, content={
            "error": "the VLM returned nothing usable; see the server log",
            "vision": assistant.vision.describe(),
        })
    return {"observation": observation.to_dict(), "vision": assistant.vision.describe()}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    logger.info("LH Cooking Coach on http://localhost:%d", port)
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=port)
