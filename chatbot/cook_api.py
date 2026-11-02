# cook_api.py - FastAPI entry point for the cooking coach
"""
Serves the phone web app and the one WebSocket it talks to.

The phone does speech both ways in the browser (Web Speech API for listening,
speechSynthesis for speaking), so the server only ever sees text plus an
optional camera frame:

    phone -> {"type": "ask", "message": "is this done?", "image": "data:image/jpeg;base64,..."}
    phone <- {"type": "response", "message": "...", "cooking": {...}, "vision": {...}}

Camera and microphone access in a mobile browser require HTTPS (localhost is
the only exception), which is why production runs behind Caddy.
"""

import asyncio
import logging
import os
import sqlite3
import uuid

from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from cook_main import CookingAssistant
from vision import ImageError, VisionRequest, decode_image

app = FastAPI(title="LH Cooking Coach API")

assistant = CookingAssistant()

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
if os.path.isdir(STATIC):
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC, "index.html"))


def _session_payload(session_id: str, message: str) -> dict:
    return {
        "type": "system",
        "client_id": session_id,
        "message": message,
        "profile": assistant.onboarding_state(session_id),
        "cooking": assistant.cooking_state(session_id),
        "vision_available": assistant.vision.available,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, client_id: str = Query(None)):
    """The phone's conversation channel.

    The phone keeps a random id in localStorage and sends it as ?client_id=,
    so reloading the page resumes the same session -- and the same recipe
    step -- instead of starting onboarding again.
    """
    await websocket.accept()
    session_id = (client_id or "").strip()[:64] or f"web_{uuid.uuid4().hex[:12]}"
    logger.info("Phone connected: %s", session_id)

    try:
        greeting = await asyncio.to_thread(assistant.start_visit, session_id)
        await websocket.send_json(_session_payload(session_id, greeting))

        while True:
            data = await websocket.receive_json()
            kind = data.get("type", "ask")

            if kind == "connect":
                requested = (data.get("client_id") or "").strip()[:64]
                if requested and requested != session_id:
                    session_id = requested
                    greeting = await asyncio.to_thread(assistant.start_visit, session_id)
                    await websocket.send_json(_session_payload(session_id, greeting))
                continue
            if kind == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            message = (data.get("message") or "").strip()
            if not message:
                continue

            image, image_error = None, None
            if data.get("image"):
                try:
                    image = decode_image(data["image"],
                                         assistant.vision.config.max_image_bytes)
                except ImageError as exc:
                    image_error = str(exc)
                    logger.warning("Frame from %s rejected: %s", session_id, exc)

            logger.info("Question from %s: %r (frame: %s)", session_id, message[:80],
                        f"{len(image[0])} bytes" if image else "none")
            await websocket.send_json({"type": "thinking"})

            result = await assistant.process_message(message, session_id, image=image)
            payload = {"type": "response", **result}
            if image_error:
                payload["image_error"] = image_error
            await websocket.send_json(payload)

    except WebSocketDisconnect:
        logger.info("Phone disconnected: %s", session_id)
    except Exception as exc:
        logger.error("WebSocket error for %s: %s", session_id, exc, exc_info=True)


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
