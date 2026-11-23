"""Run an isolated, offline UI preview at http://127.0.0.1:8001.

Uses disposable recipes/profile data and disables model calls. The normal app
and its saved recipes are unaffected. Stop with Ctrl+C when run in a terminal.
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chatbot"))
with tempfile.TemporaryDirectory(prefix="stovetop-preview-") as folder:
    os.environ.update(MEMORY_DB_PATH=str(Path(folder) / "memory.db"),
                      KNOWLEDGE_DB_PATH=str(Path(folder) / "knowledge.db"),
                      OPENAI_API_KEY="", VLM_PROVIDER="none", WEB_SEARCH_ENABLED="false")
    with patch("dotenv.load_dotenv"):
        import cook_api
    import uvicorn
    uvicorn.run(cook_api.app, host="127.0.0.1", port=8001)
