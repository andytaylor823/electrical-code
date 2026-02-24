"""FastAPI web server for the NEC expert agent.

Serves a ChatGPT-style chat interface backed by the LangGraph NEC agent.
Supports multi-turn conversations, image upload, and simple password auth.

Usage:
    python -m nec_rag.web.app
    # => Uvicorn running on http://localhost:8000
"""

import asyncio
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import HumanMessage

from nec_rag.agent.agent import build_nec_agent
from nec_rag.agent.tools import IMAGE_EXTENSIONS, get_vision_usage, reset_vision_usage

ROOT = Path(__file__).parent.parent.parent.parent.resolve()
load_dotenv(ROOT / ".env")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Shared password for gating access (set in .env or defaults to "nec2023")
APP_PASSWORD = os.getenv("NEC_APP_PASSWORD", "nec2023")

# Temp directory for uploaded images (cleaned up on process exit)
UPLOAD_DIR = Path(tempfile.mkdtemp(prefix="nec_uploads_"))

# Path to static frontend assets
STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# In-memory state (ephemeral — lost on server restart)
# ---------------------------------------------------------------------------

_AGENT = None  # LangGraph compiled agent, initialised at startup
_sessions: dict[str, list] = {}  # session_id -> list[BaseMessage]
_auth_tokens: set[str] = set()  # valid auth cookie tokens


# ---------------------------------------------------------------------------
# Agent invocation helper (runs in thread pool for async compat)
# ---------------------------------------------------------------------------


def _invoke_agent(agent, messages: list) -> tuple[dict, dict]:
    """Run the agent synchronously and return (result, token_info).

    Called via ``asyncio.to_thread`` so the FastAPI event loop is not blocked.
    Token tracking (LangChain callback + vision usage) is done in the same
    thread to avoid context-propagation issues.
    """
    reset_vision_usage()
    with get_openai_callback() as cb:
        result = agent.invoke({"messages": messages})

    vision = get_vision_usage()
    token_info = {
        "prompt_tokens": cb.prompt_tokens + vision["prompt_tokens"],
        "completion_tokens": cb.completion_tokens + vision["completion_tokens"],
        "total_tokens": cb.total_tokens + vision["total_tokens"],
        "llm_calls": cb.successful_requests,
    }
    return result, token_info


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Pre-warm the NEC agent on server startup."""
    global _AGENT  # pylint: disable=global-statement
    logger.info("Initializing NEC agent (this may take a moment)...")
    _AGENT = build_nec_agent()
    logger.info("NEC agent ready — listening for requests.")
    yield


app = FastAPI(title="NEC Code Expert", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _check_auth(request: Request) -> None:
    """Raise 401 if the request lacks a valid auth cookie."""
    token = request.cookies.get("nec_auth")
    if not token or token not in _auth_tokens:
        raise HTTPException(status_code=401, detail="Not authenticated")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the chat UI."""
    return HTMLResponse(STATIC_DIR.joinpath("index.html").read_text(encoding="utf-8"))


@app.post("/api/login")
async def login(request: Request):
    """Validate the shared password and issue an auth cookie."""
    body = await request.json()
    if body.get("password") != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")

    token = str(uuid.uuid4())
    _auth_tokens.add(token)

    response = JSONResponse({"ok": True})
    response.set_cookie("nec_auth", token, httponly=True, samesite="lax", max_age=86400)
    logger.info("Login successful — issued token %s…", token[:8])
    return response


@app.post("/api/chat")
async def chat(
    request: Request,
    session_id: str = Form(...),
    message: str = Form(""),
    images: list[UploadFile] = File(default=[]),
):
    """Handle a chat message: save images, invoke the agent, return the response."""
    _check_auth(request)

    # Save uploaded images to the temp directory
    image_paths: list[str] = []
    for img in images:
        if img.filename:
            suffix = Path(img.filename).suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                dest = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"
                content = await img.read()
                dest.write_bytes(content)
                image_paths.append(str(dest))
                logger.info("Saved uploaded image: %s (%.1f KB)", dest.name, len(content) / 1024)

    # Build user text with image attachment markers (same pattern as the CLI)
    user_text = message.strip()
    if not user_text and not image_paths:
        raise HTTPException(status_code=400, detail="Message or image required")
    if image_paths:
        attachments = ", ".join(image_paths)
        user_text += f"\n\n[Attached image(s): {attachments}]"

    # Initialise session if new
    if session_id not in _sessions:
        _sessions[session_id] = []
        logger.info("New session created: %s", session_id)

    # Append the user message to conversation history
    _sessions[session_id].append(HumanMessage(content=user_text))

    # Invoke the agent with the full conversation history
    try:
        result, token_info = await asyncio.to_thread(_invoke_agent, _AGENT, _sessions[session_id])
    except Exception:
        logger.exception("Agent error in session %s", session_id)
        # Roll back the user message so the session stays consistent
        _sessions[session_id].pop()
        raise HTTPException(status_code=500, detail="The agent encountered an error processing your request.") from None

    # Update session with the complete message history (includes tool calls/results)
    _sessions[session_id] = list(result["messages"])

    # Extract the final AI response
    final_message = result["messages"][-1]

    logger.info(
        "Chat response — session=%s | tokens=%d (prompt=%d, completion=%d) | LLM calls=%d",
        session_id,
        token_info["total_tokens"],
        token_info["prompt_tokens"],
        token_info["completion_tokens"],
        token_info["llm_calls"],
    )

    return JSONResponse(
        {
            "response": final_message.content,
            "token_info": token_info,
        }
    )


@app.post("/api/new-chat")
async def new_chat(request: Request):
    """Clear a session's conversation history to start fresh."""
    _check_auth(request)
    body = await request.json()
    session_id = body.get("session_id")
    if session_id and session_id in _sessions:
        del _sessions[session_id]
        logger.info("Session cleared: %s", session_id)
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main():
    """Start the web server via uvicorn."""
    import uvicorn  # pylint: disable=import-outside-toplevel

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
