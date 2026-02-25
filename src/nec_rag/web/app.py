"""FastAPI web server for the NEC expert agent.

Serves a ChatGPT-style chat interface backed by the LangGraph NEC agent.
Supports multi-turn conversations, image upload, and simple password auth.

Usage:
    python -m nec_rag.web.app
    # => Uvicorn running on http://localhost:8000
"""

import asyncio
import json
import logging
import os
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import AIMessageChunk, HumanMessage
from starlette.responses import StreamingResponse

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

# Context window size for the agent LLM (used by the frontend usage wheel)
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW_SIZE", "128000"))

# Directory for user-submitted feedback (conversation snapshots + comments)
FEEDBACK_DIR = ROOT / "data" / "feedback"

# ---------------------------------------------------------------------------
# In-memory state (ephemeral — lost on server restart)
# ---------------------------------------------------------------------------

_AGENT = None  # LangGraph compiled agent, initialised at startup
_sessions: dict[str, list] = {}  # session_id -> list[BaseMessage]
_auth_tokens: set[str] = set()  # valid auth cookie tokens


# ---------------------------------------------------------------------------
# Tool-call description mapping (present + past tense for shadow text)
# ---------------------------------------------------------------------------


def _describe_browse(args: dict) -> dict[str, str]:
    """Build present/past descriptions for browse_nec_structure."""
    article = args.get("article")
    if article is not None:
        return {"present": f"Inspecting Article {article}\u2019s structure", "past": f"Inspected Article {article}\u2019s structure"}
    chapter = args.get("chapter")
    if chapter is not None:
        return {"present": f"Browsing Chapter {chapter}\u2019s articles", "past": f"Browsed Chapter {chapter}\u2019s articles"}
    return {"present": "Browsing NEC table of contents", "past": "Browsed NEC table of contents"}


def _describe_lookup(args: dict) -> dict[str, str]:
    """Build present/past descriptions for nec_lookup."""
    section_ids = [s for s in (args.get("section_ids") or []) if s]
    table_ids = [t for t in (args.get("table_ids") or []) if t]
    parts: list[str] = []
    if section_ids:
        label = "Sections" if len(section_ids) > 1 else "Section"
        parts.append(f"{label} {', '.join(section_ids[:3])}")
    if table_ids:
        label = "Tables" if len(table_ids) > 1 else "Table"
        parts.append(f"{label} {', '.join(table_ids[:3])}")
    target = " and ".join(parts) if parts else "requested IDs"
    return {"present": f"Looking up {target}", "past": f"Looked up {target}"}


# Static descriptions for tools that don't depend on arguments
_STATIC_TOOL_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "rag_search": {"present": "Searching the NEC\u2026", "past": "Searched the NEC"},
    "explain_image": {"present": "Analyzing uploaded image\u2026", "past": "Analyzed uploaded image"},
}

# Tools whose descriptions depend on their arguments
_DYNAMIC_TOOL_DESCRIBERS: dict[str, callable] = {
    "browse_nec_structure": _describe_browse,
    "nec_lookup": _describe_lookup,
}


def _describe_tool_call(tool_name: str, args: dict) -> dict[str, str]:
    """Return ``{"present": ..., "past": ...}`` descriptions for a tool call."""
    if tool_name in _STATIC_TOOL_DESCRIPTIONS:
        return _STATIC_TOOL_DESCRIPTIONS[tool_name]
    describer = _DYNAMIC_TOOL_DESCRIBERS.get(tool_name)
    if describer:
        return describer(args)
    return {"present": f"Running {tool_name}\u2026", "past": f"Ran {tool_name}"}


def _sse_line(payload: dict) -> str:
    """Format a single SSE ``data:`` line (with trailing double-newline)."""
    return f"data: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# Agent streaming (runs in a background thread, pushes SSE events via queue)
# ---------------------------------------------------------------------------


def _process_agent_node(new_msgs: list, pending_descriptions: dict, put_fn: callable) -> None:
    """Handle an 'agent' node chunk — emit tool_start events for any tool calls."""
    for msg in new_msgs:
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            desc = _describe_tool_call(tc["name"], tc["args"])
            pending_descriptions[tc["id"]] = desc
            put_fn({"type": "tool_start", "description": desc["present"]})


def _process_tools_node(new_msgs: list, pending_descriptions: dict, put_fn: callable) -> None:
    """Handle a 'tools' node chunk — emit tool_end events, then a thinking event."""
    for msg in new_msgs:
        tc_id = getattr(msg, "tool_call_id", None)
        desc = pending_descriptions.pop(tc_id, None)
        if desc:
            put_fn({"type": "tool_end", "description": desc["past"]})
    # Agent will think again after receiving tool results
    put_fn({"type": "thinking"})


def _last_call_prompt_tokens(accumulated_messages: list) -> int:
    """Return prompt_tokens from the last AI message's metadata.

    This is the single-call prompt size reflecting actual context window usage
    (full history + all tool results), unlike the callback's cumulative total
    which sums across every LLM call in the invocation.

    Checks both ``response_metadata["token_usage"]`` (OpenAI-style) and
    ``usage_metadata`` (LangChain standard) since the available attribute
    depends on the LLM provider and streaming mode.
    """
    for msg in reversed(accumulated_messages):
        # OpenAI-style: response_metadata.token_usage.prompt_tokens
        rm_usage = getattr(msg, "response_metadata", {}).get("token_usage", {})
        if rm_usage:
            val = rm_usage.get("prompt_tokens", 0)
            if val:
                return val

        # LangChain standard: usage_metadata.input_tokens
        um = getattr(msg, "usage_metadata", None) or {}
        if isinstance(um, dict) and um.get("input_tokens"):
            return um["input_tokens"]

    return 0


def _stream_agent_thread(
    agent,
    messages: list,
    loop: asyncio.AbstractEventLoop,
    event_queue: asyncio.Queue,
    result_holder: dict,
) -> None:
    """Run ``agent.stream()`` synchronously and push SSE event strings into *event_queue*.

    *result_holder* is a mutable dict where the accumulated message list is
    stored under ``"messages"`` so the caller can update the session after the
    stream completes.
    """
    reset_vision_usage()
    accumulated_messages: list = []
    pending_descriptions: dict[str, dict[str, str]] = {}  # tool_call_id -> {present, past}

    def _put(payload: dict) -> None:
        loop.call_soon_threadsafe(event_queue.put_nowait, _sse_line(payload))

    # LangGraph's create_agent uses "model" for the LLM node and "tools" for tool execution
    node_handlers = {
        "model": lambda msgs: _process_agent_node(msgs, pending_descriptions, _put),
        "tools": lambda msgs: _process_tools_node(msgs, pending_descriptions, _put),
    }

    try:
        with get_openai_callback() as cb:
            # Stream both node-level updates (for tool events) and message-level
            # chunks (for token-by-token text deltas from the LLM)
            for mode, data in agent.stream({"messages": messages}, stream_mode=["messages", "updates"]):
                if mode == "updates":
                    for node_name, node_output in data.items():
                        new_msgs = node_output.get("messages", [])
                        accumulated_messages.extend(new_msgs)
                        handler = node_handlers.get(node_name)
                        if handler:
                            handler(new_msgs)
                elif mode == "messages":
                    if isinstance(data[0], AIMessageChunk) and isinstance(data[0].content, str) and data[0].content:
                        _put({"type": "text_delta", "content": data[0].content})

        # Build combined token info (agent LLM + standalone vision calls)
        vision = get_vision_usage()
        context_used = _last_call_prompt_tokens(accumulated_messages)

        token_info = {
            "prompt_tokens": cb.prompt_tokens + vision["prompt_tokens"],
            "completion_tokens": cb.completion_tokens + vision["completion_tokens"],
            "total_tokens": cb.total_tokens + vision["total_tokens"],
            "llm_calls": cb.successful_requests,
            "context_window": CONTEXT_WINDOW,
            "context_used": context_used,
        }

        # The last accumulated message is the final AI response
        final_content = ""
        if accumulated_messages:
            final_content = getattr(accumulated_messages[-1], "content", "") or ""
        _put({"type": "final", "response": final_content, "token_info": token_info})

        logger.info(
            "Stream complete — tokens=%d (prompt=%d, completion=%d) | context_used=%d/%d | LLM calls=%d",
            token_info["total_tokens"],
            token_info["prompt_tokens"],
            token_info["completion_tokens"],
            context_used,
            CONTEXT_WINDOW,
            token_info["llm_calls"],
        )

    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("Agent streaming error")
        _put({"type": "error", "detail": "The agent encountered an error processing your request."})

    # Store accumulated messages for session update by the caller
    result_holder["messages"] = list(messages) + accumulated_messages

    # Sentinel to signal the async generator that the stream is finished
    loop.call_soon_threadsafe(event_queue.put_nowait, None)


async def _sse_event_generator(agent, messages: list, session_id: str):
    """Async generator that yields SSE lines from the agent stream.

    Spawns the synchronous ``agent.stream()`` in a background thread and
    bridges events to the async world via an ``asyncio.Queue``.
    """
    loop = asyncio.get_running_loop()
    event_queue: asyncio.Queue = asyncio.Queue()
    result_holder: dict = {"messages": None}

    # Yield the initial "Thinking…" status before the thread starts producing
    yield _sse_line({"type": "thinking"})

    # Start the blocking agent stream in a daemon thread
    thread = threading.Thread(
        target=_stream_agent_thread,
        args=(agent, messages, loop, event_queue, result_holder),
        daemon=True,
    )
    thread.start()

    # Relay SSE events from the queue until the sentinel (None) arrives
    while True:
        event = await event_queue.get()
        if event is None:
            break
        yield event

    thread.join(timeout=5)

    # Persist the full conversation history (input + agent messages) into the session
    if result_holder["messages"] is not None:
        _sessions[session_id] = result_holder["messages"]
        logger.info("Session %s updated — %d messages total", session_id, len(result_holder["messages"]))


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
    """Handle a chat message: save images, stream SSE progress events, then the final response."""
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

    # Stream SSE events as the agent thinks and calls tools
    return StreamingResponse(
        _sse_event_generator(_AGENT, _sessions[session_id], session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
# Feedback
# ---------------------------------------------------------------------------


def _serialize_message(msg) -> dict:
    """Convert a LangChain BaseMessage into a plain dict for JSON export."""
    entry: dict = {"role": msg.type, "content": msg.content}
    # AI messages may carry tool calls with name, args, and id
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        entry["tool_calls"] = [{"name": tc["name"], "args": tc["args"], "id": tc["id"]} for tc in tool_calls]
    # Tool response messages reference the call they answered
    if msg.type == "tool":
        entry["tool_call_id"] = getattr(msg, "tool_call_id", None)
        entry["name"] = getattr(msg, "name", None)
    return entry


@app.post("/api/feedback")
async def submit_feedback(request: Request):
    """Save user feedback along with the full conversation snapshot to disk."""
    _check_auth(request)
    body = await request.json()

    session_id = body.get("session_id", "")
    feedback_text = (body.get("feedback_text") or "").strip()
    if not feedback_text:
        raise HTTPException(status_code=400, detail="Feedback text is required")

    # Serialize the conversation history (empty list if session has no messages yet)
    messages = _sessions.get(session_id, [])
    conversation = [_serialize_message(m) for m in messages]

    now = datetime.now(timezone.utc)
    payload = {
        "feedback_text": feedback_text,
        "session_id": session_id,
        "timestamp": now.isoformat(),
        "message_count": len(conversation),
        "conversation": conversation,
    }

    # Write to data/feedback/ with a timestamped filename
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_{session_id}.json"
    filepath = FEEDBACK_DIR / filename
    filepath.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Feedback saved: %s (%d messages, %d chars of feedback)", filename, len(conversation), len(feedback_text))
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
