"""Per-session conversation tracing and daily cost tracking.

Writes one JSON file per session to ``data/traces/{datetime}.json``.
Each user/agent exchange appends a *turn* containing the new messages,
token usage, latency, and estimated USD cost.  The session_id is stored
inside the JSON but the filename is based on the creation timestamp.

Pricing defaults target Azure OpenAI GPT-5-mini and can be overridden via env vars:
    NEC_PRICE_INPUT_PER_1K   — USD per 1 000 input tokens  (default 0.00025)
    NEC_PRICE_OUTPUT_PER_1K  — USD per 1 000 output tokens (default 0.002)

Daily cost tracking keeps a running in-memory total of USD spent today
(local server time).  Call ``rebuild_daily_cost()`` on startup to seed
from existing trace files, then ``get_daily_cost()`` on each request.
"""

import json
import logging
import os
import threading
from datetime import date, datetime, timezone
from pathlib import Path

from dateutil.parser import isoparse
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent.parent.parent.resolve()
load_dotenv(ROOT / ".env")

logger = logging.getLogger(__name__)

# Trace storage directory (gitignored, separate from feedback)
TRACES_DIR = ROOT / "data" / "traces"

# Configurable per-token pricing (USD per 1 000 tokens)
PRICE_INPUT_PER_1K = float(os.getenv("NEC_PRICE_INPUT_PER_1K", "0.00025"))
PRICE_OUTPUT_PER_1K = float(os.getenv("NEC_PRICE_OUTPUT_PER_1K", "0.002"))

# Maps session_id -> filename so follow-up turns append to the same file
_session_trace_files: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Daily cost tracking (in-memory, rebuilt from trace files on startup)
# ---------------------------------------------------------------------------

_cost_lock = threading.Lock()
_daily_cost: dict = {"date": date.today(), "total_usd": 0.0}


def serialize_message(msg) -> dict:
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


def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Return estimated USD cost from token counts and configured pricing."""
    return (prompt_tokens / 1000.0) * PRICE_INPUT_PER_1K + (completion_tokens / 1000.0) * PRICE_OUTPUT_PER_1K


def record_turn(
    session_id: str,
    user_message: str,
    new_messages: list,
    token_info: dict,
    latency_seconds: float,
) -> None:
    """Append a turn to the session's trace file on disk.

    Creates the trace file if it doesn't exist yet.  Each call adds one
    entry to the ``turns`` array with the new messages and metadata from
    this exchange.  The filename is based on the creation timestamp
    (not the session_id); a module-level mapping keeps track of which
    file belongs to which session.

    Parameters
    ----------
    session_id:
        Frontend-generated session identifier.
    user_message:
        The raw user text submitted to ``/api/chat``.
    new_messages:
        LangChain BaseMessage objects produced by the agent in this turn
        (includes the user's HumanMessage plus all AI/tool messages).
    token_info:
        Dict with ``prompt_tokens``, ``completion_tokens``,
        ``total_tokens``, ``llm_calls``, ``context_used``, and
        ``context_window`` — already computed by the streaming thread.
    latency_seconds:
        Wall-clock time the agent took to produce its response.
    """
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)

    # Resolve the trace file for this session (create or reuse)
    existing_filename = _session_trace_files.get(session_id)
    if existing_filename:
        filepath = TRACES_DIR / existing_filename

    # First turn for this session — create a new datetime-named file
    if not existing_filename or not filepath.exists():
        filename = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}.json"
        filepath = TRACES_DIR / filename
        _session_trace_files[session_id] = filename

    # Load existing trace or create a fresh skeleton
    trace = None
    if filepath.exists():
        try:
            trace = json.loads(filepath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt trace file %s — starting fresh", filepath.name)

    if trace is None:
        trace = {
            "session_id": session_id,
            "created_at": now.isoformat(),
            "turns": [],
        }

    # Build the turn payload
    prompt_tokens = token_info.get("prompt_tokens", 0)
    completion_tokens = token_info.get("completion_tokens", 0)
    cost_usd = calculate_cost(prompt_tokens, completion_tokens)

    turn = {
        "turn_index": len(trace["turns"]),
        "user_message": user_message,
        "user_timestamp": token_info.get("user_timestamp", now.isoformat()),
        "response_timestamp": now.isoformat(),
        "agent_latency_seconds": round(latency_seconds, 2),
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": token_info.get("total_tokens", 0),
            "llm_calls": token_info.get("llm_calls", 0),
        },
        "context_used": token_info.get("context_used", 0),
        "context_window": token_info.get("context_window", 0),
        "cost_usd": round(cost_usd, 6),
        "messages": [serialize_message(m) for m in new_messages],
    }

    trace["turns"].append(turn)

    # Persist to disk
    filepath.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")

    # Update the in-memory daily cost counter
    _increment_daily_cost(cost_usd)

    logger.info(
        "Trace recorded — session=%s turn=%d latency=%.1fs cost=$%.4f tokens=%d | daily total=$%.4f",
        session_id,
        turn["turn_index"],
        latency_seconds,
        cost_usd,
        turn["token_usage"]["total_tokens"],
        get_daily_cost(),
    )


# ---------------------------------------------------------------------------
# Daily cost helpers
# ---------------------------------------------------------------------------


def _local_date_from_utc(utc_iso: str) -> date:
    """Parse a UTC ISO-8601 timestamp and return the local-time date."""
    try:
        dt = isoparse(utc_iso)
        return dt.astimezone().date()
    except (ValueError, TypeError):
        return date.today()


def _increment_daily_cost(cost_usd: float) -> None:
    """Add *cost_usd* to today's running total (thread-safe)."""
    with _cost_lock:
        today = date.today()
        if _daily_cost["date"] != today:
            # Day rolled over — reset the counter
            _daily_cost["date"] = today
            _daily_cost["total_usd"] = 0.0
        _daily_cost["total_usd"] += cost_usd


def get_daily_cost() -> float:
    """Return the running USD cost for today (local server time).

    If the date has rolled over since the last call, the counter
    resets automatically.
    """
    with _cost_lock:
        today = date.today()
        if _daily_cost["date"] != today:
            _daily_cost["date"] = today
            _daily_cost["total_usd"] = 0.0
        return _daily_cost["total_usd"]


def rebuild_daily_cost() -> None:
    """Scan all trace files and sum today's costs into the in-memory counter.

    Called once at server startup so the counter survives restarts.
    """
    today = date.today()
    total = 0.0

    if not TRACES_DIR.exists():
        logger.info("No traces directory yet — daily cost starts at $0.00")
        with _cost_lock:
            _daily_cost["date"] = today
            _daily_cost["total_usd"] = 0.0
        return

    trace_files = list(TRACES_DIR.glob("*.json"))
    for filepath in trace_files:
        try:
            trace = json.loads(filepath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        for turn in trace.get("turns", []):
            ts = turn.get("response_timestamp", "")
            if _local_date_from_utc(ts) == today:
                total += turn.get("cost_usd", 0.0)

    with _cost_lock:
        _daily_cost["date"] = today
        _daily_cost["total_usd"] = total

    logger.info("Daily cost rebuilt from %d trace files — today's total: $%.4f", len(trace_files), total)
