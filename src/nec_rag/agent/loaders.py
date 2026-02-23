"""Data loaders for the NEC agent (structured JSON, section index).

Handles loading and caching of the structured NEC JSON and derived
lookup indices.  These are split from the general utility helpers to
keep data-loading concerns in their own module.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to the structured NEC JSON (chapter > article > part > subsection)
ROOT = Path(__file__).parent.parent.parent.parent.resolve()
STRUCTURED_JSON_PATH = ROOT / "data" / "prepared" / "NFPA 70 NEC 2023_structured.json"

# Module-level cache for expensive data structures
_CACHE: dict = {
    "structured_json": None,
    "section_index": None,
}


# ---------------------------------------------------------------------------
# Structured JSON loader
# ---------------------------------------------------------------------------


def load_structured_json() -> dict:
    """Load and cache the full NEC structured JSON (chapters, definitions, etc.).

    Returns the parsed dict from ``data/prepared/NFPA 70 NEC 2023_structured.json``.
    Subsequent calls return the cached object.
    """
    if _CACHE["structured_json"] is not None:
        return _CACHE["structured_json"]

    logger.info("Loading structured NEC data from %s", STRUCTURED_JSON_PATH)
    with open(STRUCTURED_JSON_PATH, "r", encoding="utf-8") as fopen:
        data: dict = json.load(fopen)

    _CACHE["structured_json"] = data
    logger.info("Structured data loaded: %d chapters", len(data.get("chapters", [])))
    return data


# ---------------------------------------------------------------------------
# Section index (section_id -> subsection dict with parent metadata)
# ---------------------------------------------------------------------------


def load_section_index() -> dict[str, dict]:
    """Build a lookup from section ID (e.g. '250.50') to its subsection dict.

    Each value is the original subsection dict from the structured JSON,
    augmented with ``article_num`` and ``article_title`` for convenience.
    """
    if _CACHE["section_index"] is not None:
        return _CACHE["section_index"]

    data = load_structured_json()
    index: dict[str, dict] = {}

    # Walk chapters -> articles -> parts -> subsections
    for chapter in data["chapters"]:  # pylint: disable=unsubscriptable-object
        for article in chapter["articles"]:
            for part in article["parts"]:
                for subsection in part["subsections"]:
                    entry = dict(subsection)
                    entry["article_num"] = article["article_num"]
                    entry["article_title"] = article["title"]
                    index[subsection["id"]] = entry

    _CACHE["section_index"] = index
    logger.info("Section index built: %d subsections", len(index))
    return index
