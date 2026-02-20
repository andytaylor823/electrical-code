"""Compiled regex patterns and constant tuples for NEC table detection.

These patterns identify structural elements in OCR-extracted NEC paragraph
data: table titles, section boundaries, page markers, footnotes, and
continuation signals.  Used by classifiers.py and detection.py.
"""

import re

# ─── Table Patterns ───────────────────────────────────────────────────────────

# Table title such as "Table 400.5(A)(1) Ampacity for ..."
TABLE_TITLE_RE = re.compile(r"^Table \d+\.\d+")

# Extract the table identifier (e.g. "Table 400.5(A)(1)") from a title string
TABLE_ID_RE = re.compile(r"(Table \d+\.\d+(?:\s*\([^)]*\))*)")


# ─── Section / Page Structure Patterns ────────────────────────────────────────

# Section number alone at page top, e.g. "400.48" or "90.1".
# NEC article numbers are always 2+ digits (70, 90, 100, ..., 840),
# so require at least 2 digits before the dot to avoid false-positives
# on table data like "2.79" or "3.05".
SECTION_NUM_ONLY_RE = re.compile(r"^\d{2,}\.\d+$")

# Section number followed by title text, e.g. "400.10 Uses Permitted."
SECTION_WITH_TEXT_RE = re.compile(r"^\d+\.\d+ [A-Z]")

# Page number footer like "70-284"
PAGE_NUM_RE = re.compile(r"^70-\d+$")

# Part header like "Part III."
PART_HEADER_RE = re.compile(r"^Part [IVX]+\.")


# ─── Cell / Footnote Patterns ────────────────────────────────────────────────

# Pure number (integer, decimal, or fraction like "1/0")
PURE_NUMBER_RE = re.compile(r"^-?\d+$|^-?\d+\.\d+$|^\d+/\d+$")

# Footnote start characters (digits, superscripts, special markers)
FOOTNOTE_START_RE = re.compile(r"^[\d\u00b9\u00b2\u00b3\u2070-\u2079?'+*]")


# ─── String-Match Constants ───────────────────────────────────────────────────

# Known page-marker prefixes (headers, footers, copyright, watermarks)
PAGE_MARKER_PREFIXES = (
    "2023 Edition NATIONAL ELECTRICAL CODE",
    "NATIONAL ELECTRICAL CODE 2023 Edition",
    "Copyright @NFPA",
    "Copyright @ NFPA",
    "EDUFIRE.IR",
    "Telegram: EDUFIRE.IR",
)

# Words found in table footnotes that reference table structure
TABLE_REFERENCE_WORDS = ("Column ", "column ", "subheading ", "ampacit")

# Preposition / conjunction endings that signal an interrupted sentence
CONTINUATION_ENDINGS = (
    " of",
    " or",
    " and",
    " the",
    " a",
    " an",
    " to",
    " in",
    " for",
    " with",
    " by",
    " from",
    " at",
    " on",
    " that",
    " which",
    " where",
    " as",
    " is",
    " are",
    " was",
    " were",
    " be",
    " but",
    " than",
    " not",
)
