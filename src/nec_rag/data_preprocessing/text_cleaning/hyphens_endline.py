"""Remove end-of-line hyphenation artifacts (e.g. 'electri- cal' -> 'electrical')."""

import re

# Matches a letter followed by '- ' (hyphen-space), indicating a broken word
HYPHEN_PATTERN = r"[A-Za-z]- "


def run(paragraphs: dict[str, dict]) -> dict[str, dict]:
    """Apply hyphenation fix to every paragraph's content."""
    # Loop over all paragraphs
    new_output = {}
    for i in range(len(paragraphs)):
        content, page = paragraphs[str(i)].values()

        # Call re.sub to sub in new text for old -- does nothing to strings with no match
        new_content = re.sub(HYPHEN_PATTERN, lambda m: m.group(0)[0], content)

        new_output[str(i)] = {"content": new_content, "page": page}

    return new_output


if __name__ == "__main__":
    import json
    from pathlib import Path

    # Read in big paragraphs file
    root = Path(__file__).parent.parent.parent.parent.parent.resolve()
    PARAGRAPHS_FILE = root / "data" / "raw" / "NFPA 70 NEC 2023_paragraphs.json"
    with open(PARAGRAPHS_FILE, "r", encoding="utf-8") as fopen:
        paragraphs = json.load(fopen)

    # Run cleaning
    output = run(paragraphs)

    # Output
    OUTPUT_FILE = root / "data" / "intermediate" / "NFPA 70 NEC 2023_cleaned_paragraphs.json"
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fopen:
        json.dump(output, fopen)
