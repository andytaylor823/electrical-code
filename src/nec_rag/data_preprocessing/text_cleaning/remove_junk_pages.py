"""Remove pages outside the main NEC content range (pages 26-717, 1-indexed)."""

FIRST_REAL_PAGE = 26  # indexed from 1
LAST_REAL_PAGE = 717  # indexed from 1


def resort_dict(d: dict[str, dict]) -> dict[str, dict]:
    """Re-index a dict with consecutive integer string keys, sorted by original key."""
    sorted_items = sorted(d.items(), key=lambda item: int(item[0]))
    return {str(new_key): value for new_key, (_, value) in enumerate(sorted_items)}


def run(paragraphs: dict[str, dict[str, str | int]]) -> dict[str, dict]:
    """Keep only paragraphs on pages between FIRST_REAL_PAGE and LAST_REAL_PAGE."""
    # Loop over all paragraphs
    new_output = {}
    for key, val in paragraphs.items():
        # Check we're between the content-bounding pages
        page = int(val["page"])  # also indexed from 1
        if FIRST_REAL_PAGE <= page <= LAST_REAL_PAGE:
            # If so, store this page
            new_output[key] = val

    # Re-order the dict and return
    new_output = resort_dict(new_output)
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
