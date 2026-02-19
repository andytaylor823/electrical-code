"""Merge sentences that were split across page boundaries by the OCR process."""


def sentence_runs_over(p1: str, p2: str) -> bool:
    """Determine whether text at the end of one page continues into the next.

    Returns True if the sentence appears to run over from p1 to p2,
    False if p2 starts a new sentence or section.
    """
    # If page 2 starts with e.g. "290.98", it's a new section number
    first_word = p2.split(" ")[0]
    try:
        float(first_word)
        return False
    except ValueError:
        pass

    # If p1 ends with sentence-ending punctuation, it's not a runover
    if p1[-1] in ".?!":
        return False

    # Multi-line content (e.g. formatted markdown tables) is never a runover
    if "\n" in p1 or "\n" in p2:
        return False

    # If p2 starts with a structural keyword, it's a new section
    structural_prefixes = ("(", "Informational", "Part", "Table", "Figure")
    if any(p2.startswith(prefix) for prefix in structural_prefixes):
        return False

    # All-caps lines are headers, not runovers
    if p1.isupper() or p2.isupper():
        return False

    return True


def resort_dict(d: dict[str, dict]) -> dict[str, dict]:
    """Re-index a dict with consecutive integer string keys, sorted by original key."""
    sorted_items = sorted(d.items(), key=lambda item: int(item[0]))
    return {str(new_key): value for new_key, (_, value) in enumerate(sorted_items)}


def run(paragraphs: dict[str, dict[str, str | int]]) -> dict:
    """Detect and merge sentences split across page boundaries."""
    # Loop over all paragraphs
    new_output = paragraphs.copy()
    paragraph_ix_page_stop, paragraph_ix_page_start = 99, 0
    skip_next = False
    for i in range(len(paragraphs)):
        if skip_next:
            skip_next = False
            continue

        content, page = paragraphs[str(i)].values()

        # Identify page start/stop markers
        if "article" in content.lower() and content.isupper():
            paragraph_ix_page_start = i
        if content == "2023 Edition NATIONAL ELECTRICAL CODE":
            paragraph_ix_page_stop = i
        if content == "NATIONAL ELECTRICAL CODE 2023 Edition":
            paragraph_ix_page_stop = i - 1

        # If we hit the start of a new page...
        if paragraph_ix_page_start > paragraph_ix_page_stop:
            p1 = paragraphs[str(paragraph_ix_page_stop - 1)]["content"]
            p2 = paragraphs[str(paragraph_ix_page_start + 1)]["content"]
            paragraph_ix_page_start = 0

            # and if we have a run-over sentence,
            if sentence_runs_over(p1, p2):
                # make the sentence whole again, and assign to the first paragraph/page slot
                new_output[str(paragraph_ix_page_stop - 1)] = {"content": p1 + " " + p2, "page": page - 1}

                # Remove tail end of the sentence
                del new_output[str(i + 1)]
                skip_next = True

    # Reorder dict with consecutive integer keys
    new_output = resort_dict(new_output)

    return new_output


if __name__ == "__main__":
    import json
    from pathlib import Path

    # Read in big paragraphs file
    root = Path(__file__).parent.parent.parent.parent.resolve()
    PARAGRAPHS_FILE = root / "data" / "raw" / "NFPA 70 NEC 2023_paragraphs.json"
    with open(PARAGRAPHS_FILE, "r", encoding="utf-8") as fopen:
        paragraphs = json.load(fopen)

    # Run cleaning
    output = run(paragraphs)

    # Output
    OUTPUT_FILE = root / "data" / "intermediate" / "NFPA 70 NEC 2023_cleaned_paragraphs.json"
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fopen:
        json.dump(output, fopen)
