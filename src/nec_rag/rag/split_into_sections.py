import json
from pathlib import Path
import re

MAIN_PATTERN = r"^[0-9]+\.[0-9]+ "


def starts_new_section_main(content: str) -> bool:
    if "ARTICLE" in content:
        return False
    if re.match(MAIN_PATTERN, content):
        return True
    return False


def starts_new_section_def(content: str) -> bool:
    if content.startswith("Informational"):
        return False
    if content.startswith("("):
        return False
    return True


def steps_past_page_end(content: str) -> int | None:
    if content == "2023 Edition NATIONAL ELECTRICAL CODE":
        return 1
    if content == "NATIONAL ELECTRICAL CODE 2023 Edition":
        return 2
    return None


def steps_before_page_start(content: str) -> int | None:
    if "ARTICLE" in content and content.isupper():
        return 1
    return None


def handle_line(line: str, starts_new_section_fn, sections: list[str], current_section: list[str], skip_mode: bool):
    # If we start a new section (e.g. "200.10"), append the old one and empty it
    if starts_new_section_fn(line):
        if len(current_section) > 0:
            sections.append("\n".join(current_section).strip())
            current_section = []

    # Check if we're at the first line of a page's endmatter
    if steps_past_page_end(line) == 1:
        skip_mode = True

    # Check if we're at the second line of a page's endmatter
    if steps_past_page_end(line) == 2:
        # If so, remove the last element from the current section
        if len(current_section) > 0:
            current_section = current_section[:-1]
        skip_mode = True

    # If we're not skipping these lines b/c of page end/front-matter, append
    if not skip_mode:
        current_section.append(line)

    # If we hit the last line of frontmatter, undo skip mode
    if skip_mode and steps_before_page_start(line):
        skip_mode = False

    return sections, current_section, skip_mode


# Read in cleaned paragraphs file
root = Path(__file__).parent.parent.parent.parent.resolve()
PARAGRAPHS_FILE = root / "data" / "intermediate" / "NFPA 70 NEC 2023_clean.json"
with open(PARAGRAPHS_FILE, "r") as fopen:
    paragraphs = json.load(fopen)


# Handle introduction
INTRO_START_PARAGRAPH_TEXT = "ARTICLE 90 Introduction"

# Handle definitions
# DEFINITIONS_START_PARAGRAPH_TEXT = "ARTICLE 100 Definitions"
DEFINITIONS_START_PARAGRAPH_TEXT = "Accessible (as applied to equipment)"

# Handle every other article after this
# OTHER_ARTICLES_START_PARAGRAPH_TEXT = "ARTICLE 110 General Requirements for Electrical Installations"
OTHER_ARTICLES_START_PARAGRAPH_TEXT = "110.1 Scope. "


def main():
    in_pre = True
    in_intro = False
    in_def = False
    in_main = False

    definitions = []
    sections = []
    current_section = []
    skip_mode = False

    # Loop over every line
    for i in range(len(paragraphs)):
        line, page = paragraphs[str(i)].values()
        if line.startswith(INTRO_START_PARAGRAPH_TEXT):
            in_pre = False
            in_intro = True
        if line.startswith(DEFINITIONS_START_PARAGRAPH_TEXT):
            in_intro = False
            in_def = True
        if line.startswith(OTHER_ARTICLES_START_PARAGRAPH_TEXT):
            in_def = False
            in_main = True
            if len(current_section) > 0:
                definitions.append("\n".join(current_section).strip())
                current_section = []

        # We skip the pre and intro sections
        if in_pre or in_intro:
            continue

        # Process lines differently if in definitions vs main
        if in_def:
            definitions, current_section, skip_mode = handle_line(line, starts_new_section_def, definitions, current_section, skip_mode)
        elif in_main:
            sections, current_section, skip_mode = handle_line(line, starts_new_section_main, sections, current_section, skip_mode)
        else:
            raise ValueError("We should never get here")

    if len(current_section) > 0:
        sections.append("\n".join(current_section).strip())

    return definitions, sections


def write(definitions, sections):
    # When done, export to....JSON?
    def get_term(definition: str) -> str:
        return definition.split(".")[0]

    def get_section_id(section: str) -> str:
        try:
            return re.match(MAIN_PATTERN, section).group().strip()
        except:
            # This happens with tables -- these are not handled well currently
            # See Table 315.10(C), for example, page 207-ish
            # print(section)
            # print()
            # raise
            return None

    defs_dkt = {i: {"term": get_term(definition), "content": definition} for i, definition in enumerate(definitions)}

    sections_dkt = {i: {"id": get_section_id(section), "section": section} for i, section in enumerate(sections)}

    for i in sections_dkt.keys():
        if sections_dkt[i]["id"] is None:
            sections_dkt[i]["id"] = sections_dkt[i - 1]["id"]

    defs_output_file = root / "vectors" / "definitions.json"
    sections_output_file = root / "vectors" / "sections.json"

    with open(defs_output_file, "w") as fopen:
        json.dump(defs_dkt, fopen)
    with open(sections_output_file, "w") as fopen:
        json.dump(sections_dkt, fopen)


if __name__ == "__main__":
    definitions, sections = main()
    write(definitions, sections)
