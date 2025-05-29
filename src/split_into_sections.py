import json
from pathlib import Path

# Read in cleaned paragraphs file
root = Path(__file__).parent.parent.parent.resolve() / 'NFPA 70 NEC 2023'
PARAGRAPHS_FILE = str(root) + '_cleaned_paragraphs.json'
with open(PARAGRAPHS_FILE, 'r') as fopen:
    paragraphs = json.load(fopen)


# Handle introduction
INTRO_START_PARAGRAPH_TEXT = "ARTICLE 90 Introduction"

# Handle definitions
DEFINITIONS_START_PARAGRAPH_TEXT = "ARTICLE 100 Definitions"

# Handle every other article after this
OTHER_ARTICLES_START_PARAGRAPH_TEXT = "ARTICLE 110 General Requirements for Electrical Installation"