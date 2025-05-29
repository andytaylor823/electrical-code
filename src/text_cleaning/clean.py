import sentence_runover
import hyphens_endline
import remove_junk_pages

import json
from pathlib import Path

# Read in big paragraphs file
root = Path(__file__).parent.parent.parent.resolve() / 'NFPA 70 NEC 2023'
PARAGRAPHS_FILE = str(root) + '_paragraphs.json'
with open(PARAGRAPHS_FILE, 'r') as fopen:
    paragraphs = json.load(fopen)

# Do cleaning
output = paragraphs.copy()
output = remove_junk_pages.run(output)
output = hyphens_endline.run(output)
output = sentence_runover.run(output)

# Output
OUTPUT_FILE = str(root) + '_cleaned_paragraphs.json'
with open(OUTPUT_FILE, 'w') as fopen:
    json.dump(output, fopen)