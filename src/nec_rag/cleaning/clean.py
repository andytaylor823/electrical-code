import json
from pathlib import Path

from nec_rag.cleaning import sentence_runover
from nec_rag.cleaning import hyphens_endline
from nec_rag.cleaning import remove_junk_pages

# Read in big paragraphs file
root = Path(__file__).parent.parent.parent.parent.resolve()
PARAGRAPHS_FILE = root / 'data' / 'intermediate' / 'NFPA 70 NEC 2023_paragraphs.json'
with open(PARAGRAPHS_FILE, 'r') as fopen:
    paragraphs = json.load(fopen)

# Do cleaning
output = paragraphs.copy()
output = remove_junk_pages.run(output)
output = sentence_runover.run(output)
output = hyphens_endline.run(output)

# Output
OUTPUT_FILE = PARAGRAPHS_FILE.parent / 'NFPA 70 NEC 2023_cleaned_paragraphs.json'
with open(OUTPUT_FILE, 'w') as fopen:
    json.dump(output, fopen)
