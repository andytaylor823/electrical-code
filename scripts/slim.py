import json
from pathlib import Path

root = Path(__file__).parent.parent.resolve()
PARAGRAPHS_FILE = root / 'data' / 'intermediate' / 'NFPA 70 NEC 2023_paragraphs.json'

# Read in current file
with open(PARAGRAPHS_FILE, 'r') as fopen:
    dkt = json.load(fopen)

# Splice out only the meaningful pages to reduce token count
PAGE_START = 25 # indexing from 0
PAGE_STOP = 716 # inclusive, indexing from 0

output_text = ''
for p in dkt.values():
    content, page = p.values()
    content = content.encode('charmap', errors='ignore').decode('charmap') # ignore confusing characters
    if page >= PAGE_START and page <= PAGE_STOP:
        output_text += content + '\n'

# Write to file
OUTPUT_FILE = root / 'data' / 'intermediate' / 'NFPA 70 NEC 2023_slim.txt'
with open(OUTPUT_FILE, 'w') as fopen:
    fopen.write(output_text)
