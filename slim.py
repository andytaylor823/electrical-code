import json
root = 'NFPA 70 NEC 2023'

# Read in current file
TEXT_FILE = root + '.txt'
PARAGRAPHS_FILE = root + '_paragraphs.json'
with open(PARAGRAPHS_FILE, 'r') as fopen:
    dkt = json.load(fopen)

# Splice out only the meaningful pages to reduce token count
PAGE_START = 25 # indexing from 0
PAGE_STOP = 716 # inclusive, indexing from 0

output_text = ''
for p in dkt.values():
    content, page = p.values()
    if page >= PAGE_START and page <= PAGE_STOP:
        output_text += content + '\n'

# Write to file
with open(root+'_slim.txt', 'w') as fopen:
    fopen.write(output_text)