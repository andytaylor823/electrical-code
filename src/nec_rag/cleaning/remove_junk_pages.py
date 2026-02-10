FIRST_REAL_PAGE = 26 # indexed from 1
LAST_REAL_PAGE = 717 # indexed from 1

def resort_dict(d: dict[str, dict]) -> dict[str, dict]:
    sorted_items = sorted(d.items(), key=lambda item: int(item[0]))
    return {str(new_key): value for new_key, (_, value) in enumerate(sorted_items)}

def run(paragraphs: dict[str, dict[str, str|int]]):
    # Loop over all paragraphs
    new_output = {}
    for key, val in paragraphs.items():

        # Check we're between the content-bounding pages
        page = int(val['page']) # also indexed from 1
        if page >= FIRST_REAL_PAGE and page <= LAST_REAL_PAGE:
            # If so, store this page
            new_output[key] = val

    # Re-order the dict and return
    new_output = resort_dict(new_output)
    return new_output



if __name__ == '__main__':
    import json
    from pathlib import Path

    # Read in big paragraphs file
    root = Path(__file__).parent.parent.parent.parent.resolve()
    PARAGRAPHS_FILE = root / 'data' / 'intermediate' / 'NFPA 70 NEC 2023_paragraphs.json'
    with open(PARAGRAPHS_FILE, 'r') as fopen:
        paragraphs = json.load(fopen)

    # Run cleaning
    output = run(paragraphs)

    # Output
    OUTPUT_FILE = root / 'data' / 'intermediate' / 'NFPA 70 NEC 2023_cleaned_paragraphs.json'
    with open(OUTPUT_FILE, 'w') as fopen:
        json.dump(output, fopen)
