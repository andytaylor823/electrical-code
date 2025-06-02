def sentence_runs_over(p1: str, p2: str) -> bool:
    # If page 2 starts with e.g. "290.98", it's not the same sentence
    first_word = p2.split(' ')[0]
    try:
        x = float(first_word)
        return False
    except:
        pass

    # If any of these conditions trigger, it's not the same sentence
    if p1.endswith('.') or p1.endswith('?') or p1.endswith('!'): return False

    if p2.startswith('('): return False
    if p2.startswith('Informational'): return False
    if p2.startswith('Part'): return False
    if p2.startswith('Table'): return False
    if p2.startswith('Figure'): return False

    if p1.isupper() or p2.isupper(): return False

    return True

def resort_dict(d: dict[str, dict]) -> dict[str, dict]:
    sorted_items = sorted(d.items(), key=lambda item: int(item[0]))
    return {str(new_key): value for new_key, (_, value) in enumerate(sorted_items)}


def run(paragraphs: dict[str, dict[str, str | int]]) -> dict:
    # Loop over all paragraphs
    new_output = paragraphs.copy()
    paragraph_ix_page_stop, paragraph_ix_page_start = 99, 0
    skip_next = False
    for i in range(len(paragraphs)):
        if skip_next:
            skip_next = False
            continue

        content, page = paragraphs[str(i)].values()

        # Identify page start/stop
        #if content.lower().startswith('article') and content.isupper():
        if 'article' in content.lower() and content.isupper():
            paragraph_ix_page_start = i    
        if '2023 Edition NATIONAL ELECTRICAL CODE' == content:
            paragraph_ix_page_stop = i
        if 'NATIONAL ELECTRICAL CODE 2023 Edition' == content:
            paragraph_ix_page_stop = i-1

        # If we hit the start of a new page...
        if paragraph_ix_page_start > paragraph_ix_page_stop:
            p1 = paragraphs[str(paragraph_ix_page_stop-1)]['content']
            p2 = paragraphs[str(paragraph_ix_page_start+1)]['content']
            paragraph_ix_page_start = 0

            # and if we have a run-over sentence, 
            if sentence_runs_over(p1, p2):
                # make the sentence whole again, and assign to the first paragraph/page slot
                new_output[str(paragraph_ix_page_stop-1)] = {
                    'content': p1 + ' ' + p2,
                    'page': page-1
                }

                # Remove tail end of the sentence
                del new_output[str(i+1)]
                skip_next = True
    
    # Reorder dict with consecutive integer keys
    new_output = resort_dict(new_output)

    return new_output

if __name__ == '__main__':
    import json
    from pathlib import Path

    # Read in big paragraphs file
    root = Path(__file__).parent.parent.parent.resolve() / 'NFPA 70 NEC 2023'
    PARAGRAPHS_FILE = str(root) + '_paragraphs.json'
    with open(PARAGRAPHS_FILE, 'r') as fopen:
        paragraphs = json.load(fopen)

    # Run cleaning
    output = run(paragraphs)

    # Output
    OUTPUT_FILE = str(root) + '_cleaned_paragraphs.json'
    with open(OUTPUT_FILE, 'w') as fopen:
        json.dump(output, fopen)