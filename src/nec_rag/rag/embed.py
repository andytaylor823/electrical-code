import json
import os
import re
import sys
from pathlib import Path

from openai import AzureOpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import tiktoken

BATCH_SIZE = 50
root = Path(__file__).parent.parent.parent.parent
load_dotenv(root / '.env')


# Create client
client = AzureOpenAI(
    api_key = os.getenv('AZURE_OPENAI_API_KEY'),
    azure_endpoint = os.getenv("EMBEDDINGS_SMALL_ENDPOINT_URL"),
    api_version = '2023-05-15'
)

def create_batches(single_array_to_embed):
    batch_size = BATCH_SIZE # handle case when it's perfectly divisible, still add 1
    while(len(single_array_to_embed) // batch_size) == 0: batch_size += 1
    batches = [
        single_array_to_embed[batch_size*i : batch_size*(i+1)]
        for i in range(len(single_array_to_embed) // batch_size + 1)
    ]
    return batches

def ntokens(s: str, model_name: str = 'gpt-4o') -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(s))
    return num_tokens


def split_into_sentences(text: str) -> list[str]:
    """Split the text into sentences using a regular expression."""
    # Basic sentence boundary using punctuation followed by space and uppercase letter
    return re.findall(r'[^.!?]*[.!?](?:\s|$)', text)

def halve_text_preserving_sentences(text: str) -> list[str]:
    """Split text roughly in half while preserving full sentences."""
    sentences = split_into_sentences(text)
    if len(sentences) <= 1:
        return [text[:len(text)//2], text[len(text)//2:]]

    total = len(sentences)
    mid = total // 2

    # Try to keep the halves balanced in token count
    left = ' '.join(sentences[:mid])
    right = ' '.join(sentences[mid:])
    return [left.strip(), right.strip()]


TOKEN_LIMIT = 8192
def recursive_halve_to_token_limit(text: str) -> list[str]:
    """Recursively halve text until all chunks are under the token limit."""
    if ntokens(text) <= TOKEN_LIMIT:
        return [text]

    chunk1, chunk2 = halve_text_preserving_sentences(text)
    result = []
    for chunk in (chunk1, chunk2):
        result.extend(recursive_halve_to_token_limit(chunk))
    return result


def embed_batch(batch: list[str]) -> list[list[float]]:
    # Verify each is under token limit
    tokens = [ntokens(s) for s in batch]
    if all(t <= TOKEN_LIMIT for t in tokens):
        response = client.embeddings.create(
            input=batch,
            model='text-embeddings-3-small'
        )
        output = [r.embedding for r in response.data]

    # If not all under token limit, have to handle that one separately
    else:
        output = []
        for s in batch:
            if ntokens(s) <= TOKEN_LIMIT:
                embedding = client.embeddings.create(
                    input=s,
                    model='text-embeddings-3-small'
                ).data[0].embedding
            else:
                substrings = recursive_halve_to_token_limit(s)
                embeddings_long = [client.embeddings.create(
                    input=substring,
                    model='text-embeddings-3-small'
                ).data[0].embedding
                    for substring in substrings
                ] # shape e.g. [2, N]
                # shape [N]
                embedding = [sum(column) / len(column) for column in zip(*embeddings_long)]

            output.append(embedding)

    return output

def main(how='sections'):

    # Load json file containing text
    json_file = root / 'vectors' / f'{how}.json'
    with open(json_file, 'r') as fopen:
        data = json.load(fopen)

    # Pull just text from nested JSON structure
    if how == 'sections': kw = 'section'
    elif kw == 'definitions': kw = 'term'
    single_array_to_embed = [dkt[kw] for dkt in data.values()]

    # Batch
    batches = create_batches(single_array_to_embed)

    # Embed by batches
    output = []
    for batch in tqdm(batches):
        output += embed_batch(batch)

    # Write out
    output_file = root / 'vectors' / f'{how}_vector.json'
    output_json = {i: output[i] for i in range(len(output))}
    with open(output_file, 'w') as fopen:
        json.dump(output_json, fopen)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        how = 'sections'
    else:
        how = sys.argv[1].strip()

    main(how)
