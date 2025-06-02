from openai import AzureOpenAI
from dotenv import load_dotenv
from pathlib import Path
import json
import os
import sys

# region -- setup
root = Path(__file__).parent.parent
load_dotenv(root / 'src' / '.env')
TOP_K = 20
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4.1")

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of the same length")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Cosine similarity is undefined for zero-length vectors")

    return dot_product / (norm1 * norm2)


# Load previously-computed vector embeddings
print('Loading vectors and text...')
vector_path = root / 'vectors' / 'sections_vector.json'
with open(vector_path, 'r') as fopen:
    vectors = json.load(fopen)

# Load sections text
sections_path = root / 'vectors' / 'sections.json'
with open(sections_path, 'r') as fopen:
    sections = json.load(fopen)

# Establish llm + embedding clients
llm_client = AzureOpenAI(  
    api_key = os.getenv('AZURE_OPENAI_API_KEY'),
    azure_endpoint = os.getenv('GPT_41_ENDPOINT_URL'),
    api_version = '2025-01-01-preview'
) 
embedding_client = AzureOpenAI(
    api_key = os.getenv('AZURE_OPENAI_API_KEY'),
    azure_endpoint = os.getenv("EMBEDDINGS_SMALL_ENDPOINT_URL"),
    api_version = '2023-05-15'
)

# Define prompt templates
answer_prompt_template = """You are a legal expert with domain expertise in electrical codes. Given the following context from the National
Electrical Code (NEC), answer the following question.

Answer concisely, and cite your source from the provided NEC text. If the answer is not found in the provided context, you may draw upon your training knowledge, but be sure to note that the answer was not found in the provided NEC context.

Question:
{question}

NEC text:
{context}
"""

verification_prompt_template = """
You are a fact-checker whose job is to evaluate the correctness of citations of a large language model (LLM). You will be provided:
(1) the QUESTION that the LLM was asked,
(2) the associated CONTEXT that was provided for the LLM to answer the question, and
(3) the LLM's ANSWER to the QUESTION, including its citations.

Your job is to respond with "all good" if the citations provided in the ANSWER match what is present in the CONTEXT, and if the ANSWER correctly answers the QUESTION. If these conditions are not met, identify where the errors occur and provide an updated citation and answer.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}
"""


# endregion


# ENTRY POINT!

# Get question from user
print()
user_prompt = "Enter a question to ask the RAG app:\n>> "
question = input(user_prompt)
if question.lower() == 'x': sys.exit(10) # skip gate

# Embed question
question_embedded = embedding_client.embeddings.create(
    input=question,
    model='text-embeddings-3-small'
).data[0].embedding

# Create cosine sim dict and sort
similarities = {
    key: cosine_similarity(question_embedded, val)
    for key, val in vectors.items()
}
sorted_keys = sorted(similarities, key=similarities.get, reverse=True)
sorted_sections = [sections[key] for key in sorted_keys]

# Create context and then prompt
context = '\n'.join([sorted_sections[i]['section'] for i in range(TOP_K)])
answer_prompt = answer_prompt_template.format(question=question, context=context)
chat_messages = [
    {
        'role': 'user',
        'content': answer_prompt
    }
]

# Ask LLM
print('Asking LLM question...')
response = llm_client.chat.completions.create(
    model=deployment,
    messages=chat_messages
)
answer = json.loads(response.to_json())['choices'][0]['message']['content']


# Create second prompt for source-citing verification
verification_prompt = verification_prompt_template.format(
    question=question,
    context=context,
    answer=answer
)
# some comment
chat_messages_2 = [
    {
        'role': 'user',
        'content': verification_prompt
    }
]

# Ask another LLM to verify
print('Asking follow-up verification of cited sources...')
response2 = llm_client.chat.completions.create(
    model=deployment,
    messages=chat_messages_2
)
answer2 = json.loads(response2.to_json())['choices'][0]['message']['content']

# Print output
print()
print('Initial answer:', answer)
print()
if 'all good' in answer2.lower():
    print('Initial provided answer was correct!')
else:
    print('Corrected:', answer2)