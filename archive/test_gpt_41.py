import json
import os
from pathlib import Path

from openai import AzureOpenAI
from dotenv import load_dotenv

root = Path(__file__).parent.parent.resolve()
load_dotenv(root / '.env')
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4.1")

# Initialize Azure OpenAI Service client with key-based authentication
client = AzureOpenAI(
    api_key = os.getenv('AZURE_OPENAI_API_KEY'),
    azure_endpoint = os.getenv('GPT_41_ENDPOINT_URL'),
    api_version = '2025-01-01-preview'
)

# Prepare the chat prompt
chat_prompt = [
    {
        "role": "user",
        "content": "What color is the sky?"
    }
]

# Include speech result if speech is enabled
messages = chat_prompt

# Generate the completion
completion = client.chat.completions.create(
    model=deployment,
    messages=messages,
)

x = json.loads(completion.to_json())
print(x['choices'][0]['message']['content'])
