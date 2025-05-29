from openai import AzureOpenAI
from dotenv import load_dotenv
import os
load_dotenv('src/.env')

client = AzureOpenAI(
    api_key = os.getenv('AZURE_OPENAI_API_KEY'),
    azure_endpoint = os.getenv('EMBEDDINGS_SMALL_ENDPOINT_URL'),
    api_version = '2023-05-15'
)

# Try vector embeddings
response = client.embeddings.create(
    input=["Your text string goes here"],
    model="text-embedding-3-small"
)

print(len(response.data[0].embedding))

x = """

Model	~ Pages per dollar	Performance on MTEB eval	Max input
text-embedding-3-small	62,500	62.3%	8192
text-embedding-3-large	9,615	64.6%	8192
text-embedding-ada-002	12,500	61.0%	8192
"""