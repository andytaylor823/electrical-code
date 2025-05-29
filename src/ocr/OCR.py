import os
import json
import time
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient, models
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.resolve() / '.env'
load_dotenv(env_path)

root = Path(__file__).parent.parent.parent.resolve() / 'NFPA 70 NEC 2023'
#PDF_FILE = root + '_sample.pdf'
PDF_FILE = root.with_suffix('.pdf')

def get_client() -> DocumentIntelligenceClient:
    # Initialize OCR client
    key = os.getenv('AZURE_DOCUMENT_INTELLIGENCE_KEY')
    endpoint = os.getenv('DOCUMENT_INTELLIGENCE_ENDPOINT_URL')
    if key is None or endpoint is None:
        raise ValueError(f"Must provide both key ({key}) and endpoint ({endpoint})")

    client  = DocumentIntelligenceClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )

    return client

def run_OCR(client: DocumentIntelligenceClient) -> models.AnalyzeResult:
    # Load PDF and do OCR (about one second per page?)
    # about 200 seconds for the whole thing
    START = time.time()
    print('Starting OCR')
    with open(PDF_FILE, 'rb') as fopen:
        poller = client.begin_analyze_document('prebuilt-read', fopen)
    result = poller.result()
    print('Finished OCR')
    print(f'OCR took {time.time()-START:.2f} seconds')
    return result

def save_text(result: models.AnalyzeResult):
    # Save full text
    content = result.content.encode('charmap', errors='ignore').decode('charmap') # remove special characters
    with open(PDF_FILE.with_suffix('.txt'), 'w') as fopen:
        fopen.write(content)

    # Save by paragraph
    output_json = {
        i: {
            'content': paragraph['content'].encode('charmap', errors='ignore').decode('charmap'),
            'page': paragraph['boundingRegions'][0]['pageNumber']
        }
        for i, paragraph in enumerate(result.paragraphs)
    }
    with open(PDF_FILE.with_suffix('.pdf', '_paragraphs.json'), 'w') as fopen:
        json.dump(output_json, fopen)

if __name__ == '__main__':
    client = get_client()
    result = run_OCR(client)
    save_text(result)
