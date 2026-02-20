"""
OCR script for the NFPA 70 NEC 2023 PDF.

This script uses Azure Document Intelligence to extract text from the raw PDF.
It produces two output files in data/raw/:
  - NFPA 70 NEC 2023.txt           (full text dump)
  - NFPA 70 NEC 2023_paragraphs.json  (paragraph-indexed JSON: {index: {content, page}})

Required pip packages (not included in pyproject.toml):
  - azure-ai-documentintelligence
  - azure-core (transitive dependency of the above)

Required environment variables (set in .env at project root):
  - AZURE_DOCUMENT_INTELLIGENCE_KEY: API key for Azure Document Intelligence resource
  - DOCUMENT_INTELLIGENCE_ENDPOINT_URL: Endpoint URL for the Azure Document Intelligence resource

IMPORTANT — this script is NOT re-runnable.  The Azure Document Intelligence
resource that was used to OCR the original PDF has been decommissioned.  The
raw output files in data/raw/ are the only artefacts we have.

Known limitations of the existing OCR output:
  - Tables were NOT detected as structured objects by Document Intelligence.
    Instead, individual table cells were captured as separate paragraphs in
    reading order (left-to-right, top-to-bottom).  This means the downstream
    cleaning pipeline (see src/nec_rag/data_preprocessing/tables/tables.py) must reconstruct
    table structure from the flat paragraph stream.
  - Two-column page layouts cause the OCR to read the left column first, then
    the right column.  When a table spans both columns, its content may be
    interleaved with unrelated section text from the opposite column.
  - Page headers, footers, copyright notices, and section-number markers are
    interspersed with the main content and must be stripped by the cleaning
    pipeline.
"""

import json
import logging
import os
import time
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient, models
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Resolve project root and load environment variables
ROOT = Path(__file__).parent.parent.resolve()
load_dotenv(ROOT / ".env")

PDF_FILE = ROOT / "data" / "raw" / "NFPA 70 NEC 2023.pdf"


def get_client() -> DocumentIntelligenceClient:
    """Initialize and return an Azure Document Intelligence client."""
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    endpoint = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT_URL")
    if key is None or endpoint is None:
        raise ValueError(f"Must provide both key ({key}) and endpoint ({endpoint})")

    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    return client


def run_ocr(client: DocumentIntelligenceClient) -> models.AnalyzeResult:
    """Load the PDF and run OCR via Azure Document Intelligence (~200s for the full document)."""
    start = time.time()
    logger.info("Starting OCR on %s", PDF_FILE.name)
    with open(PDF_FILE, "rb") as fopen:
        poller = client.begin_analyze_document("prebuilt-read", fopen)
    result = poller.result()
    logger.info("Finished OCR in %.2f seconds", time.time() - start)
    return result


def save_text(result: models.AnalyzeResult) -> None:
    """Save OCR results as both a full text file and a paragraph-indexed JSON file."""
    # Save full text (strip non-ASCII via charmap encoding)
    content = result.content.encode("charmap", errors="ignore").decode("charmap")
    txt_file = ROOT / "data" / "raw" / "NFPA 70 NEC 2023.txt"
    with open(txt_file, "w", encoding="utf-8") as fopen:
        fopen.write(content)
    logger.info("Wrote full text to %s", txt_file)

    # Save paragraph-indexed JSON: {index: {content, page}}
    output_json = {
        i: {
            "content": paragraph["content"].encode("charmap", errors="ignore").decode("charmap"),
            "page": paragraph["boundingRegions"][0]["pageNumber"],
        }
        for i, paragraph in enumerate(result.paragraphs)
    }
    output_file = ROOT / "data" / "raw" / "NFPA 70 NEC 2023_paragraphs.json"
    with open(output_file, "w", encoding="utf-8") as fopen:
        json.dump(output_json, fopen)
    logger.info("Wrote paragraph JSON to %s", output_file)


def main():
    """Run the full OCR pipeline: initialize client, run OCR, save results."""
    ocr_client = get_client()
    ocr_result = run_ocr(ocr_client)
    save_text(ocr_result)


if __name__ == "__main__":
    main()
