# electrical-code

WIP RAG application to answer questions relating to the NFPA 70 electrical code.

## Project Structure

```
├── data/
│   ├── raw/                    # Source PDFs
│   └── intermediate/           # Processed JSON/text outputs
├── scripts/                    # Standalone utility scripts
├── src/
│   └── nec_rag/                # Main Python package
│       ├── cleaning/           # OCR + text cleaning pipeline
│       └── rag/                # RAG agent + embeddings
├── tests/                      # Test scripts
└── frontend/                   # Future frontend app
```

## Setup

```bash
# Install in editable mode (from project root)
pip install -e ".[dev]"
```

Create a `.env` file in the project root with your Azure credentials:

```
AZURE_OPENAI_API_KEY=...
GPT_41_ENDPOINT_URL=...
EMBEDDINGS_SMALL_ENDPOINT_URL=...
AZURE_DOCUMENT_INTELLIGENCE_KEY=...
DOCUMENT_INTELLIGENCE_ENDPOINT_URL=...
DEPLOYMENT_NAME=gpt-4.1
```

## Pipeline

1. **OCR**: Extract text from source PDF
   ```bash
   python -m nec_rag.cleaning.ocr
   ```

2. **Clean**: Run the text cleaning pipeline
   ```bash
   python -m nec_rag.cleaning.clean
   ```

3. **Split**: Break cleaned text into sections and definitions
   ```bash
   python -m nec_rag.rag.split_into_sections
   ```

4. **Embed**: Generate vector embeddings for sections
   ```bash
   python -m nec_rag.rag.embed
   ```

5. **Ask**: Query the RAG application
   ```bash
   python -m nec_rag.rag.ask
   ```
