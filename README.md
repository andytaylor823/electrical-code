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
│       ├── data_preprocessing/ # Data pipeline
│       │   ├── text_cleaning/  # OCR text cleaning (5-step pipeline + structuring)
│       │   ├── tables/         # Table detection & LLM reconstruction
│       │   └── embedding/      # Chunking, embedding, vector storage
│       └── rag/                # Question-answering agent
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
   python scripts/ocr.py
   ```

2. **Clean**: Run the text cleaning pipeline
   ```bash
   python -m nec_rag.data_preprocessing.text_cleaning.clean
   ```

3. **Structure**: Parse cleaned text into hierarchical JSON
   ```bash
   python -m nec_rag.data_preprocessing.text_cleaning.structure
   ```

4. **Embed**: Generate vector embeddings for sections
   ```bash
   python -m nec_rag.data_preprocessing.embedding.embed
   ```

5. **Ask**: Query the RAG application
   ```bash
   python -m nec_rag.rag.ask
   ```
