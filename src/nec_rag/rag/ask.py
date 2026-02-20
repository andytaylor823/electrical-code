"""Ask a question against the NEC RAG pipeline.

Embeds the user's question, retrieves the top-K most relevant subsections from
ChromaDB, then asks Azure OpenAI GPT for an answer with citation verification.

Supports multiple embedding models via --model (must match what was used in embed.py).

Usage:
    python -m nec_rag.rag.ask                         # default: qwen3
    python -m nec_rag.rag.ask --model azure-large     # use Azure embeddings
"""

import argparse
import json
import logging
import os
import sys
import time

import chromadb
from dotenv import load_dotenv
from openai import AzureOpenAI

from nec_rag.data_preprocessing.embedding.config import COLLECTION_NAME, MODELS, ROOT, chroma_path

logger = logging.getLogger(__name__)

# region -- setup
load_dotenv(ROOT / ".env")

TOP_K = 20

LLM_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.2-chat")
LLM_CLIENT = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
)

ANSWER_PROMPT_TEMPLATE = """You are a legal expert with domain expertise in electrical codes. Given the following context from the National
Electrical Code (NEC), answer the following question.

Answer concisely, and cite your source from the provided NEC text. If the answer is not found in the provided context, you may draw upon your training knowledge, but be sure to note that the answer was not found in the provided NEC context.

Question:
{question}

NEC text:
{context}
"""

VERIFICATION_PROMPT_TEMPLATE = """
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


def load_resources(model_key: str):
    """Load the embedding model/client and ChromaDB collection.

    Returns (embed_fn, collection) where embed_fn accepts a string and returns a list[float].
    """
    model_cfg = MODELS[model_key]

    # Build the query embedding function based on model type
    if model_cfg["type"] == "local":
        import torch  # pylint: disable=import-outside-toplevel
        from sentence_transformers import SentenceTransformer  # pylint: disable=import-outside-toplevel

        logger.info("Loading local embedding model '%s'...", model_cfg["display_name"])
        t0 = time.time()
        st_model = SentenceTransformer(
            model_cfg["display_name"],
            model_kwargs={"torch_dtype": torch.float16},
            tokenizer_kwargs={"padding_side": "left"},
        )
        logger.info("Model loaded in %.1f seconds", time.time() - t0)

        def embed_fn(text: str) -> list[float]:
            return st_model.encode(text, prompt_name="query").tolist()

    elif model_cfg["type"] == "azure":
        embedding_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        )
        logger.info("Using Azure OpenAI embedding model '%s'", model_cfg["display_name"])

        def embed_fn(text: str) -> list[float]:  # type: ignore[misc]
            response = embedding_client.embeddings.create(input=text, model=model_cfg["display_name"])
            return response.data[0].embedding

    else:
        raise ValueError(f"Unknown model type: {model_cfg['type']}")

    # Load ChromaDB collection
    store_path = chroma_path(model_key)
    client = chromadb.PersistentClient(path=str(store_path))
    collection = client.get_collection(name=COLLECTION_NAME)
    logger.info("ChromaDB collection '%s' loaded with %d items from %s", COLLECTION_NAME, collection.count(), store_path)

    return embed_fn, collection


def retrieve(question: str, embed_fn, collection: chromadb.Collection) -> list[dict]:
    """Embed the question and retrieve the top-K most relevant subsections."""
    query_embedding = embed_fn(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    # Unpack ChromaDB's nested list structure (single query -> index 0)
    retrieved = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        retrieved.append({"document": doc, "metadata": meta, "distance": dist})
    return retrieved


def build_context(retrieved: list[dict]) -> str:
    """Build the context string from retrieved subsections, annotated with source info."""
    sections = []
    for item in retrieved:
        meta = item["metadata"]
        header = f"[Section {meta['section_id']}, Article {meta['article_num']}, page {meta['page']}]"
        sections.append(f"{header}\n{item['document']}")
    return "\n\n".join(sections)


def ask_llm(question: str, context: str) -> tuple[str, str]:
    """Send the question + context to GPT and run citation verification. Returns (answer, verification)."""
    answer_prompt = ANSWER_PROMPT_TEMPLATE.format(question=question, context=context)
    logger.info("Asking LLM for answer...")
    response = LLM_CLIENT.chat.completions.create(
        model=LLM_DEPLOYMENT,
        messages=[{"role": "user", "content": answer_prompt}],
    )
    answer = json.loads(response.to_json())["choices"][0]["message"]["content"]

    verification_prompt = VERIFICATION_PROMPT_TEMPLATE.format(question=question, context=context, answer=answer)
    logger.info("Asking LLM for citation verification...")
    response2 = LLM_CLIENT.chat.completions.create(
        model=LLM_DEPLOYMENT,
        messages=[{"role": "user", "content": verification_prompt}],
    )
    verification = json.loads(response2.to_json())["choices"][0]["message"]["content"]

    return answer, verification


def main():
    """Interactive loop: load model once, then accept questions until user quits."""
    parser = argparse.ArgumentParser(description="Ask questions against the NEC RAG pipeline")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="qwen3", help="Which embedding model to use for retrieval (default: qwen3)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    embed_fn, collection = load_resources(args.model)
    logger.info("Using embedding model: %s", MODELS[args.model]["display_name"])

    print()
    while True:
        question = input("Enter a question to ask the RAG app (or 'x' to quit):\n>> ")
        if question.strip().lower() == "x":
            sys.exit(0)

        # Retrieve relevant context
        retrieved = retrieve(question, embed_fn, collection)
        context = build_context(retrieved)

        # Ask GPT
        answer, verification = ask_llm(question, context)

        # Display results
        print()
        print("Answer:", answer)
        print()
        if "all good" in verification.lower():
            print("Citation verification: PASSED")
        else:
            print("Corrected:", verification)
        print()


if __name__ == "__main__":
    main()
