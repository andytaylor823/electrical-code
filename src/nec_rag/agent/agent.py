"""NEC expert agent powered by LangGraph and Azure OpenAI.

Creates a ReAct-style agent that uses tool calling to search the NEC and
analyze images before answering user questions.

Usage:
    python -m nec_rag.agent.agent                       # default: azure-large embeddings
    python -m nec_rag.agent.agent --model qwen3          # use local Qwen3 embeddings
"""

import argparse
import logging
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain_community.callbacks import get_openai_callback

from nec_rag.agent.prompts import AGENT_SYSTEM_PROMPT
from nec_rag.agent.resources import get_agent_llm, load_embedding_resources
from nec_rag.agent.tools import IMAGE_EXTENSIONS, browse_nec_structure, explain_image, get_vision_usage, nec_lookup, rag_search, reset_vision_usage
from nec_rag.data_preprocessing.embedding.config import MODELS

logger = logging.getLogger(__name__)


def build_nec_agent(embedding_model_key: str = "azure-large"):
    """Build and return the LangGraph agent with NEC tools.

    Pre-loads the RAG embedding resources so the first tool call is fast.
    """
    # Pre-warm the embedding model and ChromaDB collection
    load_embedding_resources(embedding_model_key)

    llm = get_agent_llm()
    tools = [rag_search, browse_nec_structure, nec_lookup, explain_image]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=AGENT_SYSTEM_PROMPT,
    )

    logger.info("Agent created with %d tools: %s", len(tools), [t.name for t in tools])
    return agent


def _detect_image_paths(text: str) -> list[str]:
    """Extract tokens from user input that look like paths to image files."""
    image_paths = []
    for token in text.split():
        path = Path(token).expanduser()
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.exists():
            image_paths.append(str(path.resolve()))
    return image_paths


def main():
    """Interactive CLI loop: accept user questions and stream agent responses."""
    parser = argparse.ArgumentParser(description="NEC expert agent (LangGraph + Azure OpenAI)")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="azure-large",
        help="Embedding model for RAG retrieval (default: azure-large)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    agent = build_nec_agent(embedding_model_key=args.model)
    logger.info("Agent ready. Type your question or 'x' to quit.")
    print()

    while True:
        try:
            user_input = input(">> ")
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        if user_input.strip().lower() == "x":
            sys.exit(0)
        if not user_input.strip():
            continue

        # If the user references an image file, add a note so the agent knows
        image_paths = _detect_image_paths(user_input)
        if image_paths:
            attachments = ", ".join(image_paths)
            user_input += f"\n\n[Attached image(s): {attachments}]"

        # Reset vision usage counters before each invocation
        reset_vision_usage()

        # Invoke the agent inside the token-tracking callback
        with get_openai_callback() as cb:
            result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})

        # Combine agent LLM usage (from callback) with standalone vision usage
        vision = get_vision_usage()
        total_prompt = cb.prompt_tokens + vision["prompt_tokens"]
        total_completion = cb.completion_tokens + vision["completion_tokens"]
        total_tokens = cb.total_tokens + vision["total_tokens"]

        logger.info(
            "Token usage — agent: %d prompt + %d completion = %d total (%d LLM calls) | "
            "vision: %d prompt + %d completion = %d total | "
            "combined: %d prompt + %d completion = %d total",
            cb.prompt_tokens,
            cb.completion_tokens,
            cb.total_tokens,
            cb.successful_requests,
            vision["prompt_tokens"],
            vision["completion_tokens"],
            vision["total_tokens"],
            total_prompt,
            total_completion,
            total_tokens,
        )

        # Print the final response (last AI message)
        final_message = result["messages"][-1]
        print()
        print(final_message.content)
        print(f"\n[Tokens: {total_tokens:,} ({total_prompt:,} prompt + {total_completion:,} completion) | LLM calls: {cb.successful_requests}]")
        print()


if __name__ == "__main__":
    main()
