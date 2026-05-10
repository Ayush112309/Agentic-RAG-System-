#!/usr/bin/env python3
# main.py
"""
Agentic RAG System — CLI Entry Point
Provides a command-line interface for document ingestion and querying.

Usage:
    python main.py --ingest data/samples/
    python main.py --query "What is the return policy?"
    python main.py --interactive
    python main.py --status
    python main.py --reset
"""

import argparse
import logging
import sys

from app.utils.logging_config import setup_logging
from app.utils.pipeline import RAGPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Agentic RAG System — Document Q&A with LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --ingest data/samples/
  python main.py --query "What are the main features?"
  python main.py --interactive
  python main.py --status
  python main.py --reset
        """,
    )

    parser.add_argument(
        "--ingest",
        metavar="PATH",
        help="Path to a file or directory to ingest into the vector store.",
    )
    parser.add_argument(
        "--query",
        metavar="QUESTION",
        help="Ask a single question and print the answer.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive chat session in the terminal.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show system status (document count, sources, config).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the vector store and clear conversation memory.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level (default: INFO).",
    )

    return parser.parse_args()


def print_response(response) -> None:
    """Pretty-print an AgentResponse to the terminal."""
    print("\n" + "=" * 60)
    print("ANSWER:")
    print("-" * 60)
    print(response.answer)

    if response.sources:
        print("\nSOURCES:")
        for src in response.sources:
            print(f"  • {src}")

    if response.retrieved_chunks:
        print(f"\nRELEVANCE SCORE: {response.top_relevance_score:.2f}")
        print(f"CHUNKS RETRIEVED: {len(response.retrieved_chunks)}")

    if response.used_tools:
        print(f"TOOLS USED: {', '.join(response.used_tools)}")

    print("=" * 60)


def run_interactive(pipeline: RAGPipeline) -> None:
    """
    Run an interactive chat loop in the terminal.

    Commands:
      /quit or /exit  — Exit the chat
      /clear          — Clear conversation memory
      /status         — Show system status
      /sources        — Show ingested sources
    """
    print("\n" + "=" * 60)
    print("  AGENTIC RAG SYSTEM — Interactive Mode")
    print("  Commands: /quit, /clear, /status, /sources")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        # Handle special commands
        if user_input.lower() in ("/quit", "/exit"):
            print("Goodbye!")
            break
        elif user_input.lower() == "/clear":
            pipeline.clear_memory()
            print("Conversation memory cleared.\n")
            continue
        elif user_input.lower() == "/status":
            status = pipeline.status()
            print(f"\nStatus: {status}\n")
            continue
        elif user_input.lower() == "/sources":
            sources = pipeline.vector_store.get_sources()
            if sources:
                print("\nIngested sources:")
                for s in sources:
                    print(f"  • {s}")
            else:
                print("No documents ingested yet.")
            print()
            continue

        # Process query
        try:
            response = pipeline.query(user_input)
            print_response(response)
        except Exception as e:
            logging.error(f"Query failed: {e}")
            print(f"Error: {e}\n")

        print()  # Blank line for readability


def main() -> int:
    """Main entry point. Returns exit code."""
    args = parse_args()

    # Setup logging before anything else
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting Agentic RAG System...")

    # Initialize pipeline
    try:
        pipeline = RAGPipeline.from_config()
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        print(f"Initialization error: {e}", file=sys.stderr)
        return 1

    # Handle --reset
    if args.reset:
        confirm = input("⚠️  This will delete all ingested documents. Confirm? [y/N]: ")
        if confirm.lower() == "y":
            pipeline.reset()
            print("Pipeline reset complete.")
        else:
            print("Reset cancelled.")
        return 0

    # Handle --status
    if args.status:
        status = pipeline.status()
        print("\n=== SYSTEM STATUS ===")
        print(f"Documents in vector store: {status['vector_store']['document_count']}")
        print(f"Persist directory: {status['vector_store']['persist_dir']}")
        print(f"Ingested sources: {status['vector_store']['sources']}")
        print(f"Memory turns: {status['memory']['turn_count']}")
        print(f"LLM model: {status['config']['llm_model']}")
        print(f"Embedding backend: {status['config']['embedding_backend']}")
        print(f"Chunk size: {status['config']['chunk_size']}")
        print(f"Retrieval k: {status['config']['retrieval_k']}")
        return 0

    # Handle --ingest
    if args.ingest:
        from pathlib import Path
        path = Path(args.ingest)

        try:
            if path.is_file():
                count = pipeline.ingest_file(path)
                print(f"✅ Ingested '{path.name}': {count} chunks stored.")
            elif path.is_dir():
                stats = pipeline.ingest_directory(path)
                print(
                    f"✅ Ingested directory '{path}': "
                    f"{stats['files']} files, "
                    f"{stats['chunks']} chunks, "
                    f"{stats['stored']} stored."
                )
            else:
                print(f"❌ Path not found: {path}", file=sys.stderr)
                return 1
        except Exception as e:
            logger.error(f"Ingestion error: {e}")
            print(f"❌ Ingestion failed: {e}", file=sys.stderr)
            return 1

    # Handle --query
    if args.query:
        try:
            response = pipeline.query(args.query)
            print_response(response)
        except Exception as e:
            logger.error(f"Query error: {e}")
            print(f"❌ Query failed: {e}", file=sys.stderr)
            return 1
        return 0

    # Handle --interactive (or default if nothing else specified)
    if args.interactive or (not args.ingest and not args.query and not args.status):
        run_interactive(pipeline)

    return 0


if __name__ == "__main__":
    sys.exit(main())