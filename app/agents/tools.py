# app/agents/tools.py
"""
Agent Tools Module
Defines the tools available to the RAG agent.

Tools implemented:
  1. document_retrieval_tool — searches the vector store (primary RAG tool)
  2. calculator_tool         — evaluates mathematical expressions
  3. web_search_placeholder  — stub for future web search integration

LangChain tool pattern: each tool is a decorated Python function.
The agent LLM decides which tool to invoke based on the user query.
"""

import logging
import math
import operator
from typing import Any

from langchain.tools import tool

logger = logging.getLogger(__name__)

# Global reference to retriever — set by RAGAgent before using tools
_retriever = None


def set_retriever(retriever) -> None:
    """
    Set the global retriever instance for the document retrieval tool.

    Called once by RAGAgent during initialization.

    Args:
        retriever: RAGRetriever instance.
    """
    global _retriever
    _retriever = retriever
    logger.info("Global retriever set for agent tools.")


@tool
def document_retrieval_tool(query: str) -> str:
    """
    Search the ingested documents for information relevant to the query.
    Use this tool whenever the user asks about anything that might be in the documents.
    Always prefer this tool over generating answers from memory.

    Args:
        query: The search query derived from the user's question.

    Returns:
        Retrieved context from documents, or a not-found message.
    """
    if _retriever is None:
        return "Document retriever is not initialized. Please ingest documents first."

    try:
        result = _retriever.retrieve(query)
        if not result.has_results:
            return (
                "I could not find relevant information in the provided documents "
                "for this query."
            )

        # Return the assembled context (already formatted with sources + scores)
        return f"Retrieved context:\n\n{result.context}\n\nSources: {', '.join(result.sources)}"

    except Exception as e:
        logger.error(f"document_retrieval_tool error: {e}")
        return f"An error occurred during document retrieval: {str(e)}"


@tool
def calculator_tool(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    Use this tool for arithmetic, percentage calculations, or numeric operations.
    Supports: +, -, *, /, **, sqrt, abs, round, and standard math functions.

    Args:
        expression: A mathematical expression string (e.g., '2 * (3 + 4)', 'sqrt(16)').

    Returns:
        The computed result as a string.
    """
    try:
        # Safe evaluation: whitelist of allowed names
        safe_globals = {
            "__builtins__": {},
            "abs": abs,
            "round": round,
            "sqrt": math.sqrt,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pow": math.pow,
            "pi": math.pi,
            "e": math.e,
            "floor": math.floor,
            "ceil": math.ceil,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
        }

        result = eval(expression.strip(), safe_globals, {})  # nosec — controlled whitelist
        logger.info(f"Calculator: '{expression}' = {result}")
        return f"Result: {result}"

    except ZeroDivisionError:
        return "Error: Division by zero."
    except SyntaxError:
        return f"Error: Invalid mathematical expression: '{expression}'"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@tool
def web_search_tool(query: str) -> str:
    """
    Search the web for real-time information not available in the documents.
    Use this ONLY when the user explicitly asks for current/live information,
    or when document retrieval returns no results for a factual question.

    Args:
        query: The web search query.

    Returns:
        Placeholder message (web search not yet implemented).
    """
    # NOTE: This is a placeholder for future integration.
    # To implement: integrate with SerpAPI, Tavily, or DuckDuckGo.
    logger.info(f"Web search requested for: '{query}' (not yet implemented)")
    return (
        f"Web search for '{query}' is not yet implemented in this version. "
        "Please refer to the ingested documents for information, "
        "or upgrade to a version with live web search enabled."
    )


def get_agent_tools(retriever=None) -> list:
    """
    Return the list of tools to be given to the agent.

    Args:
        retriever: Optional RAGRetriever instance to register globally.

    Returns:
        List of LangChain Tool objects.
    """
    if retriever is not None:
        set_retriever(retriever)

    return [
        document_retrieval_tool,
        calculator_tool,
        web_search_tool,
    ]