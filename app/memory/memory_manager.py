# app/memory/memory_manager.py
"""
Memory Manager Module
Manages conversation history for the RAG agent.

Uses LangChain's ConversationBufferWindowMemory:
  - Stores recent N turns of conversation
  - Provides chat history as LangChain message objects
  - Supports clearing and exporting history

Why windowed memory:
  - Prevents context window overflow for long conversations
  - Keeps the most recent and relevant exchanges
  - Configurable window size
"""

import logging
from typing import List

from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage

logger = logging.getLogger(__name__)

# Default number of conversation turns to remember
DEFAULT_WINDOW_SIZE = 10


class MemoryManager:
    """
    Manages conversation memory for the RAG agent.

    Wraps ConversationBufferWindowMemory to provide:
    - Simple add/get interface
    - Clear history
    - Export history as formatted string
    - Message count

    Usage:
        memory = MemoryManager(window_size=10)
        memory.add_user_message("What is RAG?")
        memory.add_ai_message("RAG stands for Retrieval-Augmented Generation...")
        history = memory.get_history()
    """

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE) -> None:
        """
        Initialize the memory manager.

        Args:
            window_size: Number of conversation turns (human+AI pairs) to retain.
        """
        self.window_size = window_size
        self._memory = ConversationBufferWindowMemory(
            k=window_size,
            return_messages=True,   # Return as LangChain BaseMessage objects
            memory_key="chat_history",
            input_key="input",
            output_key="output",
        )

        # Internal list for full history display (not windowed)
        self._full_history: List[dict] = []

        logger.info(f"MemoryManager initialized with window_size={window_size}")

    def add_user_message(self, message: str) -> None:
        """
        Record a user (human) message.

        Args:
            message: The user's message text.
        """
        self._memory.chat_memory.add_user_message(message)
        self._full_history.append({"role": "user", "content": message})
        logger.debug(f"User message added to memory: '{message[:80]}'")

    def add_ai_message(self, message: str) -> None:
        """
        Record an AI (assistant) message.

        Args:
            message: The AI's response text.
        """
        self._memory.chat_memory.add_ai_message(message)
        self._full_history.append({"role": "assistant", "content": message})
        logger.debug(f"AI message added to memory: '{message[:80]}'")

    def get_history(self) -> List[BaseMessage]:
        """
        Return conversation history as LangChain message objects.
        Only returns messages within the window.

        Returns:
            List of HumanMessage and AIMessage objects.
        """
        return self._memory.chat_memory.messages

    def get_full_history(self) -> List[dict]:
        """
        Return the complete conversation history (not windowed).
        Useful for UI display.

        Returns:
            List of dicts with 'role' and 'content' keys.
        """
        return self._full_history.copy()

    def get_formatted_history(self) -> str:
        """
        Return the windowed conversation history as a formatted string.

        Returns:
            Multi-line string of conversation turns.
        """
        messages = self.get_history()
        if not messages:
            return "(No conversation history)"

        lines = []
        for msg in messages:
            role = "Human" if msg.type == "human" else "Assistant"
            lines.append(f"{role}: {msg.content}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all conversation memory."""
        self._memory.clear()
        self._full_history.clear()
        logger.info("Conversation memory cleared.")

    @property
    def message_count(self) -> int:
        """Return total number of messages in full history."""
        return len(self._full_history)

    @property
    def turn_count(self) -> int:
        """Return number of complete conversation turns (user+AI pairs)."""
        return len(self._full_history) // 2