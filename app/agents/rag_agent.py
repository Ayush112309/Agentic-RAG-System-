# app/agents/rag_agent.py
"""
RAG Agent Module
The main orchestration layer that ties together:
  - Document retrieval (RAGRetriever)
  - LLM response generation
  - Tool usage (calculator, web search placeholder)
  - Conversation memory
  - Hallucination prevention (strict context-grounding)

The agent uses LangChain's OpenAI Functions agent pattern,
which allows reliable tool calling via the OpenAI function-calling API.
Falls back to a simpler chain if tool-calling is unavailable.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain.schema import BaseMessage

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """
    Structured response from the RAG agent.

    Attributes:
        answer: The LLM's generated answer.
        sources: Files the answer is grounded in.
        retrieved_chunks: Raw chunks with scores for display.
        top_relevance_score: Best match score (0–1).
        used_tools: List of tool names invoked during this response.
        is_grounded: Whether the answer came from retrieved documents.
    """
    answer: str
    sources: List[str]
    retrieved_chunks: list
    top_relevance_score: float
    used_tools: List[str]
    is_grounded: bool


# Strict system prompt — forces LLM to stay within document context
# Strict grounded system prompt
RAG_SYSTEM_PROMPT = """
You are a grounded document question-answering assistant.

You MUST answer ONLY using the retrieved document context.

STRICT RULES:
- Do NOT use outside knowledge.
- Do NOT guess.
- Do NOT hallucinate.
- Do NOT invent definitions.
- Do NOT paraphrase technical definitions incorrectly.
- Use exact terminology from the retrieved context whenever possible.
- Prefer exact phrases and wording from the retrieved documents.
- If the answer is not explicitly present in the context, say:
  'I could not find relevant information in the provided documents.'

ANSWERING STYLE:
- Keep answers concise and factual.
- Stay grounded in the provided context.
- Mention source names only if relevant.
- Avoid unnecessary explanations.
- If multiple retrieved chunks conflict, prefer the most relevant chunk.

TOOL USAGE:
- Use the `document_retrieval_tool` before answering factual questions.
- Use the `calculator_tool` for mathematical calculations.

The retrieved context will be provided below.
"""


class RAGAgent:
    """
    The primary agent that orchestrates RAG + tool use + memory.

    Two operating modes:
    1. AGENT MODE (default): Uses LangChain's OpenAI Functions agent
       with explicit tool calling. Best for complex, multi-step queries.
    2. CHAIN MODE (fallback): Simple retrieval → LLM chain when
       agent dependencies are unavailable.

    Usage:
        agent = RAGAgent(retriever, memory_manager)
        response = agent.query("What is the company's refund policy?")
        print(response.answer)
    """

    def __init__(
        self,
        retriever,
        memory_manager,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
        use_agent_mode: bool = True,
    ) -> None:
        """
        Initialize the RAG agent.

        Args:
            retriever: RAGRetriever instance for document lookup.
            memory_manager: MemoryManager for conversation history.
            model_name: LLM model identifier (e.g., 'gpt-4o-mini').
            temperature: LLM temperature (low = less creative, more factual).
            use_agent_mode: If True, use full agent with tools; else use chain.
        """
        self.retriever = retriever
        self.memory_manager = memory_manager
        self.model_name = model_name
        self.temperature = temperature
        self.use_agent_mode = use_agent_mode

        self._llm = None
        self._agent_executor = None

        logger.info(
            f"RAGAgent initialized. Model: {model_name}, "
            f"Mode: {'agent' if use_agent_mode else 'chain'}, "
            f"Temperature: {temperature}"
        )

    @property
    def llm(self):
        """Lazy-load the LLM."""
        if self._llm is None:
            self._llm = self._init_llm()
        return self._llm

    def _init_llm(self):
        """Initialize the language model."""
        api_key = os.getenv("OPENAI_API_KEY")

        if api_key:
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    openai_api_key=api_key,
                    streaming=False,
                )
                logger.info(f"ChatOpenAI LLM initialized: {self.model_name}")
                return llm
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI LLM: {e}")
                raise
        else:
            logger.warning("No OPENAI_API_KEY — attempting local Ollama LLM fallback.")
            return self._init_ollama_llm()

    def _init_ollama_llm(self):
        """Fallback to local Ollama (llama3) if OpenAI is unavailable."""
        try:
            from langchain_community.llms import Ollama
            llm = Ollama(model="llama3", temperature=self.temperature)
            logger.info("Ollama LLM (llama3) initialized as fallback.")
            return llm
        except Exception as e:
            logger.error(f"Ollama initialization failed: {e}")
            raise RuntimeError(
                "No LLM available. Set OPENAI_API_KEY or install Ollama with llama3."
            )

    def _build_agent_executor(self):
        """
        Build the LangChain OpenAI Functions agent executor.

        Returns:
            AgentExecutor ready for invocation.
        """
        from langchain.agents import AgentExecutor, create_openai_functions_agent
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from app.agents.tools import get_agent_tools

        # Register retriever with tools
        tools = get_agent_tools(retriever=self.retriever)

        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt,
        )

        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

        logger.info("Agent executor built successfully.")
        return executor

    def query(self, user_query: str) -> AgentResponse:
        """
        Process a user query through the RAG agent pipeline.

        Pipeline:
          1. Retrieve relevant context from vector store
          2. If agent mode: invoke agent (may call tools autonomously)
          3. If chain mode: directly prompt LLM with context
          4. Package result into AgentResponse

        Args:
            user_query: The user's natural language question.

        Returns:
            AgentResponse with answer, sources, and metadata.
        """
        logger.info(f"Processing query: '{user_query[:100]}'")

        # Always retrieve context directly (for display and fallback)
        retrieval_result = self.retriever.retrieve(user_query)

        if self.use_agent_mode and os.getenv("OPENAI_API_KEY"):
            return self._agent_query(user_query, retrieval_result)
        else:
            return self._chain_query(user_query, retrieval_result)

    def _agent_query(self, user_query: str, retrieval_result) -> AgentResponse:
        """
        Run query through the full agent with tool calling.

        Args:
            user_query: User's question.
            retrieval_result: Pre-fetched retrieval result.

        Returns:
            AgentResponse.
        """
        if self._agent_executor is None:
            try:
                self._agent_executor = self._build_agent_executor()
            except Exception as e:
                logger.warning(f"Agent build failed ({e}), falling back to chain mode.")
                return self._chain_query(user_query, retrieval_result)

        try:
            chat_history = self.memory_manager.get_history()

            result = self._agent_executor.invoke({
                "input": user_query,
                "chat_history": chat_history,
            })

            answer = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])

            # Extract tool names from intermediate steps
            used_tools = [
                step[0].tool
                for step in intermediate_steps
                if hasattr(step[0], "tool")
            ]

            # Persist conversation turn
            self.memory_manager.add_user_message(user_query)
            self.memory_manager.add_ai_message(answer)

            return AgentResponse(
                answer=answer,
                sources=retrieval_result.sources,
                retrieved_chunks=retrieval_result.documents,
                top_relevance_score=retrieval_result.top_score,
                used_tools=used_tools,
                is_grounded=retrieval_result.has_results,
            )

        except Exception as e:
            logger.error(f"Agent query failed: {e}")
            return self._chain_query(user_query, retrieval_result)

    def _chain_query(self, user_query: str, retrieval_result) -> AgentResponse:
        """
        Simple chain mode: retrieve context → prompt LLM directly.
        Used as fallback when agent mode is unavailable.

        Args:
            user_query: User's question.
            retrieval_result: Pre-fetched retrieval result.

        Returns:
            AgentResponse.
        """
        if not retrieval_result.has_results:
            answer = (
                "I could not find relevant information in the provided documents. "
                "Please ensure the relevant documents have been ingested."
            )
            return AgentResponse(
                answer=answer,
                sources=[],
                retrieved_chunks=[],
                top_relevance_score=0.0,
                used_tools=[],
                is_grounded=False,
            )

        # Build prompt with retrieved context
        chat_history_str = self._format_chat_history()
        prompt = self._build_chain_prompt(
            user_query=user_query,
            context=retrieval_result.context,
            chat_history=chat_history_str,
        )

        try:
            response = self.llm.invoke(prompt)
            # Handle both string and AIMessage responses
            answer = response.content if hasattr(response, "content") else str(response)

        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            answer = f"LLM error: {str(e)}"

        # Save to memory
        self.memory_manager.add_user_message(user_query)
        self.memory_manager.add_ai_message(answer)

        return AgentResponse(
            answer=answer,
            sources=retrieval_result.sources,
            retrieved_chunks=retrieval_result.documents,
            top_relevance_score=retrieval_result.top_score,
            used_tools=["document_retrieval_tool"],
            is_grounded=True,
        )

    def _build_chain_prompt(
        self,
        user_query: str,
        context: str,
        chat_history: str = "",
    ) -> str:
        """
        Build the LLM prompt for chain mode.
        """

        history_section = ""

        if chat_history:
            history_section = (
                f"\n\nConversation History:\n{chat_history}\n"
            )

        return f"""{RAG_SYSTEM_PROMPT}
{history_section}

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{user_query}

Provide a concise factual answer using ONLY the document context.

If the answer exists in the context:
- answer clearly
- use exact terminology when possible
- summarize naturally

If the answer does not exist:
say exactly:
"I could not find relevant information in the provided documents."

ANSWER:
"""

    def _format_chat_history(self) -> str:
        """
        Format conversation history as a readable string.
        """

        messages = self.memory_manager.get_history()

        if not messages:
            return ""

        lines = []

        for msg in messages:
            role = "Human" if msg.type == "human" else "Assistant"
            lines.append(f"{role}: {msg.content}")

        return "\n".join(lines)

    def clear_memory(self) -> None:
        """Clear the conversation memory."""
        self.memory_manager.clear()
        logger.info("Agent memory cleared.")