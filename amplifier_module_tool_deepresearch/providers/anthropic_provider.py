"""Anthropic Deep Research provider implementation.

Implements iterative deep research using Claude with:
- Extended thinking for complex reasoning
- Web search tool for real-time information
- Multi-iteration search refinement based on complexity

This follows the pattern from anthropic-deep-research where:
1. Claude analyzes the query and determines search strategy
2. Multiple iterations refine searches based on previous results
3. Final synthesis with citations and structured output
"""

import asyncio
import logging
import re
from typing import Any

from anthropic import AsyncAnthropic

from .._constants import (
    ANTHROPIC_DEFAULT_MAX_ITERATIONS,
    ANTHROPIC_DEFAULT_MAX_TOKENS,
    ANTHROPIC_DEFAULT_MODEL,
    ANTHROPIC_DEFAULT_TIMEOUT,
    ANTHROPIC_MODELS,
    ANTHROPIC_THINKING_BUDGET,
    TaskComplexity,
)
from .base import (
    Citation,
    DeepResearchProviderBase,
    DeepResearchRequest,
    DeepResearchResult,
    ReasoningStep,
    WebSearchStep,
)

logger = logging.getLogger(__name__)


# System prompts for iterative deep research
MAIN_AGENT_SYSTEM = """
You are a helpful assistant with access to web search for conducting deep research.

DEEP WEB RESEARCH CAPABILITIES:
- When information gathering is required, use the 'web_search' tool
- Perform comprehensive, multi-level research through successive search refinements
- Explore topics with increasing depth, identify key subtopics, resolve knowledge gaps
- Cross-reference information across credible sources
- Verify information and present with clear citations

WORKFLOW APPROACH:
- Break complex queries into logical sub-questions
- Analyze and synthesize search results comprehensively
- Distinguish between factual information and interpretations/opinions
- Present information in a structured, easy-to-understand format
- Acknowledge limitations when definitive information isn't available

Respond with thorough, accurate, and well-sourced information.
"""

ITERATIVE_SEARCH_SYSTEM = """
TASK: Conduct iterative deep web search for comprehensive, accurate information.
MAX ITERATIONS: {max_iterations}
CURRENT ITERATION: {current_iteration}

SEARCH METHODOLOGY:
1. Begin with broad queries to establish baseline understanding
2. Identify key subtopics, perspectives, and knowledge gaps
3. Formulate increasingly specific queries based on findings
4. Cross-reference across multiple credible sources
5. Prioritize recent sources for time-sensitive topics

ITERATION PROCESS:
After each search, analyze:
- What new information was discovered?
- What questions remain unanswered?
- What conflicting information requires verification?
- What specialized terminology could improve queries?

DELIVERABLE FORMAT:
1. Executive Summary (key findings and conclusions)
2. Detailed Analysis (organized by subtopic)
3. Evidence Summary (with citations and URLs)
4. Remaining Questions (incomplete or conflicting areas)
5. Search Process Documentation (queries used)

QUALITY CRITERIA:
- Multiple distinct credible sources
- Multiple perspectives for subjective topics
- Clear distinction between facts and interpretation
- Citation of primary sources when available
- Transparent acknowledgment of limitations
"""


# Web search tool definition for Claude
WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
}


class AnthropicDeepResearchProvider(DeepResearchProviderBase):
    """Anthropic Deep Research provider using Claude with iterative search.

    Uses Claude's extended thinking and web search capabilities to
    conduct multi-step research with iterative refinement.
    """

    name = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: AsyncAnthropic | None = None,
        timeout: float = ANTHROPIC_DEFAULT_TIMEOUT,
        debug: bool = False,
    ):
        """Initialize Anthropic Deep Research provider.

        Args:
            api_key: Anthropic API key
            client: Pre-configured AsyncAnthropic client (for testing)
            timeout: Default request timeout in seconds
            debug: Enable debug logging
        """
        if client is not None:
            self._client = client
            self._api_key = None
        elif api_key is not None:
            self._client = AsyncAnthropic(api_key=api_key)
            self._api_key = api_key
        else:
            self._client = None
            self._api_key = None

        self.timeout = timeout
        self.debug = debug

    @property
    def is_available(self) -> bool:
        """Check if provider has valid credentials."""
        return self._client is not None

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models for deep research."""
        return [
            {
                "id": "claude-sonnet-4-5-20250514",
                "display_name": "Claude Sonnet 4.5",
                "context_window": 200000,
                "max_output_tokens": 16384,
                "capabilities": ["deep_research", "extended_thinking", "web_search", "citations"],
                "description": "Latest Claude with extended thinking for complex research",
            },
            {
                "id": "claude-3-7-sonnet-latest",
                "display_name": "Claude 3.7 Sonnet",
                "context_window": 200000,
                "max_output_tokens": 16384,
                "capabilities": ["deep_research", "extended_thinking", "web_search", "citations"],
                "description": "Claude 3.7 with extended thinking capabilities",
            },
            {
                "id": "claude-3-5-sonnet-latest",
                "display_name": "Claude 3.5 Sonnet",
                "context_window": 200000,
                "max_output_tokens": 8192,
                "capabilities": ["deep_research", "web_search", "citations", "fast"],
                "description": "Fast Claude 3.5 for quicker research tasks",
            },
        ]

    async def research(self, request: DeepResearchRequest) -> DeepResearchResult:
        """Execute iterative deep research with Claude.

        Args:
            request: Research request with query and configuration

        Returns:
            DeepResearchResult with report and metadata
        """
        if not self._client:
            raise RuntimeError("Anthropic client not initialized - missing API key")

        model = request.model or ANTHROPIC_DEFAULT_MODEL
        if model not in ANTHROPIC_MODELS:
            logger.warning(f"Model {model} not in known models list: {ANTHROPIC_MODELS}")

        # Determine complexity and iterations
        complexity = TaskComplexity(request.task_complexity)
        max_iterations = request.max_iterations or self._get_max_iterations(complexity)
        thinking_budget = ANTHROPIC_THINKING_BUDGET.get(complexity, 2048)

        logger.info(
            f"[Anthropic Deep Research] Starting with model={model}, "
            f"complexity={complexity.value}, max_iterations={max_iterations}"
        )

        # Run iterative search
        return await self._iterative_search(
            query=request.query,
            instructions=request.instructions,
            model=model,
            max_iterations=max_iterations,
            thinking_budget=thinking_budget,
            max_tokens=request.max_output_tokens or ANTHROPIC_DEFAULT_MAX_TOKENS,
            timeout=request.timeout or self.timeout,
        )

    def _get_max_iterations(self, complexity: TaskComplexity) -> int:
        """Get max iterations based on task complexity."""
        match complexity:
            case TaskComplexity.LOW:
                return 2
            case TaskComplexity.MEDIUM:
                return ANTHROPIC_DEFAULT_MAX_ITERATIONS
            case TaskComplexity.HIGH:
                return 8

    async def _iterative_search(
        self,
        query: str,
        instructions: str | None,
        model: str,
        max_iterations: int,
        thinking_budget: int,
        max_tokens: int,
        timeout: float,
    ) -> DeepResearchResult:
        """Perform iterative deep search with Claude.

        This implements the multi-iteration pattern where each search
        builds on previous results.
        """
        all_reasoning: list[ReasoningStep] = []
        all_searches: list[WebSearchStep] = []
        conversation: list[dict[str, Any]] = []

        # Initial user message
        user_content = query
        if instructions:
            user_content = f"{instructions}\n\n{query}"

        conversation.append({"role": "user", "content": user_content})

        final_text = ""
        citations: list[Citation] = []
        iteration = 0
        total_input_tokens = 0
        total_output_tokens = 0

        while iteration < max_iterations:
            iteration += 1

            # Build system prompt for this iteration
            system = ITERATIVE_SEARCH_SYSTEM.format(
                max_iterations=max_iterations,
                current_iteration=iteration,
            )

            if iteration == 1:
                system = MAIN_AGENT_SYSTEM + "\n\n" + system

            try:
                # Type ignore needed for dynamic message/tool construction
                response = await asyncio.wait_for(
                    self._client.messages.create(  # type: ignore[union-attr]
                        model=model,
                        max_tokens=max_tokens,
                        system=system,
                        messages=conversation,  # type: ignore[arg-type]
                        tools=[WEB_SEARCH_TOOL],  # type: ignore[arg-type]
                        thinking={
                            "type": "enabled",
                            "budget_tokens": thinking_budget,
                        },
                    ),
                    timeout=timeout,
                )
            except TimeoutError:
                logger.error(f"[Anthropic Deep Research] Timeout at iteration {iteration}")
                break

            # Track usage
            if hasattr(response, "usage"):
                total_input_tokens += getattr(response.usage, "input_tokens", 0)
                total_output_tokens += getattr(response.usage, "output_tokens", 0)

            # Process response content
            assistant_content: list[dict[str, Any]] = []
            tool_use_blocks: list[dict[str, Any]] = []

            for block in response.content:
                block_type = getattr(block, "type", None)

                if block_type == "thinking":
                    thinking_text = getattr(block, "thinking", "")
                    if thinking_text:
                        all_reasoning.append(ReasoningStep(text=thinking_text, iteration=iteration))
                    assistant_content.append(
                        {
                            "type": "thinking",
                            "thinking": thinking_text,
                            "signature": getattr(block, "signature", ""),
                        }
                    )

                elif block_type == "text":
                    text = getattr(block, "text", "")
                    final_text = text  # Keep updating to get final answer
                    assistant_content.append({"type": "text", "text": text})

                    # Extract citations from text (URLs in markdown format)
                    citations.extend(self._extract_citations_from_text(text))

                elif block_type == "tool_use":
                    tool_name = getattr(block, "name", "")
                    tool_input = getattr(block, "input", {})
                    tool_id = getattr(block, "id", "")

                    if tool_name == "web_search":
                        search_query = tool_input.get("query", "") if isinstance(tool_input, dict) else ""
                        all_searches.append(WebSearchStep(query=search_query, status="completed"))

                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": tool_name,
                            "input": tool_input,
                        }
                    )
                    tool_use_blocks.append(
                        {
                            "id": tool_id,
                            "name": tool_name,
                            "input": tool_input,
                        }
                    )

                elif block_type == "web_search_tool_result":
                    # Web search results from Claude's native tool
                    search_results = getattr(block, "content", [])
                    for result in search_results:
                        if hasattr(result, "url") and hasattr(result, "title"):
                            citations.append(
                                Citation(
                                    title=getattr(result, "title", ""),
                                    url=getattr(result, "url", ""),
                                )
                            )

            # Add assistant response to conversation
            conversation.append({"role": "assistant", "content": assistant_content})

            # Check stop reason
            stop_reason = getattr(response, "stop_reason", "end_turn")

            if stop_reason == "tool_use" and tool_use_blocks:
                # Need to provide tool results (Claude handles web search internally)
                # For Claude's built-in web search, we just acknowledge it
                tool_results = []
                for tool in tool_use_blocks:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool["id"],
                            "content": "Search completed.",
                        }
                    )
                conversation.append({"role": "user", "content": tool_results})
            else:
                # End turn - research complete
                logger.info(f"[Anthropic Deep Research] Completed after {iteration} iterations")
                break

        return DeepResearchResult(
            report_text=final_text,
            provider="anthropic",
            model=model,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            citations=self._dedupe_citations(citations),
            reasoning_steps=all_reasoning,
            web_searches=all_searches,
            iterations_completed=iteration,
            max_iterations=max_iterations,
        )

    def _extract_citations_from_text(self, text: str) -> list[Citation]:
        """Extract citations from markdown-formatted text."""
        citations = []

        # Match markdown links: [title](url)
        pattern = r"\[([^\]]+)\]\((https?://[^\)]+)\)"
        for match in re.finditer(pattern, text):
            title = match.group(1)
            url = match.group(2)
            citations.append(
                Citation(
                    title=title,
                    url=url,
                    start_index=match.start(),
                    end_index=match.end(),
                )
            )

        return citations

    def _dedupe_citations(self, citations: list[Citation]) -> list[Citation]:
        """Remove duplicate citations by URL."""
        seen_urls: set[str] = set()
        deduped = []
        for c in citations:
            if c.url not in seen_urls:
                seen_urls.add(c.url)
                deduped.append(c)
        return deduped
