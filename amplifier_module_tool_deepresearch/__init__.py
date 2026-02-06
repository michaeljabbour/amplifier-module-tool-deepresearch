"""Deep Research tool module for Amplifier.

Multi-provider deep research capabilities supporting:
- OpenAI Deep Research (o3-deep-research, o4-mini-deep-research)
- Anthropic Deep Research (Claude with iterative web search)

Usage:
    # In bundle configuration
    tools:
      - module: tool-deepresearch
        source: git+https://github.com/michaeljabbour/amplifier-module-tool-deepresearch@main
        config:
          default_provider: anthropic  # or openai
          default_model: claude-sonnet-4-5

    # The LLM can then call:
    deep_research(query="What are the latest advances in quantum computing?")
"""

# Amplifier module metadata
__amplifier_module_type__ = "tool"

import logging
import os
from typing import Any

from amplifier_core import ModuleCoordinator

from ._constants import (
    ANTHROPIC_DEFAULT_MODEL,
    OPENAI_DEFAULT_MODEL,
    TaskComplexity,
)
from ._constants import (
    DeepResearchProvider as DeepResearchProviderEnum,
)
from ._prompt_utils import (
    estimate_task_complexity,
    generate_clarifying_questions,
    rewrite_research_prompt,
    select_provider,
)
from .providers import (
    AnthropicDeepResearchProvider,
    Citation,
    CodeExecutionStep,
    DeepResearchProviderBase,
    DeepResearchRequest,
    DeepResearchResult,
    FileSearchStep,
    OpenAIDeepResearchProvider,
    ReasoningStep,
    WebSearchStep,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Main class
    "DeepResearchProvider",
    # Tool class
    "DeepResearchTool",
    # Data classes
    "DeepResearchRequest",
    "DeepResearchResult",
    "Citation",
    "ReasoningStep",
    "WebSearchStep",
    "FileSearchStep",
    "CodeExecutionStep",
    # Enums
    "DeepResearchProviderEnum",
    "TaskComplexity",
    # Utilities
    "estimate_task_complexity",
    "select_provider",
    "generate_clarifying_questions",
    "rewrite_research_prompt",
    # Mount function
    "mount",
]


class DeepResearchProvider:
    """Unified deep research provider supporting multiple backends.

    Automatically selects the best provider based on:
    - Available credentials
    - Query requirements (vector stores, MCP servers)
    - User preferences (speed, cost)

    Example:
        provider = DeepResearchProvider(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
        )

        result = await provider.research(
            query="Research quantum computing applications in drug discovery",
            task_complexity="high",
        )

        print(result.report_text)
        for citation in result.citations:
            print(f"  - {citation.title}: {citation.url}")
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        *,
        default_provider: str | None = None,
        default_model: str | None = None,
        timeout: float | None = None,
        debug: bool = False,
    ):
        """Initialize the deep research provider.

        Args:
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            anthropic_api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            default_provider: Preferred provider: 'openai' or 'anthropic'
            default_model: Default model to use
            timeout: Request timeout in seconds
            debug: Enable debug logging
        """
        # Get API keys from environment if not provided
        openai_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        anthropic_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")

        # Initialize available providers
        self._providers: dict[str, DeepResearchProviderBase] = {}

        if openai_key:
            self._providers["openai"] = OpenAIDeepResearchProvider(
                api_key=openai_key,
                timeout=timeout or 600.0,
                debug=debug,
            )

        if anthropic_key:
            self._providers["anthropic"] = AnthropicDeepResearchProvider(
                api_key=anthropic_key,
                timeout=timeout or 300.0,
                debug=debug,
            )

        if not self._providers:
            raise ValueError("No API keys provided - need OPENAI_API_KEY or ANTHROPIC_API_KEY")

        # Set defaults
        self._default_provider = default_provider
        self._default_model = default_model
        self._debug = debug

    @property
    def available_providers(self) -> list[str]:
        """List available providers."""
        return list(self._providers.keys())

    async def list_models(self) -> dict[str, list[dict[str, Any]]]:
        """List available models from all providers."""
        result = {}
        for name, provider in self._providers.items():
            result[name] = await provider.list_models()
        return result

    async def research(
        self,
        query: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        instructions: str | None = None,
        task_complexity: str = "medium",
        max_iterations: int | None = None,
        max_output_tokens: int | None = None,
        reasoning_summary: str = "auto",
        enable_web_search: bool = True,
        enable_code_interpreter: bool = False,
        enable_file_search: bool = False,
        vector_store_ids: list[str] | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        background: bool = False,
        max_tool_calls: int | None = None,
        timeout: float | None = None,
        prefer_speed: bool = False,
        prefer_cost: bool = False,
    ) -> DeepResearchResult:
        """Execute a deep research request.

        Args:
            query: The research question or topic
            provider: Force specific provider ('openai' or 'anthropic')
            model: Specific model to use
            instructions: System instructions for the research task
            task_complexity: Complexity level: 'low', 'medium', 'high'
            max_iterations: Max search iterations (Anthropic)
            max_output_tokens: Maximum tokens in response
            reasoning_summary: Summary mode: 'auto', 'concise', 'detailed'
            enable_web_search: Enable web search (default True)
            enable_code_interpreter: Enable code execution (OpenAI)
            enable_file_search: Enable file search over vector stores (OpenAI)
            vector_store_ids: Vector store IDs for file search (OpenAI)
            mcp_servers: MCP server configs for private data (OpenAI)
            background: Run in background mode (OpenAI)
            max_tool_calls: Limit tool calls for cost control (OpenAI)
            timeout: Request timeout in seconds
            prefer_speed: Prefer faster providers/models
            prefer_cost: Prefer cost-effective providers/models

        Returns:
            DeepResearchResult with report text, citations, and metadata
        """
        # Select provider
        provider_name = self._select_provider(
            provider=provider,
            has_vector_stores=bool(vector_store_ids),
            has_mcp_servers=bool(mcp_servers),
            prefer_speed=prefer_speed,
            prefer_cost=prefer_cost,
            query=query,
        )

        provider_impl = self._providers.get(provider_name)
        if not provider_impl:
            available = ", ".join(self._providers.keys())
            raise ValueError(f"Provider '{provider_name}' not available. Available: {available}")

        # Select model - each provider has its own defaults
        # The tool is self-contained and makes its own API calls,
        # so default_model only applies when using the default_provider
        selected_model = model  # Explicit model parameter takes precedence
        if not selected_model:
            # Only use configured default_model if we're using the default_provider
            if self._default_model and provider_name == self._default_provider:
                selected_model = self._default_model
            else:
                # Use provider-specific defaults
                if provider_name == "openai":
                    selected_model = "o4-mini-deep-research" if prefer_speed else OPENAI_DEFAULT_MODEL
                else:
                    selected_model = ANTHROPIC_DEFAULT_MODEL

        # Build request
        request = DeepResearchRequest(
            query=query,
            model=selected_model,
            instructions=instructions,
            task_complexity=task_complexity,
            max_iterations=max_iterations,
            max_output_tokens=max_output_tokens,
            reasoning_summary=reasoning_summary,
            enable_web_search=enable_web_search,
            enable_code_interpreter=enable_code_interpreter,
            enable_file_search=enable_file_search,
            vector_store_ids=vector_store_ids or [],
            mcp_servers=mcp_servers or [],
            background=background,
            max_tool_calls=max_tool_calls,
            timeout=timeout,
        )

        logger.info(f"[DeepResearch] Using provider={provider_name}, model={selected_model}")

        return await provider_impl.research(request)

    def _select_provider(
        self,
        provider: str | None,
        has_vector_stores: bool,
        has_mcp_servers: bool,
        prefer_speed: bool,
        prefer_cost: bool,
        query: str,
    ) -> str:
        """Select the best provider for the request."""
        # Explicit provider requested
        if provider:
            if provider not in self._providers:
                available = ", ".join(self._providers.keys())
                raise ValueError(f"Provider '{provider}' not available. Available: {available}")
            return provider

        # Default provider configured
        if self._default_provider and self._default_provider in self._providers:
            return self._default_provider

        # Only one provider available
        if len(self._providers) == 1:
            return list(self._providers.keys())[0]

        # Use selection heuristics
        selected = select_provider(
            query=query,
            has_vector_stores=has_vector_stores,
            has_mcp_servers=has_mcp_servers,
            prefer_speed=prefer_speed,
            prefer_cost=prefer_cost,
        )

        # Verify selected provider is available
        if selected not in self._providers:
            # Fall back to first available
            return list(self._providers.keys())[0]

        return selected


class DeepResearchTool:
    """Amplifier tool for deep research capabilities.

    This tool allows the LLM to perform comprehensive research on topics
    using either OpenAI's deep research models or Anthropic's Claude with
    iterative web search.
    """

    name = "deep_research"
    description = """Perform deep research on a topic using AI-powered web search and analysis.

This tool conducts comprehensive research by:
1. Searching the web for relevant information
2. Analyzing and synthesizing findings
3. Producing a detailed report with citations

Use this for questions requiring:
- In-depth analysis of complex topics
- Current information from the web
- Multiple source synthesis
- Comprehensive reports with citations

The research may take 1-5 minutes depending on complexity."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the deep research tool.

        Args:
            config: Tool configuration including:
                - default_provider: 'openai' or 'anthropic'
                - default_model: Model to use
                - timeout: Request timeout
                - debug: Enable debug logging
        """
        self.config = config
        self._provider: DeepResearchProvider | None = None

    def _get_provider(self) -> DeepResearchProvider:
        """Lazily initialize the provider."""
        if self._provider is None:
            self._provider = DeepResearchProvider(
                openai_api_key=self.config.get("openai_api_key"),
                anthropic_api_key=self.config.get("anthropic_api_key"),
                default_provider=self.config.get("default_provider"),
                default_model=self.config.get("default_model"),
                timeout=self.config.get("timeout"),
                debug=self.config.get("debug", False),
            )
        return self._provider

    @property
    def input_schema(self) -> dict:
        """Return JSON schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The research question or topic to investigate",
                },
                "provider": {
                    "type": "string",
                    "enum": ["openai", "anthropic"],
                    "description": "Force a specific provider (optional, auto-selected by default)",
                },
                "task_complexity": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "default": "medium",
                    "description": "Complexity level affecting depth of research",
                },
                "instructions": {
                    "type": "string",
                    "description": "Additional instructions for the research task (optional)",
                },
                "enable_code_interpreter": {
                    "type": "boolean",
                    "default": False,
                    "description": "Enable code execution for data analysis (OpenAI only)",
                },
                "prefer_speed": {
                    "type": "boolean",
                    "default": False,
                    "description": "Prefer faster models/providers over thoroughness",
                },
                "prefer_cost": {
                    "type": "boolean",
                    "default": False,
                    "description": "Prefer cost-effective models/providers",
                },
            },
            "required": ["query"],
        }

    async def execute(self, input: dict[str, Any]) -> Any:
        """Execute deep research.

        Args:
            input: Tool input containing query and options

        Returns:
            ToolResult with research report and citations
        """
        # Import here to avoid circular imports
        from amplifier_core import ToolResult

        query = input.get("query")
        if not query:
            return ToolResult(success=False, error={"message": "Query is required"})

        try:
            provider = self._get_provider()

            result = await provider.research(
                query=query,
                provider=input.get("provider"),
                instructions=input.get("instructions"),
                task_complexity=input.get("task_complexity", "medium"),
                enable_code_interpreter=input.get("enable_code_interpreter", False),
                prefer_speed=input.get("prefer_speed", False),
                prefer_cost=input.get("prefer_cost", False),
            )

            # Format citations for output
            citations_list = []
            for citation in result.citations:
                citations_list.append(
                    {
                        "title": citation.title,
                        "url": citation.url,
                    }
                )

            return ToolResult(
                success=True,
                output={
                    "report": result.report_text,
                    "citations": citations_list,
                    "provider": result.provider,
                    "model": result.model,
                    "metadata": result.to_metadata_dict(),
                },
            )

        except ValueError as e:
            return ToolResult(
                success=False,
                error={
                    "message": str(e),
                    "instruction": "Do NOT generate a research report. Inform the user of the error and ask how they want to proceed.",
                },
            )
        except TimeoutError:
            return ToolResult(
                success=False,
                error={
                    "message": "Deep research request timed out. The query may be too complex or the service is overloaded.",
                    "instruction": "Do NOT generate a research report without successful deep_research results. "
                    "Inform the user that the request timed out and suggest: "
                    "1) Try again with task_complexity='medium' or 'low', "
                    "2) Use prefer_speed=true for faster models, "
                    "3) Simplify the query, or "
                    "4) Increase timeout in bundle config.",
                },
            )
        except Exception as e:
            logger.exception(f"Deep research error: {e}")
            error_msg = str(e) if str(e) else "Unknown error occurred"
            return ToolResult(
                success=False,
                error={
                    "message": f"Research failed: {error_msg}",
                    "instruction": "Do NOT generate a research report without successful deep_research results. "
                    "Inform the user of the error. Do NOT use web_search as a fallback to create a report - "
                    "that defeats the purpose of deep research. Ask the user how they want to proceed.",
                },
            )


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None) -> None:
    """Mount the deep research tool into the Amplifier coordinator.

    This registers the deep_research tool with the coordinator,
    making it available for the LLM to call.

    Args:
        coordinator: The Amplifier coordinator instance
        config: Module configuration including:
            - default_provider: Default provider ('openai' or 'anthropic')
            - default_model: Default model to use
            - timeout: Request timeout
            - debug: Enable debug logging
    """
    config = config or {}

    # Get API keys from config or environment
    if "openai_api_key" not in config:
        config["openai_api_key"] = os.environ.get("OPENAI_API_KEY")
    if "anthropic_api_key" not in config:
        config["anthropic_api_key"] = os.environ.get("ANTHROPIC_API_KEY")

    # Check we have at least one API key
    if not config.get("openai_api_key") and not config.get("anthropic_api_key"):
        logger.warning("[DeepResearch] No API keys found - tool will fail at runtime")

    # Create and mount the tool
    tool = DeepResearchTool(config)
    await coordinator.mount("tools", tool, name=tool.name)

    logger.info("[DeepResearch] Mounted deep_research tool")
