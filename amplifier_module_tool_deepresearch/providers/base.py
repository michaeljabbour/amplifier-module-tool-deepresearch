"""Base class for deep research providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Citation:
    """A citation from a deep research response."""

    title: str
    url: str
    start_index: int = 0
    end_index: int = 0


@dataclass
class WebSearchStep:
    """A web search step from the research process."""

    query: str
    status: str = "completed"
    action_type: str = "search"  # search, open_page, find_in_page


@dataclass
class FileSearchStep:
    """A file search step over vector stores."""

    query: str
    vector_store_ids: list[str] = field(default_factory=list)
    status: str = "completed"


@dataclass
class CodeExecutionStep:
    """A code execution step from the research process."""

    input_code: str
    output: str | None = None


@dataclass
class ReasoningStep:
    """A reasoning/thinking step from the research process."""

    text: str
    iteration: int = 0


@dataclass
class DeepResearchResult:
    """Unified result from any deep research provider.

    Contains the research report and all metadata about the research process.
    """

    # Main output
    report_text: str

    # Provider info
    provider: str
    model: str

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0

    # Response metadata
    response_id: str | None = None
    status: str | None = None

    # Research process details
    citations: list[Citation] = field(default_factory=list)
    reasoning_steps: list[ReasoningStep] = field(default_factory=list)
    web_searches: list[WebSearchStep] = field(default_factory=list)
    file_searches: list[FileSearchStep] = field(default_factory=list)
    code_executions: list[CodeExecutionStep] = field(default_factory=list)

    # Iteration tracking (for Anthropic iterative search)
    iterations_completed: int = 0
    max_iterations: int = 0

    # Raw response for debugging
    raw_response: Any = None

    def to_metadata_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for ChatResponse metadata."""
        result: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
        }

        if self.response_id:
            result["response_id"] = self.response_id
        if self.status:
            result["status"] = self.status

        if self.citations:
            result["citations"] = [
                {
                    "title": c.title,
                    "url": c.url,
                    "start_index": c.start_index,
                    "end_index": c.end_index,
                }
                for c in self.citations
            ]

        if self.reasoning_steps:
            result["reasoning_steps"] = [{"text": r.text, "iteration": r.iteration} for r in self.reasoning_steps]

        if self.web_searches:
            result["web_searches"] = [
                {"query": w.query, "status": w.status, "action_type": w.action_type} for w in self.web_searches
            ]

        if self.file_searches:
            result["file_searches"] = [
                {"query": f.query, "vector_store_ids": f.vector_store_ids, "status": f.status}
                for f in self.file_searches
            ]

        if self.code_executions:
            result["code_executions"] = [{"input": c.input_code, "output": c.output} for c in self.code_executions]

        if self.iterations_completed > 0:
            result["iterations"] = {
                "completed": self.iterations_completed,
                "max": self.max_iterations,
            }

        return result


@dataclass
class DeepResearchRequest:
    """Unified request for deep research across providers.

    Encapsulates all options that may be used by different providers.
    """

    # Required
    query: str

    # Model selection
    model: str | None = None

    # System/developer instructions
    instructions: str | None = None

    # Output configuration
    max_output_tokens: int | None = None
    reasoning_summary: str = "auto"  # auto, concise, detailed

    # Task complexity (for Anthropic iterative search)
    task_complexity: str = "medium"  # low, medium, high
    max_iterations: int | None = None

    # OpenAI-specific: Tool configuration
    enable_web_search: bool = True
    enable_code_interpreter: bool = False
    enable_file_search: bool = False

    # OpenAI-specific: Vector stores for file search
    vector_store_ids: list[str] = field(default_factory=list)

    # OpenAI-specific: MCP servers
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)

    # OpenAI-specific: Background mode
    background: bool = False

    # OpenAI-specific: Limit tool calls for cost/latency control
    max_tool_calls: int | None = None

    # Timeout override
    timeout: float | None = None


class DeepResearchProviderBase(ABC):
    """Abstract base class for deep research providers."""

    name: str = "base"

    @abstractmethod
    async def research(self, request: DeepResearchRequest) -> DeepResearchResult:
        """Execute a deep research request.

        Args:
            request: The research request with all configuration

        Returns:
            DeepResearchResult with report and metadata
        """
        ...

    @abstractmethod
    async def list_models(self) -> list[dict[str, Any]]:
        """List available models for this provider.

        Returns:
            List of model info dictionaries
        """
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available (has credentials)."""
        ...
