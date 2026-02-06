"""Constants for the deep research module.

Shared constants across all provider implementations.
"""

from enum import Enum


class DeepResearchProvider(str, Enum):
    """Available deep research providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class TaskComplexity(str, Enum):
    """Task complexity levels for research depth."""

    LOW = "low"  # Quick factual lookups, definitions
    MEDIUM = "medium"  # Multiple perspectives, historical context
    HIGH = "high"  # In-depth research, specialized domains


# =============================================================================
# OpenAI Deep Research Constants
# =============================================================================

# Available OpenAI Deep Research models
OPENAI_MODELS = [
    "o3-deep-research",
    "o3-deep-research-2025-06-26",
    "o4-mini-deep-research",
    "o4-mini-deep-research-2025-06-26",
]

# Default OpenAI model
OPENAI_DEFAULT_MODEL = "o3-deep-research"

# OpenAI timeout (Deep Research can take 15+ minutes for complex queries)
# This is the asyncio.wait_for timeout - httpx client timeout is set separately
OPENAI_DEFAULT_TIMEOUT = 1800.0  # 30 minutes

# Polling interval for background mode (seconds)
OPENAI_BACKGROUND_POLL_INTERVAL = 5.0

# Max polling attempts before giving up
OPENAI_BACKGROUND_MAX_POLLS = 360  # 30 minutes at 5s intervals

# OpenAI max output tokens
OPENAI_DEFAULT_MAX_TOKENS = 100000

# =============================================================================
# Anthropic Deep Research Constants
# =============================================================================

# Anthropic models for deep research
ANTHROPIC_MODELS = [
    "claude-sonnet-4-5-20250929",  # Latest Claude Sonnet 4.5
    "claude-sonnet-4-5",  # Alias (auto-updates)
    "claude-opus-4-5-20251101",  # Premium model
]

# Default Anthropic model for research
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

# Anthropic timeout
ANTHROPIC_DEFAULT_TIMEOUT = 300.0

# Anthropic max tokens
ANTHROPIC_DEFAULT_MAX_TOKENS = 16384

# Max iterations for Anthropic iterative search
ANTHROPIC_DEFAULT_MAX_ITERATIONS = 5

# Thinking budget tokens by complexity
ANTHROPIC_THINKING_BUDGET = {
    TaskComplexity.LOW: 1024,
    TaskComplexity.MEDIUM: 2048,
    TaskComplexity.HIGH: 4096,
}

# =============================================================================
# Shared Constants
# =============================================================================

# Default reasoning summary mode
DEFAULT_REASONING_SUMMARY = "auto"

# Debug truncation length
DEFAULT_DEBUG_TRUNCATE_LENGTH = 500

# Metadata keys
METADATA_RESPONSE_ID = "response_id"
METADATA_PROVIDER = "provider"
METADATA_STATUS = "status"
METADATA_CITATIONS = "citations"
METADATA_REASONING_STEPS = "reasoning_steps"
METADATA_WEB_SEARCHES = "web_searches"
METADATA_CODE_EXECUTIONS = "code_executions"
METADATA_FILE_SEARCHES = "file_searches"
METADATA_ITERATIONS = "iterations"

# =============================================================================
# Provider Selection Heuristics
# =============================================================================

# When to prefer OpenAI Deep Research:
# - Legal or scientific research requiring extensive citations
# - Market analysis with data synthesis
# - Large bodies of internal data (file search/vector stores)
# - Need for comprehensive multi-source reports
# - Background/async execution needed

# When to prefer Anthropic Deep Research:
# - More conversational/interactive research
# - Extended thinking for complex reasoning
# - When using other Anthropic tools in workflow
# - Cost-sensitive scenarios (iterative approach)
# - Faster turnaround needed (o4-mini alternative available in both)
