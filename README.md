# amplifier-module-tool-deepresearch

Multi-provider deep research module for Amplifier, enabling automated research workflows with comprehensive citation support.

## Features

- **Multi-Provider Support**: Choose between OpenAI and Anthropic backends
- **OpenAI Deep Research**: Native o3/o4-mini models with web search, file search, code interpreter
- **Anthropic Deep Research**: Claude with extended thinking and iterative web search
- **Intelligent Provider Selection**: Auto-selects best provider based on query requirements
- **Prompt Utilities**: Clarification questions and prompt rewriting for better results
- **Background Mode**: Long-running research tasks (OpenAI)
- **Citation Tracking**: Full citation metadata with source URLs

## Installation

```bash
uv add git+https://github.com/michaeljabbour/amplifier-module-tool-deepresearch
```

## Quick Start

```python
from amplifier_module_tool_deepresearch import DeepResearchProvider

# Initialize with API keys (or use environment variables)
provider = DeepResearchProvider(
    openai_api_key="...",      # or OPENAI_API_KEY env var
    anthropic_api_key="...",   # or ANTHROPIC_API_KEY env var
)

# Run research
result = await provider.research(
    query="Research the economic impact of AI on healthcare",
    task_complexity="high",
)

print(result.report_text)
for citation in result.citations:
    print(f"  - {citation.title}: {citation.url}")
```

## Provider Selection

The module automatically selects the best provider based on:

| Requirement | Provider |
|-------------|----------|
| Vector stores (file search) | OpenAI |
| MCP servers | OpenAI |
| High complexity research | OpenAI |
| Speed preference | Anthropic |
| Cost preference | Anthropic |

You can also explicitly specify a provider:

```python
result = await provider.research(
    query="...",
    provider="anthropic",  # Force Anthropic
)
```

## OpenAI Deep Research

Uses OpenAI's specialized deep research models:

- `o3-deep-research`: Comprehensive, highest quality synthesis
- `o4-mini-deep-research`: Faster, cost-effective option

### Features

```python
result = await provider.research(
    query="Analyze market trends",
    provider="openai",
    model="o3-deep-research",
    
    # Tools
    enable_web_search=True,
    enable_code_interpreter=True,  # Data analysis
    enable_file_search=True,
    vector_store_ids=["vs_123"],   # Your vector stores
    
    # MCP for private data
    mcp_servers=[{
        "label": "internal_docs",
        "url": "https://your-mcp-server.com/sse/",
    }],
    
    # Performance
    background=True,      # Async execution
    max_tool_calls=50,    # Cost control
)
```

## Anthropic Deep Research

Uses Claude with iterative web search and extended thinking:

- `claude-sonnet-4-5-20250929`: Claude Sonnet 4.5 with extended thinking

### Features

```python
result = await provider.research(
    query="Research quantum computing applications",
    provider="anthropic",
    task_complexity="high",  # Controls iterations (2/5/8)
    max_iterations=10,       # Override default
)

# Access reasoning steps
for step in result.reasoning_steps:
    print(f"Iteration {step.iteration}: {step.text[:100]}...")
```

## Prompt Utilities

Improve research quality with clarification and rewriting:

```python
from amplifier_module_tool_deepresearch import (
    generate_clarifying_questions,
    rewrite_research_prompt,
    estimate_task_complexity,
)

# Get clarifying questions
questions = await generate_clarifying_questions(
    client,
    "Research surfboards for me",
)
print(questions)  # Asks about skill level, budget, location, etc.

# Rewrite prompt for better results
detailed_prompt = await rewrite_research_prompt(
    client,
    user_query="Research surfboards",
    clarifications="Beginner, $500 budget, California",
)

# Estimate complexity
complexity = estimate_task_complexity(query)  # low/medium/high
```

## Amplifier Integration

### Bundle Configuration

Add to your bundle's `tools:` section:

```yaml
tools:
  - module: tool-deepresearch
    source: git+https://github.com/michaeljabbour/amplifier-module-tool-deepresearch@main
    config:
      default_provider: anthropic    # or "openai"
      default_model: claude-sonnet-4-5-20250929  # optional, provider uses its default
      timeout: 600                   # seconds (default: 600)
```

### Programmatic Mount

```python
from amplifier_module_tool_deepresearch import mount

await mount(coordinator, {
    "default_provider": "anthropic",
    "timeout": 600,
})
```

### Usage in Session

Once mounted, the `deep_research` tool is available to the LLM:

```
Use deep research to analyze the current state of quantum computing
Use deep research with openai to find recent news about Tesla
Use deep research with anthropic and task_complexity high to research mRNA vaccines
```

## Response Structure

```python
@dataclass
class DeepResearchResult:
    report_text: str           # Main research report
    provider: str              # "openai" or "anthropic"
    model: str                 # Model used
    
    # Token usage
    input_tokens: int
    output_tokens: int
    
    # Citations with full metadata
    citations: list[Citation]  # title, url, start_index, end_index
    
    # Research process details
    reasoning_steps: list[ReasoningStep]
    web_searches: list[WebSearchStep]
    file_searches: list[FileSearchStep]
    code_executions: list[CodeExecutionStep]
    
    # Iteration tracking (Anthropic)
    iterations_completed: int
    max_iterations: int
```

## When to Use Each Provider

### Use OpenAI Deep Research when:
- Legal or scientific research requiring extensive citations
- Market analysis with data synthesis from multiple sources
- Working with internal data (vector stores, MCP servers)
- Need comprehensive multi-source reports
- Background/async execution is beneficial

### Use Anthropic Deep Research when:
- More conversational/interactive research needed
- Extended thinking benefits complex reasoning
- Using other Anthropic tools in your workflow
- Cost-sensitive scenarios (iterative approach)
- Faster turnaround needed

## License

MIT
