"""Prompt utilities for deep research.

Provides clarification and prompt rewriting capabilities to improve
research quality. These replicate the ChatGPT Deep Research workflow
where an intermediate model clarifies intent before research begins.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Clarification Prompts
# =============================================================================

CLARIFICATION_INSTRUCTIONS = """
You are talking to a user who is asking for a research task to be conducted.
Your job is to gather more information from the user to successfully complete the task.

GUIDELINES:
- Be concise while gathering all necessary information
- Make sure to gather all the information needed to carry out the research task
- Use bullet points or numbered lists if appropriate for clarity
- Don't ask for unnecessary information, or information that the user has already provided
- Prioritize the 3-6 questions that would most reduce ambiguity

IMPORTANT: Do NOT conduct any research yourself, just gather information that will
be given to a researcher to conduct the research task.
"""


REWRITE_INSTRUCTIONS = """
You will be given a research task by a user. Your job is to produce a set of
instructions for a researcher that will complete the task. Do NOT complete the
task yourself, just provide instructions on how to complete it.

GUIDELINES:
1. **Maximize Specificity and Detail**
   - Include all known user preferences and explicitly list key attributes
   - All details from the user must be included in the instructions

2. **Fill in Unstated But Necessary Dimensions as Open-Ended**
   - If certain attributes are essential but not provided, state they are open-ended
   - Default to no specific constraint where user hasn't specified

3. **Avoid Unwarranted Assumptions**
   - If the user has not provided a detail, do not invent one
   - Guide the researcher to treat unspecified items as flexible

4. **Use the First Person**
   - Phrase the request from the perspective of the user

5. **Tables**
   - If tables will help organize the research output, explicitly request them
   - Examples: product comparisons, competitor analysis, budget planning

6. **Headers and Formatting**
   - Include expected output format in the prompt
   - Ask for structured format (report, plan, etc.) when appropriate

7. **Language**
   - If user input is non-English, tell researcher to respond in that language
   - Unless user explicitly asks for a different language

8. **Sources**
   - Specify source preferences if relevant
   - For products/travel: prefer official/primary websites
   - For academic queries: prefer original papers over summaries
   - Prioritize sources in the query's language
"""


async def generate_clarifying_questions(
    client: Any,
    user_query: str,
    model: str = "gpt-4.1",
    max_questions: int = 5,
) -> str:
    """Generate clarifying questions for a research query.

    Uses a fast model to ask follow-up questions before conducting
    deep research, improving result quality and relevance.

    Args:
        client: OpenAI or Anthropic client
        user_query: The user's original research query
        model: Model to use for clarification (fast/cheap model recommended)
        max_questions: Maximum number of questions to generate

    Returns:
        String containing clarifying questions to ask the user
    """
    instructions = CLARIFICATION_INSTRUCTIONS + f"\n\nLimit to {max_questions} questions maximum."

    # Detect client type and call appropriately
    if hasattr(client, "responses"):
        # OpenAI client
        response = await client.responses.create(
            model=model,
            input=user_query,
            instructions=instructions,
        )
        return response.output_text if hasattr(response, "output_text") else str(response.output[-1].content[0].text)
    elif hasattr(client, "messages"):
        # Anthropic client
        response = await client.messages.create(
            model=model.replace("gpt-4.1", "claude-3-5-haiku-latest"),  # Map to fast Anthropic model
            max_tokens=1024,
            system=instructions,
            messages=[{"role": "user", "content": user_query}],
        )
        return response.content[0].text
    else:
        raise ValueError("Unknown client type")


async def rewrite_research_prompt(
    client: Any,
    user_query: str,
    clarifications: str | None = None,
    model: str = "gpt-4.1",
) -> str:
    """Rewrite a user query into detailed research instructions.

    Takes the original query (and optional clarifications) and produces
    a well-structured prompt optimized for deep research.

    Args:
        client: OpenAI or Anthropic client
        user_query: The user's original research query
        clarifications: Optional user responses to clarifying questions
        model: Model to use for rewriting (fast model recommended)

    Returns:
        Rewritten, detailed research instructions
    """
    # Combine query with any clarifications
    full_input = user_query
    if clarifications:
        full_input = f"Original query: {user_query}\n\nUser clarifications: {clarifications}"

    # Detect client type and call appropriately
    if hasattr(client, "responses"):
        # OpenAI client
        response = await client.responses.create(
            model=model,
            input=full_input,
            instructions=REWRITE_INSTRUCTIONS,
        )
        return response.output_text if hasattr(response, "output_text") else str(response.output[-1].content[0].text)
    elif hasattr(client, "messages"):
        # Anthropic client
        response = await client.messages.create(
            model=model.replace("gpt-4.1", "claude-3-5-haiku-latest"),
            max_tokens=2048,
            system=REWRITE_INSTRUCTIONS,
            messages=[{"role": "user", "content": full_input}],
        )
        return response.content[0].text
    else:
        raise ValueError("Unknown client type")


def estimate_task_complexity(query: str) -> str:
    """Estimate task complexity based on query characteristics.

    Simple heuristic to determine appropriate research depth.

    Args:
        query: The research query

    Returns:
        Complexity level: 'low', 'medium', or 'high'
    """
    query_lower = query.lower()
    word_count = len(query.split())

    # High complexity indicators
    high_indicators = [
        "comprehensive",
        "in-depth",
        "analyze",
        "compare and contrast",
        "economic impact",
        "market analysis",
        "legal",
        "scientific",
        "research",
        "synthesize",
        "multiple perspectives",
        "conflicting",
    ]

    # Low complexity indicators
    low_indicators = [
        "what is",
        "define",
        "when did",
        "who is",
        "simple",
        "quick",
        "brief",
    ]

    high_count = sum(1 for indicator in high_indicators if indicator in query_lower)
    low_count = sum(1 for indicator in low_indicators if indicator in query_lower)

    # Decision logic
    if high_count >= 2 or word_count > 50:
        return "high"
    elif low_count >= 1 and word_count < 20:
        return "low"
    else:
        return "medium"


def select_provider(
    query: str,
    has_vector_stores: bool = False,
    has_mcp_servers: bool = False,
    prefer_speed: bool = False,
    prefer_cost: bool = False,
) -> str:
    """Select the best provider based on query and requirements.

    Args:
        query: The research query
        has_vector_stores: Whether vector stores are configured
        has_mcp_servers: Whether MCP servers are configured
        prefer_speed: Whether to prefer faster responses
        prefer_cost: Whether to prefer lower cost

    Returns:
        Provider name: 'openai' or 'anthropic'
    """
    # OpenAI is required for vector stores and certain MCP configurations
    if has_vector_stores:
        logger.info("Selecting OpenAI: vector stores require file_search tool")
        return "openai"

    # OpenAI Deep Research has native MCP with search/fetch interface
    if has_mcp_servers:
        logger.info("Selecting OpenAI: MCP servers configured")
        return "openai"

    # Estimate complexity
    complexity = estimate_task_complexity(query)

    # For high complexity, OpenAI's native deep research is more thorough
    if complexity == "high" and not prefer_speed and not prefer_cost:
        logger.info("Selecting OpenAI: high complexity task")
        return "openai"

    # For speed/cost sensitive, Anthropic with iterative search can be faster
    if prefer_speed or prefer_cost:
        logger.info("Selecting Anthropic: speed/cost preference")
        return "anthropic"

    # Default to OpenAI for comprehensive research
    return "openai"
