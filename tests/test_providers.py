"""Tests for deep research providers."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_module_tool_deepresearch import (
    DeepResearchProvider,
    DeepResearchRequest,
    DeepResearchResult,
    estimate_task_complexity,
    select_provider,
)
from amplifier_module_tool_deepresearch.providers import (
    AnthropicDeepResearchProvider,
    Citation,
    OpenAIDeepResearchProvider,
)

from .conftest import (
    create_openai_response_with_code_execution,
    create_openai_response_with_file_search,
)


class TestOpenAIProvider:
    """Tests for OpenAI Deep Research provider."""

    @pytest.mark.asyncio
    async def test_research_basic(self, mock_openai_client: MagicMock, openai_research_response: MagicMock) -> None:
        """Test basic research request."""
        mock_openai_client.responses.create = AsyncMock(return_value=openai_research_response)

        provider = OpenAIDeepResearchProvider(client=mock_openai_client)
        request = DeepResearchRequest(query="Test research query")

        result = await provider.research(request)

        assert result.provider == "openai"
        assert result.report_text == "# Research Report\n\nAI has significant economic impact..."
        assert len(result.citations) == 1
        assert result.citations[0].url == "https://example.com/ai-study"
        assert len(result.web_searches) == 1
        assert result.web_searches[0].query == "economic impact of AI"

    @pytest.mark.asyncio
    async def test_research_with_code_interpreter(self, mock_openai_client: MagicMock) -> None:
        """Test research with code interpreter enabled."""
        mock_openai_client.responses.create = AsyncMock(return_value=create_openai_response_with_code_execution())

        provider = OpenAIDeepResearchProvider(client=mock_openai_client)
        request = DeepResearchRequest(
            query="Analyze the data",
            enable_code_interpreter=True,
        )

        result = await provider.research(request)

        assert len(result.code_executions) == 1
        assert "pandas" in result.code_executions[0].input_code

    @pytest.mark.asyncio
    async def test_research_with_file_search(self, mock_openai_client: MagicMock) -> None:
        """Test research with file search over vector stores."""
        mock_openai_client.responses.create = AsyncMock(return_value=create_openai_response_with_file_search())

        provider = OpenAIDeepResearchProvider(client=mock_openai_client)
        request = DeepResearchRequest(
            query="Research internal documents",
            enable_file_search=True,
            vector_store_ids=["vs_123", "vs_456"],
        )

        result = await provider.research(request)

        assert len(result.file_searches) == 1
        mock_openai_client.responses.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_models(self, mock_openai_client: MagicMock) -> None:
        """Test listing available models."""
        provider = OpenAIDeepResearchProvider(client=mock_openai_client)
        models = await provider.list_models()

        assert len(models) == 4
        model_ids = [m["id"] for m in models]
        assert "o3-deep-research" in model_ids
        assert "o4-mini-deep-research" in model_ids

    def test_is_available(self, mock_openai_client: MagicMock) -> None:
        """Test availability check."""
        provider = OpenAIDeepResearchProvider(client=mock_openai_client)
        assert provider.is_available is True

        provider_no_client = OpenAIDeepResearchProvider()
        assert provider_no_client.is_available is False


class TestAnthropicProvider:
    """Tests for Anthropic Deep Research provider."""

    @pytest.mark.asyncio
    async def test_research_basic(
        self, mock_anthropic_client: MagicMock, anthropic_research_response: MagicMock
    ) -> None:
        """Test basic iterative research request."""
        mock_anthropic_client.messages.create = AsyncMock(return_value=anthropic_research_response)

        provider = AnthropicDeepResearchProvider(client=mock_anthropic_client)
        request = DeepResearchRequest(query="Test research query", task_complexity="medium")

        result = await provider.research(request)

        assert result.provider == "anthropic"
        assert "Research Findings" in result.report_text
        assert len(result.reasoning_steps) >= 1
        assert result.iterations_completed >= 1

    @pytest.mark.asyncio
    async def test_research_high_complexity(
        self, mock_anthropic_client: MagicMock, anthropic_research_response: MagicMock
    ) -> None:
        """Test high complexity research with more iterations."""
        mock_anthropic_client.messages.create = AsyncMock(return_value=anthropic_research_response)

        provider = AnthropicDeepResearchProvider(client=mock_anthropic_client)
        request = DeepResearchRequest(
            query="Complex research requiring deep analysis",
            task_complexity="high",
        )

        result = await provider.research(request)

        assert result.provider == "anthropic"
        assert result.max_iterations == 8  # High complexity = 8 iterations

    @pytest.mark.asyncio
    async def test_citation_extraction(self, mock_anthropic_client: MagicMock) -> None:
        """Test extraction of citations from markdown text."""
        provider = AnthropicDeepResearchProvider(client=mock_anthropic_client)

        text = "According to [Study A](https://example.com/a) and [Study B](https://example.com/b)..."
        citations = provider._extract_citations_from_text(text)

        assert len(citations) == 2
        assert citations[0].title == "Study A"
        assert citations[0].url == "https://example.com/a"
        assert citations[1].title == "Study B"

    @pytest.mark.asyncio
    async def test_list_models(self, mock_anthropic_client: MagicMock) -> None:
        """Test listing available models."""
        provider = AnthropicDeepResearchProvider(client=mock_anthropic_client)
        models = await provider.list_models()

        assert len(models) == 3
        model_ids = [m["id"] for m in models]
        assert "claude-sonnet-4-5-20250514" in model_ids

    def test_is_available(self, mock_anthropic_client: MagicMock) -> None:
        """Test availability check."""
        provider = AnthropicDeepResearchProvider(client=mock_anthropic_client)
        assert provider.is_available is True


class TestUnifiedProvider:
    """Tests for the unified DeepResearchProvider."""

    def test_init_with_openai_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization with OpenAI key only."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        provider = DeepResearchProvider()
        assert "openai" in provider.available_providers
        assert "anthropic" not in provider.available_providers

    def test_init_with_anthropic_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization with Anthropic key only."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        provider = DeepResearchProvider()
        assert "anthropic" in provider.available_providers
        assert "openai" not in provider.available_providers

    def test_init_with_both_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization with both API keys."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

        provider = DeepResearchProvider()
        assert "openai" in provider.available_providers
        assert "anthropic" in provider.available_providers

    def test_init_no_keys_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that initialization without keys raises error."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with pytest.raises(ValueError, match="No API keys provided"):
            DeepResearchProvider()


class TestPromptUtilities:
    """Tests for prompt utilities."""

    def test_estimate_complexity_high(self) -> None:
        """Test high complexity estimation."""
        query = "Provide a comprehensive analysis comparing and contrasting the economic impact of multiple AI systems"
        complexity = estimate_task_complexity(query)
        assert complexity == "high"

    def test_estimate_complexity_low(self) -> None:
        """Test low complexity estimation."""
        query = "What is machine learning?"
        complexity = estimate_task_complexity(query)
        assert complexity == "low"

    def test_estimate_complexity_medium(self) -> None:
        """Test medium complexity estimation."""
        query = "Explain how neural networks work in image recognition"
        complexity = estimate_task_complexity(query)
        assert complexity == "medium"

    def test_select_provider_with_vector_stores(self) -> None:
        """Test provider selection with vector stores."""
        provider = select_provider(
            query="Research query",
            has_vector_stores=True,
        )
        assert provider == "openai"

    def test_select_provider_with_mcp(self) -> None:
        """Test provider selection with MCP servers."""
        provider = select_provider(
            query="Research query",
            has_mcp_servers=True,
        )
        assert provider == "openai"

    def test_select_provider_prefer_speed(self) -> None:
        """Test provider selection preferring speed."""
        provider = select_provider(
            query="Quick research query",
            prefer_speed=True,
        )
        assert provider == "anthropic"

    def test_select_provider_prefer_cost(self) -> None:
        """Test provider selection preferring cost."""
        provider = select_provider(
            query="Research query",
            prefer_cost=True,
        )
        assert provider == "anthropic"


class TestDataClasses:
    """Tests for data classes."""

    def test_citation(self) -> None:
        """Test Citation dataclass."""
        citation = Citation(
            title="Test Study",
            url="https://example.com",
            start_index=10,
            end_index=50,
        )
        assert citation.title == "Test Study"
        assert citation.url == "https://example.com"

    def test_deep_research_result_metadata(self) -> None:
        """Test DeepResearchResult metadata conversion."""
        result = DeepResearchResult(
            report_text="Test report",
            provider="openai",
            model="o3-deep-research",
            input_tokens=100,
            output_tokens=500,
            citations=[Citation(title="Test", url="https://example.com")],
        )

        metadata = result.to_metadata_dict()

        assert metadata["provider"] == "openai"
        assert metadata["model"] == "o3-deep-research"
        assert len(metadata["citations"]) == 1
        assert metadata["citations"][0]["title"] == "Test"

    def test_deep_research_request_defaults(self) -> None:
        """Test DeepResearchRequest default values."""
        request = DeepResearchRequest(query="Test query")

        assert request.enable_web_search is True
        assert request.enable_code_interpreter is False
        assert request.task_complexity == "medium"
        assert request.reasoning_summary == "auto"
