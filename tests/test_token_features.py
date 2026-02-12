"""Tests for token savings calculator, burn rate, and model router features."""

import pytest
from pathlib import Path
import tempfile
import json
from click.testing import CliRunner

from know.stats import TokenSavingsCalculator, StatsTracker
from know.model_router import SmartModelRouter, ModelInfo
from know.config import Config


class TestTokenSavingsCalculator:
    """Test TokenSavingsCalculator class."""
    
    def test_basic_savings_calculation(self):
        """Test basic token savings calculation."""
        calc = TokenSavingsCalculator()
        result = calc.calculate_savings(used_tokens=10000)
        
        assert result["tokens_used"] == 10000
        assert result["tokens_naive"] == 50000
        assert result["tokens_saved"] == 40000
        assert result["percent_saved"] == 80.0
    
    def test_custom_naive_tokens(self):
        """Test with custom naive token count."""
        calc = TokenSavingsCalculator()
        result = calc.calculate_savings(used_tokens=5000, naive_tokens=10000)
        
        assert result["tokens_naive"] == 10000
        assert result["tokens_saved"] == 5000
        assert result["percent_saved"] == 50.0
    
    def test_cost_savings_claude(self):
        """Test dollar savings for Claude pricing."""
        calc = TokenSavingsCalculator()
        result = calc.calculate_savings(used_tokens=10000, model="claude")
        
        # 40000 tokens saved * $15/1M = $0.60
        assert result["dollar_savings"] == 0.6
        assert result["model"] == "claude"
    
    def test_cost_savings_gpt4(self):
        """Test dollar savings for GPT-4 pricing."""
        calc = TokenSavingsCalculator()
        result = calc.calculate_savings(used_tokens=10000, model="gpt4")
        
        # 40000 tokens saved * $10/1M = $0.40
        assert result["dollar_savings"] == 0.4
        assert result["model"] == "gpt4"
    
    def test_cumulative_savings(self):
        """Test cumulative savings across multiple events."""
        calc = TokenSavingsCalculator()
        events = [
            {"tokens_used": 5000},
            {"tokens_used": 8000},
            {"tokens_used": 3000},
        ]
        result = calc.calculate_cumulative_savings(events)
        
        assert result["tokens_used"] == 16000
        assert result["tokens_naive"] == 150000  # 3 events * 50000


class TestSmartModelRouter:
    """Test SmartModelRouter class."""
    
    def test_route_simple_task(self):
        """Test routing for simple task."""
        router = SmartModelRouter()
        model = router.route("summarize this document", quality_threshold=0.8)
        
        assert model == "gemini-1.5-flash"
    
    def test_route_complex_task(self):
        """Test routing for complex task with higher threshold."""
        router = SmartModelRouter()
        model = router.route("design a new architecture", quality_threshold=0.9)
        
        # Should choose cheapest model with quality >= 0.9
        assert model in ["gemini-1.5-pro", "gpt-4o", "claude-3-sonnet", "claude-3-opus"]
    
    def test_route_expert_task(self):
        """Test routing for expert-level task."""
        router = SmartModelRouter()
        model = router.route("research novel approaches to quantum computing")
        
        # Expert tasks get auto-elevated threshold
        assert router.get_model(model).quality_score >= 0.9
    
    def test_complexity_detection(self):
        """Test task complexity detection."""
        router = SmartModelRouter()
        
        assert router._detect_complexity("summarize this") == "simple"
        assert router._detect_complexity("explain the codebase") == "moderate"
        assert router._detect_complexity("design a new architecture") == "complex"
        assert router._detect_complexity("research comprehensive solutions") == "expert"
    
    def test_route_with_cost_calculation(self):
        """Test routing with cost breakdown."""
        router = SmartModelRouter()
        result = router.route_with_cost(
            task="analyze this code",
            input_tokens=10000,
            output_tokens=1000,
            quality_threshold=0.8
        )
        
        assert "recommended" in result
        assert "recommended_cost" in result
        assert "all_options" in result
        assert len(result["all_options"]) > 0
    
    def test_list_models(self):
        """Test listing all models."""
        router = SmartModelRouter()
        models = router.list_models()
        
        assert len(models) == 7  # 7 default models
        # Should be sorted by cost
        costs = [m["cost_per_1m_input"] for m in models]
        assert costs == sorted(costs)
    
    def test_get_model(self):
        """Test getting specific model info."""
        router = SmartModelRouter()
        model = router.get_model("gpt-4o")
        
        assert model is not None
        assert model.name == "gpt-4o"
        assert model.quality_score == 0.95
    
    def test_custom_models(self):
        """Test with custom model configurations."""
        router = SmartModelRouter(custom_models={
            "custom-model": ModelInfo(
                name="custom-model",
                cost_per_1m_input=0.01,
                cost_per_1m_output=0.02,
                quality_score=0.99,
                max_context=100000,
                strengths=["custom"],
            )
        })
        
        model = router.get_model("custom-model")
        assert model is not None
        assert model.name == "custom-model"
        
        # Should choose custom model for high quality needs
        result = router.route("expert research", quality_threshold=0.95)
        assert result == "custom-model"


class TestBurnRate:
    """Test burn rate functionality."""
    
    @pytest.fixture
    def temp_project(self):
        """Create a temporary project directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".know").mkdir()
            yield root
    
    @pytest.fixture
    def config(self, temp_project):
        """Create a test config."""
        return Config.create_default(temp_project)
    
    def test_get_burn_rate_empty(self, config):
        """Test burn rate with no data."""
        tracker = StatsTracker(config)
        burn = tracker.get_burn_rate(days=30)
        
        assert burn["total_tokens"] == 0
        assert burn["total_queries"] == 0
    
    def test_get_burn_rate_with_data(self, config):
        """Test burn rate with recorded context calls."""
        tracker = StatsTracker(config)
        
        # Record some context calls
        tracker.record_context("query1", budget=8000, tokens_used=5000, duration_ms=100)
        tracker.record_context("query2", budget=8000, tokens_used=6000, duration_ms=150)
        
        burn = tracker.get_burn_rate(days=30)
        
        assert burn["total_tokens"] == 11000
        assert burn["total_queries"] == 2
        assert "projections" in burn
        assert "savings" in burn
    
    def test_project_breakdown_single(self, config):
        """Test project breakdown for single project."""
        tracker = StatsTracker(config)
        tracker.record_context("query1", budget=8000, tokens_used=5000, duration_ms=100)
        
        projects = tracker.get_project_breakdown()
        
        assert len(projects) == 1
        assert projects[0]["tokens"] == 5000


class TestCLICommands:
    """Test CLI commands for new features."""
    
    def test_route_list_command(self):
        """Test 'know route --list' command."""
        from know.cli import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ["route", "--list"])
        
        assert result.exit_code == 0
        assert "gemini-1.5-flash" in result.output
        assert "gpt-4o" in result.output
    
    def test_route_task_command(self):
        """Test 'know route <task>' command."""
        from know.cli import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ["route", "summarize this document"])
        
        assert result.exit_code == 0
        assert "Recommended" in result.output
        assert "gemini-1.5-flash" in result.output
    
    def test_route_json_output(self):
        """Test 'know route --json' output."""
        from know.cli import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ["--json", "route", "analyze this"])
        
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "recommended" in data
    
    def test_burnrate_command(self):
        """Test 'know burnrate' command."""
        from know.cli import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ["burnrate", "--days", "30"])
        
        assert result.exit_code == 0
        # Should contain dashboard elements
        assert "Burn Rate" in result.output or "Usage" in result.output or result.output == "{}\n"


class TestModelInfo:
    """Test ModelInfo dataclass."""
    
    def test_total_cost_calculation(self):
        """Test total cost calculation."""
        model = ModelInfo(
            name="test-model",
            cost_per_1m_input=1.0,
            cost_per_1m_output=2.0,
            quality_score=0.9,
            max_context=100000,
            strengths=["test"],
        )
        
        # 100k input + 10k output = $0.1 + $0.02 = $0.12
        cost = model.total_cost(100000, 10000)
        assert abs(cost - 0.12) < 0.0001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])