"""Tests for stats tracking functionality."""

import pytest
from pathlib import Path
import tempfile

from know.stats import StatsTracker
from know.config import Config


class TestStatsTracker:
    """Test StatsTracker class."""

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

        tracker.record_context("query1", budget=8000, tokens_used=5000, duration_ms=100)
        tracker.record_context("query2", budget=8000, tokens_used=6000, duration_ms=150)

        burn = tracker.get_burn_rate(days=30)

        assert burn["total_tokens"] == 11000
        assert burn["total_queries"] == 2
        assert "projections" in burn

    def test_project_breakdown_single(self, config):
        """Test project breakdown for single project."""
        tracker = StatsTracker(config)
        tracker.record_context("query1", budget=8000, tokens_used=5000, duration_ms=100)

        projects = tracker.get_project_breakdown()

        assert len(projects) == 1
        assert projects[0]["tokens"] == 5000

    def test_get_summary(self, config):
        """Test summary aggregation."""
        tracker = StatsTracker(config)

        tracker.record_context("q1", budget=8000, tokens_used=4000, duration_ms=100)
        tracker.record_search("search1", results_count=5, duration_ms=50)
        tracker.record_remember("insight", source="manual")
        tracker.record_recall("recall_q", results_count=3, duration_ms=30)

        summary = tracker.get_summary()

        assert summary["context_queries"] == 1
        assert summary["search_queries"] == 1
        assert summary["remember_count"] == 1
        assert summary["recall_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
