"""Test suite for know CLI."""

import os
import tempfile
from pathlib import Path
import pytest

from know.ai import AISummarizer, TokenOptimizer, AIResponseCache, CodeComponent
from know.config import Config, ProjectConfig, AIConfig, OutputConfig
from know.scanner import CodebaseScanner


class TestTokenOptimizer:
    """Test token optimization utilities."""
    
    def test_compress_code_removes_comments(self):
        """Test that code compression removes comments."""
        optimizer = TokenOptimizer()
        code = '''
def hello():
    # This is a comment
    print("hello")  # inline comment
    return 42
'''
        compressed = optimizer.compress_code(code)
        assert "# This is a comment" not in compressed
        assert "# inline comment" not in compressed
        assert 'print("hello")' in compressed
    
    def test_compress_code_removes_docstrings(self):
        """Test that docstrings are replaced."""
        optimizer = TokenOptimizer()
        code = '''
def hello():
    """This is a docstring.
    It has multiple lines.
    """
    pass
'''
        compressed = optimizer.compress_code(code)
        assert 'This is a docstring' not in compressed
        assert '"""..."""' in compressed or "'''...'''" in compressed
    
    def test_compress_code_truncates_long_content(self):
        """Test that long content is truncated."""
        optimizer = TokenOptimizer()
        code = "x = 1\n" * 1000  # Very long code
        compressed = optimizer.compress_code(code, max_chars=500)
        assert len(compressed) <= 550  # Some buffer for truncation message
        assert "[truncated]" in compressed
    
    def test_extract_key_signatures(self):
        """Test signature extraction."""
        optimizer = TokenOptimizer()
        code = '''
def hello(name: str) -> str:
    """Docstring"""
    return f"Hello {name}"

class MyClass:
    def method(self, x: int) -> int:
        return x * 2
'''
        signatures = optimizer.extract_key_signatures(code)
        assert "def hello(name: str) -> str:" in signatures
        assert "class MyClass:" in signatures
        assert 'return f"Hello {name}"' not in signatures  # Body should be excluded


class TestAIResponseCache:
    """Test AI response caching."""
    
    def test_cache_save_and_retrieve(self):
        """Test saving and retrieving from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AIResponseCache(cache_dir=Path(tmpdir))
            
            # Save a response
            cache.set("test content", "explain", "claude-sonnet", "test response", 100)
            
            # Retrieve it
            cached = cache.get("test content", "explain", "claude-sonnet")
            assert cached == "test response"
    
    def test_cache_miss_returns_none(self):
        """Test that cache miss returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AIResponseCache(cache_dir=Path(tmpdir))
            
            # Try to get non-existent entry
            cached = cache.get("nonexistent", "explain", "claude-sonnet")
            assert cached is None
    
    def test_cache_different_tasks(self):
        """Test that different task types are cached separately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AIResponseCache(cache_dir=Path(tmpdir))
            
            cache.set("content", "explain", "claude-sonnet", "explain response", 100)
            cache.set("content", "summarize", "claude-sonnet", "summarize response", 50)
            
            assert cache.get("content", "explain", "claude-sonnet") == "explain response"
            assert cache.get("content", "summarize", "claude-sonnet") == "summarize response"


class TestAISummarizer:
    """Test AI summarization with mocked API."""
    
    @pytest.fixture
    def config(self):
        """Create a test config."""
        return Config(
            project=ProjectConfig(name="Test Project", description="A test project"),
            ai=AIConfig(provider="anthropic", model="claude-haiku-4-5-20251022"),
            output=OutputConfig(directory="docs")
        )
    
    def test_fallback_when_no_api_key(self, config, monkeypatch):
        """Test that fallback is used when no API key."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        
        summarizer = AISummarizer(config)
        component = CodeComponent(
            name="test_func",
            type="function",
            file_path="test.py",
            content="def test(): pass",
            docstring="Test function"
        )
        
        result = summarizer.explain_component(component)
        assert "Test function" in result  # Should use docstring from fallback
        assert "test_func" in result
    
    def test_component_content_truncation(self, config, monkeypatch):
        """Test that long component content is truncated."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        
        summarizer = AISummarizer(config)
        long_content = "x = 1\n" * 1000
        component = CodeComponent(
            name="long_func",
            type="function",
            file_path="test.py",
            content=long_content
        )
        
        # Should compress/truncate the content
        compressed = summarizer.optimizer.compress_code(long_content, max_chars=1200)
        assert len(compressed) < len(long_content)
    
    def test_cost_calculation(self, config, monkeypatch):
        """Test cost calculation for API calls."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        
        summarizer = AISummarizer(config)
        
        # Test pricing structure
        assert summarizer.MODEL_HAIKU in summarizer.PRICING
        assert summarizer.MODEL_SONNET in summarizer.PRICING
        
        # Haiku should be cheaper
        assert summarizer.PRICING[summarizer.MODEL_HAIKU]["input"] < summarizer.PRICING[summarizer.MODEL_SONNET]["input"]


class TestCodebaseScanner:
    """Test codebase scanning functionality."""
    
    @pytest.fixture
    def temp_project(self):
        """Create a temporary project structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create Python files
            (root / "src").mkdir()
            (root / "src" / "main.py").write_text('''
def main():
    """Main function."""
    print("Hello")

class MyClass:
    def method(self):
        pass
''')
            
            # Create TypeScript files
            (root / "web").mkdir()
            (root / "web" / "app.ts").write_text('''
function greet(name: string): string {
    return `Hello ${name}`;
}
''')
            
            yield root
    
    def test_scan_detects_python_files(self, temp_project):
        """Test that scanner detects Python files."""
        config = Config.create_default(temp_project)
        scanner = CodebaseScanner(config)
        
        structure = scanner.get_structure()
        
        assert structure["file_count"] > 0
        # Should detect the Python file
        py_files = [f for f in structure.get("files", []) if f.endswith(".py")]
        assert len(py_files) > 0
    
    def test_scan_counts_functions_and_classes(self, temp_project):
        """Test that scanner counts functions and classes."""
        config = Config.create_default(temp_project)
        scanner = CodebaseScanner(config)
        
        structure = scanner.get_structure()
        
        # Should detect main() and MyClass.method()
        assert structure.get("function_count", 0) >= 1
        # Should detect MyClass
        assert structure.get("class_count", 0) >= 1


class TestConfig:
    """Test configuration management."""
    
    def test_default_config_creation(self):
        """Test creating default config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config.create_default(Path(tmpdir))
            
            assert config.project.name == Path(tmpdir).name
            assert "python" in config.languages
            assert config.output.directory == "docs"
    
    def test_config_save_and_load(self):
        """Test saving and loading config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / ".know" / "config.yaml"
            config_path.parent.mkdir(parents=True)
            
            # Create and save config
            config = Config.create_default(Path(tmpdir))
            config.project.name = "Test Project"
            config.save(config_path)
            
            # Load it back
            loaded = Config.load(config_path)
            assert loaded.project.name == "Test Project"
            assert loaded.root == Path(tmpdir).resolve()


class TestIntegration:
    """Integration tests."""
    
    def test_full_workflow_without_ai(self, tmp_path, monkeypatch):
        """Test full workflow with AI disabled (no API key)."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        
        # Create a simple project
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text('''
def hello():
    """Say hello."""
    print("Hello")
''')
        
        # Initialize config
        config = Config.create_default(tmp_path)
        config.output.directory = "docs"
        
        # Scan codebase
        scanner = CodebaseScanner(config)
        structure = scanner.get_structure()
        
        assert structure["file_count"] == 1
        assert structure["function_count"] == 1
        
        # Generate docs (should use fallback without AI)
        from know.generator import DocGenerator
        generator = DocGenerator(config)
        
        doc_path = generator.generate_system_doc(structure)
        assert doc_path.exists()
        content = doc_path.read_text()
        assert "Test Project" in content or tmp_path.name in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
