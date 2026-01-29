"""Comprehensive unit tests for know-cli."""

import ast
import tempfile
import shutil
from pathlib import Path
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from know.parsers import (
    ParserFactory, 
    PythonParser, 
    TypeScriptParser, 
    GoParser,
    BaseParser
)
from know.scanner import CodebaseScanner
from know.models import ModuleInfo, FunctionInfo, ClassInfo
from know.index import CodebaseIndex
from know.exceptions import ParseError


class TestPythonParser:
    """Tests for PythonParser."""
    
    @pytest.fixture
    def temp_dir(self):
        temp = Path(tempfile.mkdtemp())
        yield temp
        shutil.rmtree(temp)
    
    def test_parse_simple_function(self, temp_dir):
        """Test parsing a simple function."""
        parser = PythonParser()
        test_file = temp_dir / "test.py"
        test_file.write_text("""
def hello():
    \"\"\"Say hello.\"\"\"
    print("Hello")
""")
        
        module = parser.parse(test_file, temp_dir)
        
        assert module.name == "test"
        assert len(module.functions) == 1
        assert module.functions[0].name == "hello"
        assert module.functions[0].docstring == "Say hello."
    
    def test_parse_class_with_methods(self, temp_dir):
        """Test parsing a class with methods."""
        parser = PythonParser()
        test_file = temp_dir / "test.py"
        test_file.write_text("""
class MyClass:
    \"\"\"A test class.\"\"\"
    
    def method1(self):
        pass
    
    def method2(self, x: int) -> str:
        return str(x)
""")
        
        module = parser.parse(test_file, temp_dir)
        
        assert len(module.classes) == 1
        assert module.classes[0].name == "MyClass"
        assert module.classes[0].docstring == "A test class."
        assert len(module.classes[0].methods) == 2
    
    def test_parse_async_function(self, temp_dir):
        """Test parsing async functions."""
        parser = PythonParser()
        test_file = temp_dir / "test.py"
        test_file.write_text("""
async def fetch_data():
    pass
""")
        
        module = parser.parse(test_file, temp_dir)
        
        assert len(module.functions) == 1
        assert module.functions[0].is_async
    
    def test_parse_decorators(self, temp_dir):
        """Test parsing function decorators."""
        parser = PythonParser()
        test_file = temp_dir / "test.py"
        test_file.write_text("""
@staticmethod
@property
def my_func():
    pass
""")
        
        module = parser.parse(test_file, temp_dir)
        
        assert "staticmethod" in module.functions[0].decorators
        assert "property" in module.functions[0].decorators
    
    def test_parse_syntax_error(self, temp_dir):
        """Test handling syntax errors."""
        parser = PythonParser()
        test_file = temp_dir / "test.py"
        test_file.write_text("def broken(: pass")
        
        with pytest.raises(ParseError) as exc_info:
            parser.parse(test_file, temp_dir)
        
        assert "Syntax error" in str(exc_info.value)
    
    def test_parse_encoding_error(self, temp_dir):
        """Test handling encoding errors."""
        parser = PythonParser()
        test_file = temp_dir / "test.py"
        # Write invalid UTF-8
        test_file.write_bytes(b'\xff\xfe\x00\x00')
        
        with pytest.raises(ParseError) as exc_info:
            parser.parse(test_file, temp_dir)
        
        assert "Invalid encoding" in str(exc_info.value) or "Cannot read file" in str(exc_info.value)


class TestTypeScriptParser:
    """Tests for TypeScriptParser."""
    
    @pytest.fixture
    def temp_dir(self):
        temp = Path(tempfile.mkdtemp())
        yield temp
        shutil.rmtree(temp)
    
    def test_parse_function_regex(self, temp_dir):
        """Test parsing TypeScript functions with regex."""
        parser = TypeScriptParser(use_treesitter=False)
        test_file = temp_dir / "test.ts"
        test_file.write_text("""
function hello(): void {
    console.log("Hello");
}

export function world(): string {
    return "World";
}

async function asyncFunc(): Promise<void> {
    await something();
}
""")
        
        module = parser.parse(test_file, temp_dir)
        
        assert len(module.functions) == 3
        function_names = {f.name for f in module.functions}
        assert "hello" in function_names
        assert "world" in function_names
        assert "asyncFunc" in function_names
    
    def test_parse_class_regex(self, temp_dir):
        """Test parsing TypeScript classes with regex."""
        parser = TypeScriptParser(use_treesitter=False)
        test_file = temp_dir / "test.ts"
        test_file.write_text("""
class MyClass {
    private value: number;
    
    constructor() {
        this.value = 0;
    }
    
    getValue(): number {
        return this.value;
    }
}
""")
        
        module = parser.parse(test_file, temp_dir)
        
        assert len(module.classes) == 1
        assert module.classes[0].name == "MyClass"


class TestGoParser:
    """Tests for GoParser."""
    
    @pytest.fixture
    def temp_dir(self):
        temp = Path(tempfile.mkdtemp())
        yield temp
        shutil.rmtree(temp)
    
    def test_parse_function(self, temp_dir):
        """Test parsing Go functions."""
        parser = GoParser(use_treesitter=False)
        test_file = temp_dir / "test.go"
        test_file.write_text("""
package main

import "fmt"

func Hello() {
    fmt.Println("Hello")
}

func Add(a, b int) int {
    return a + b
}

func (r *Receiver) Method() {
    // method
}
""")
        
        module = parser.parse(test_file, temp_dir)
        
        assert len(module.functions) >= 2
        function_names = {f.name for f in module.functions}
        assert "Hello" in function_names
        assert "Add" in function_names
    
    def test_parse_struct(self, temp_dir):
        """Test parsing Go structs."""
        parser = GoParser(use_treesitter=False)
        test_file = temp_dir / "test.go"
        test_file.write_text("""
package main

type Person struct {
    Name string
    Age  int
}

type Empty struct{}
""")
        
        module = parser.parse(test_file, temp_dir)
        
        assert len(module.classes) == 2
        struct_names = {c.name for c in module.classes}
        assert "Person" in struct_names
        assert "Empty" in struct_names


class TestParserFactory:
    """Tests for ParserFactory."""
    
    def test_get_parser_python(self):
        """Test getting Python parser."""
        parser = ParserFactory.get_parser("python")
        assert isinstance(parser, PythonParser)
    
    def test_get_parser_typescript(self):
        """Test getting TypeScript parser."""
        parser = ParserFactory.get_parser("typescript")
        assert isinstance(parser, TypeScriptParser)
    
    def test_get_parser_go(self):
        """Test getting Go parser."""
        parser = ParserFactory.get_parser("go")
        assert isinstance(parser, GoParser)
    
    def test_get_parser_unknown(self):
        """Test getting unknown parser."""
        parser = ParserFactory.get_parser("unknown")
        assert parser is None
    
    def test_get_parser_for_file(self):
        """Test getting parser by file path."""
        assert isinstance(ParserFactory.get_parser_for_file(Path("test.py")), PythonParser)
        assert isinstance(ParserFactory.get_parser_for_file(Path("test.ts")), TypeScriptParser)
        assert isinstance(ParserFactory.get_parser_for_file(Path("test.go")), GoParser)
        assert ParserFactory.get_parser_for_file(Path("test.txt")) is None
    
    def test_parser_caching(self):
        """Test that parsers are cached."""
        parser1 = ParserFactory.get_parser("python")
        parser2 = ParserFactory.get_parser("python")
        assert parser1 is parser2


class TestCodebaseScanner:
    """Tests for CodebaseScanner."""
    
    @pytest.fixture
    def temp_project(self):
        """Create a temporary project with sample files."""
        temp = Path(tempfile.mkdtemp())
        
        # Create config
        from know.config import Config, OutputConfig
        (temp / ".know" / "cache").mkdir(parents=True, exist_ok=True)
        
        # Create Python files
        (temp / "src").mkdir()
        (temp / "src" / "main.py").write_text("""
def main():
    pass

class App:
    pass
""")
        (temp / "src" / "utils.py").write_text("""
def helper():
    pass
""")
        
        # Create TypeScript files
        (temp / "frontend").mkdir()
        (temp / "frontend" / "app.ts").write_text("""
function init() {}
class Component {}
""")
        
        yield temp
        shutil.rmtree(temp)
    
    @pytest.fixture
    def config(self, temp_project):
        from know.config import Config, OutputConfig
        return Config(
            root=temp_project,
            exclude=[".git", "__pycache__"],
            include=[],
            output=OutputConfig(directory="docs", watch=OutputConfig.WatchConfig())
        )
    
    def test_scan_discovers_files(self, config, temp_project):
        """Test that scanner discovers all file types."""
        scanner = CodebaseScanner(config)
        stats = scanner.scan()
        
        assert stats["files"] == 3
        assert len(scanner.modules) == 3
    
    def test_scan_counts_functions(self, config, temp_project):
        """Test that scanner counts functions correctly."""
        scanner = CodebaseScanner(config)
        stats = scanner.scan()
        
        assert stats["functions"] == 3  # main, helper, init
        assert stats["classes"] == 2    # App, Component
    
    def test_scan_with_caching(self, config, temp_project):
        """Test that scanning uses cache on second run."""
        scanner1 = CodebaseScanner(config)
        stats1 = scanner1.scan()
        
        scanner2 = CodebaseScanner(config)
        stats2 = scanner2.scan()
        
        assert stats2.get("cached_files", 0) == 3
        assert stats2.get("changed_files", 0) == 0
    
    def test_scan_respects_exclude(self, config, temp_project):
        """Test that scanner respects exclude patterns."""
        # Create a file in __pycache__
        (temp_project / "__pycache__").mkdir()
        (temp_project / "__pycache__" / "cached.py").write_text("def cached(): pass")
        
        scanner = CodebaseScanner(config)
        stats = scanner.scan()
        
        # Should not include __pycache__ files
        assert stats["files"] == 3
    
    def test_find_component(self, config, temp_project):
        """Test finding components by name."""
        scanner = CodebaseScanner(config)
        scanner.scan()
        
        results = scanner.find_component("main")
        assert len(results) == 1
        assert results[0]["type"] == "function"
        
        results = scanner.find_component("App")
        assert len(results) == 1
        assert results[0]["type"] == "class"
    
    def test_get_structure(self, config, temp_project):
        """Test getting structure."""
        scanner = CodebaseScanner(config)
        structure = scanner.get_structure()
        
        assert "modules" in structure
        assert "file_count" in structure
        assert structure["file_count"] == 3


class TestCodebaseIndex:
    """Tests for CodebaseIndex."""
    
    @pytest.fixture
    def temp_index(self):
        temp = Path(tempfile.mkdtemp())
        (temp / ".know" / "cache").mkdir(parents=True, exist_ok=True)
        
        from know.config import Config, OutputConfig
        config = Config(
            root=temp,
            exclude=[],
            include=[],
            output=OutputConfig(directory="docs", watch=OutputConfig.WatchConfig())
        )
        
        yield temp, config
        shutil.rmtree(temp)
    
    def test_cache_and_retrieve(self, temp_index):
        """Test caching and retrieving file metadata."""
        temp, config = temp_index
        
        index = CodebaseIndex(config)
        test_file = temp / "test.py"
        test_file.write_text("def hello(): pass")
        
        # Cache the file
        index.cache_file(test_file, "python", {"name": "test", "functions": []})
        
        # Retrieve metadata
        metadata = index.get_file_metadata(test_file)
        assert metadata is not None
        assert metadata["language"] == "python"
    
    def test_is_file_changed(self, temp_index):
        """Test detecting file changes."""
        temp, config = temp_index
        
        index = CodebaseIndex(config)
        test_file = temp / "test.py"
        test_file.write_text("def hello(): pass")
        
        # Initially changed (not cached)
        assert index.is_file_changed(test_file)
        
        # Cache it
        index.cache_file(test_file, "python", {"name": "test"})
        
        # Now unchanged
        assert not index.is_file_changed(test_file)
        
        # Modify file
        test_file.write_text("def hello(): return 42")
        
        # Should detect change
        assert index.is_file_changed(test_file)
    
    def test_get_stats(self, temp_index):
        """Test getting index statistics."""
        temp, config = temp_index
        
        index = CodebaseIndex(config)
        
        # Initially empty
        stats = index.get_stats()
        assert stats["cached_files"] == 0
        
        # Add some files
        for i in range(3):
            test_file = temp / f"test{i}.py"
            test_file.write_text(f"def func{i}(): pass")
            index.cache_file(test_file, "python", {"name": f"test{i}"})
        
        stats = index.get_stats()
        assert stats["cached_files"] == 3
    
    def test_clear(self, temp_index):
        """Test clearing the index."""
        temp, config = temp_index
        
        index = CodebaseIndex(config)
        test_file = temp / "test.py"
        test_file.write_text("def hello(): pass")
        index.cache_file(test_file, "python", {"name": "test"})
        
        assert index.get_stats()["cached_files"] == 1
        
        index.clear()
        
        assert index.get_stats()["cached_files"] == 0


class TestIntegration:
    """Integration tests for full workflow."""
    
    @pytest.fixture
    def temp_project(self):
        """Create a realistic temporary project."""
        temp = Path(tempfile.mkdtemp())
        (temp / ".know" / "cache").mkdir(parents=True, exist_ok=True)
        
        # Create a realistic project structure
        (temp / "src" / "api").mkdir(parents=True)
        (temp / "src" / "models").mkdir(parents=True)
        (temp / "tests").mkdir()
        
        # API routes
        (temp / "src" / "api" / "routes.py").write_text("""
from flask import Flask

app = Flask(__name__)

@app.route('/users')
def get_users():
    \"\"\"Get all users.\"\"\"
    return []

@app.route('/users/<id>')
def get_user(id):
    \"\"\"Get user by ID.\"\"\"
    return {}
""")
        
        # Models
        (temp / "src" / "models" / "user.py").write_text("""
class User:
    \"\"\"User model.\"\"\"
    
    def __init__(self, name: str):
        self.name = name
    
    def save(self):
        \"\"\"Save user to database.\"\"\"
        pass
""")
        
        # Main app
        (temp / "src" / "main.py").write_text("""
def create_app():
    \"\"\"Application factory.\"\"\"
    from .api.routes import app
    return app

if __name__ == '__main__':
    app = create_app()
    app.run()
""")
        
        from know.config import Config, OutputConfig
        config = Config(
            root=temp,
            exclude=[".git", "__pycache__", "tests/"],
            include=["src/"],
            output=OutputConfig(directory="docs", watch=OutputConfig.WatchConfig())
        )
        
        yield temp, config
        shutil.rmtree(temp)
    
    def test_full_scan_workflow(self, temp_project):
        """Test complete scan workflow."""
        temp, config = temp_project
        
        scanner = CodebaseScanner(config)
        stats = scanner.scan()
        
        assert stats["files"] == 3
        assert stats["functions"] >= 4
        assert stats["classes"] == 1
    
    def test_api_route_extraction(self, temp_project):
        """Test extracting API routes."""
        temp, config = temp_project
        
        scanner = CodebaseScanner(config)
        routes = scanner.extract_api_routes()
        
        assert len(routes) == 2
        route_paths = {r.path for r in routes}
        assert "/users" in route_paths


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
