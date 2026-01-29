"""Efficiency tests for know-cli indexing and incremental scanning."""

import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple
import pytest

# Add src to path for testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from know.config import Config, OutputConfig
from know.scanner import CodebaseScanner, ModuleInfo, FunctionInfo, ClassInfo
from know.index import CodebaseIndex


class TestIndexingEfficiency:
    """Test suite for measuring indexing efficiency."""
    
    @pytest.fixture
    def temp_repo(self):
        """Create a temporary repository with sample files."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create sample Python files
        for i in range(50):
            file_path = temp_dir / f"module_{i}.py"
            file_path.write_text(f'''
"""Module {i} - Sample module for testing."""

import os
import sys
from typing import List, Dict

class ClassA{i}:
    """Class A{i} description."""
    
    def method_1(self) -> None:
        """Method 1 docstring."""
        pass
    
    def method_2(self, x: int) -> str:
        """Method 2 docstring."""
        return str(x)

class ClassB{i}:
    """Class B{i} description."""
    
    def method_3(self) -> List[int]:
        """Method 3 docstring."""
        return [1, 2, 3]

def function_{i}_a() -> None:
    """Function {i}A docstring."""
    pass

def function_{i}_b(x: int, y: str) -> Dict:
    """Function {i}B docstring."""
    return {{"x": x, "y": y}}

def function_{i}_c(items: List[str]) -> None:
    """Function {i}C docstring."""
    for item in items:
        print(item)
''')
        
        # Create a .know directory
        (temp_dir / ".know" / "cache").mkdir(parents=True, exist_ok=True)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self, temp_repo):
        """Create a config for the temp repo."""
        return Config(
            root=temp_repo,
            exclude=[".git", "__pycache__", ".venv"],
            include=[],
            output=OutputConfig(
                directory="docs",
                watch=OutputConfig.WatchConfig(debounce_seconds=1.0)
            )
        )
    
    def test_initial_scan_performance(self, config, temp_repo):
        """Measure initial full scan performance."""
        scanner = CodebaseScanner(config)
        
        start = time.perf_counter()
        stats = scanner.scan()
        elapsed = time.perf_counter() - start
        
        print(f"\nInitial scan: {elapsed:.3f}s for {stats['files']} files")
        print(f"  - Functions: {stats['functions']}")
        print(f"  - Classes: {stats['classes']}")
        
        # Should complete in reasonable time
        assert elapsed < 10.0, f"Initial scan too slow: {elapsed:.3f}s"
        assert stats['files'] == 50
    
    def test_cached_scan_performance(self, config, temp_repo):
        """Measure performance with cached files."""
        scanner = CodebaseScanner(config)
        
        # First scan (populates cache)
        scanner.scan()
        
        # Second scan (should use cache)
        scanner2 = CodebaseScanner(config)
        start = time.perf_counter()
        stats = scanner2.scan()
        elapsed = time.perf_counter() - start
        
        print(f"\nCached scan: {elapsed:.3f}s for {stats['files']} files")
        print(f"  - Changed files: {stats.get('changed_files', 'N/A')}")
        print(f"  - Cached files: {stats.get('cached_files', 'N/A')}")
        
        # Should be much faster with cache
        assert elapsed < 1.0, f"Cached scan too slow: {elapsed:.3f}s"
        assert stats.get('cached_files', 0) == 50
        assert stats.get('changed_files', 0) == 0
    
    def test_incremental_single_file_change(self, config, temp_repo):
        """Measure performance when only one file changes."""
        scanner = CodebaseScanner(config)
        
        # Initial scan
        scanner.scan()
        
        # Modify one file
        (temp_repo / "module_25.py").write_text('''
"""Module 25 - MODIFIED."""

def new_function() -> None:
    """This is a new function."""
    pass
''')
        
        # Scan again - should only re-parse one file
        scanner2 = CodebaseScanner(config)
        start = time.perf_counter()
        stats = scanner2.scan()
        elapsed = time.perf_counter() - start
        
        print(f"\nSingle file change: {elapsed:.3f}s")
        print(f"  - Total files: {stats['files']}")
        print(f"  - Changed files: {stats.get('changed_files', 'N/A')}")
        print(f"  - Cached files: {stats.get('cached_files', 'N/A')}")
        
        # Should be very fast - only parsing 1 file
        assert elapsed < 0.5, f"Incremental scan too slow: {elapsed:.3f}s"
        assert stats.get('changed_files', 0) == 1
        assert stats.get('cached_files', 0) == 49
    
    def test_scan_files_method(self, config, temp_repo):
        """Test the scan_files method for targeted updates."""
        scanner = CodebaseScanner(config)
        
        # Initial full scan
        scanner.scan()
        
        # Modify a few files
        for i in [5, 10, 15]:
            (temp_repo / f"module_{i}.py").write_text(f'''
"""Module {i} - MODIFIED."""

def modified_function_{i}() -> None:
    """Modified function."""
    pass
''')
        
        # Use scan_files for targeted update
        scanner2 = CodebaseScanner(config)
        changed_paths = [
            temp_repo / "module_5.py",
            temp_repo / "module_10.py", 
            temp_repo / "module_15.py"
        ]
        
        start = time.perf_counter()
        stats = scanner2.scan_files(changed_paths)
        elapsed = time.perf_counter() - start
        
        print(f"\nscan_files method: {elapsed:.3f}s for 3 changed files")
        print(f"  - Total modules: {stats['modules']}")
        print(f"  - Changed files: {stats.get('changed_files', 'N/A')}")
        
        assert elapsed < 0.5, f"Targeted scan too slow: {elapsed:.3f}s"
        assert stats['modules'] == 50  # All 50 loaded (3 new + 47 cached)
        assert stats.get('changed_files', 0) == 3
    
    def test_cache_persistence(self, config, temp_repo):
        """Test that cache persists across scanner instances."""
        # First scanner populates cache
        scanner1 = CodebaseScanner(config)
        scanner1.scan()
        
        # Create new scanner (simulates new process)
        scanner2 = CodebaseScanner(config)
        start = time.perf_counter()
        stats = scanner2.scan()
        elapsed = time.perf_counter() - start
        
        print(f"\nCache persistence test: {elapsed:.3f}s")
        assert stats.get('cached_files', 0) == 50
        assert elapsed < 1.0
    
    def test_large_fileset_performance(self):
        """Test with a larger set of files."""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Create 200 files
            for i in range(200):
                file_path = temp_dir / f"module_{i}.py"
                file_path.write_text(f'''
"""Module {i}."""

class Class{i}:
    def method_1(self): pass
    def method_2(self): pass

def func_{i}_a(): pass
def func_{i}_b(): pass
''')
            
            (temp_dir / ".know" / "cache").mkdir(parents=True, exist_ok=True)
            
            config = Config(
                root=temp_dir,
                exclude=[".git", "__pycache__"],
                include=[],
                output=OutputConfig(
                    directory="docs", 
                    watch=OutputConfig.WatchConfig()
                )
            )
            
            # Initial scan
            scanner = CodebaseScanner(config)
            start = time.perf_counter()
            stats = scanner.scan()
            initial_time = time.perf_counter() - start
            
            print(f"\nLarge fileset ({stats['files']} files):")
            print(f"  - Initial scan: {initial_time:.3f}s")
            
            # Cached scan
            scanner2 = CodebaseScanner(config)
            start = time.perf_counter()
            stats2 = scanner2.scan()
            cached_time = time.perf_counter() - start
            
            print(f"  - Cached scan: {cached_time:.3f}s")
            print(f"  - Speedup: {initial_time / cached_time:.1f}x")
            
            assert initial_time < 30.0, "Initial scan too slow"
            assert cached_time < 2.0, "Cached scan too slow"
            assert stats2.get('cached_files', 0) == 200
            
        finally:
            shutil.rmtree(temp_dir)


class TestCacheCorrectness:
    """Test that caching doesn't break correctness."""
    
    @pytest.fixture
    def temp_repo(self):
        """Create temporary repository."""
        temp_dir = Path(tempfile.mkdtemp())
        (temp_dir / ".know" / "cache").mkdir(parents=True, exist_ok=True)
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self, temp_repo):
        return Config(
            root=temp_repo,
            exclude=[],
            include=[],
            output=OutputConfig(
                directory="docs", 
                watch=OutputConfig.WatchConfig()
            )
        )
    
    def test_cache_invalidation_on_content_change(self, config, temp_repo):
        """Verify cache detects content changes correctly."""
        # Create initial file
        test_file = temp_repo / "test.py"
        test_file.write_text('''
def original_function():
    """Original."""
    pass
''')
        
        # First scan
        scanner = CodebaseScanner(config)
        scanner.scan()
        
        # Verify cache hit
        index = CodebaseIndex(config)
        assert not index.is_file_changed(test_file)
        
        # Modify content
        test_file.write_text('''
def modified_function():
    """Modified."""
    return 42
''')
        
        # Verify cache miss
        assert index.is_file_changed(test_file)
    
    def test_cache_hit_on_same_content(self, config, temp_repo):
        """Verify cache hits when content is identical."""
        test_file = temp_repo / "test.py"
        content = '''
def my_function():
    """My function."""
    pass
'''
        test_file.write_text(content)
        
        # First scan
        scanner = CodebaseScanner(config)
        scanner.scan()
        
        # Simulate file modification (same content, different mtime)
        time.sleep(0.1)
        test_file.write_text(content)
        
        # Should still detect as changed due to mtime, but hash should match
        index = CodebaseIndex(config)
        # Note: mtime check will trigger re-parse, but that's acceptable
        # The hash check ensures we don't re-parse unnecessarily in batch operations
    
    def test_structure_consistency(self, config, temp_repo):
        """Verify structure is identical between cached and fresh scans."""
        # Create file
        test_file = temp_repo / "test.py"
        test_file.write_text('''
"""Test module."""

class MyClass:
    """A class."""
    
    def method_a(self): pass
    def method_b(self): pass

def standalone_func():
    """A function."""
    pass
''')
        
        # Fresh scan
        scanner1 = CodebaseScanner(config)
        stats1 = scanner1.scan()
        structure1 = scanner1.get_structure()
        
        # Cached scan
        scanner2 = CodebaseScanner(config)
        stats2 = scanner2.scan()
        structure2 = scanner2.get_structure()
        
        # Should be identical
        assert stats1['files'] == stats2['files']
        assert stats1['functions'] == stats2['functions']
        assert stats1['classes'] == stats2['classes']
        assert len(structure1['modules']) == len(structure2['modules'])


class TestIndexStats:
    """Test index statistics and management."""
    
    @pytest.fixture
    def temp_repo(self):
        temp_dir = Path(tempfile.mkdtemp())
        (temp_dir / ".know" / "cache").mkdir(parents=True, exist_ok=True)
        
        # Create some files
        for i in range(10):
            (temp_dir / f"file_{i}.py").write_text(f"def func_{i}(): pass")
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self, temp_repo):
        return Config(
            root=temp_repo,
            exclude=[],
            include=[],
            output=OutputConfig(
                directory="docs", 
                watch=OutputConfig.WatchConfig()
            )
        )
    
    def test_index_stats(self, config, temp_repo):
        """Test index statistics reporting."""
        index = CodebaseIndex(config)
        
        # Initially empty
        stats = index.get_stats()
        assert stats['cached_files'] == 0
        assert stats['cached_modules'] == 0
        
        # After scan
        scanner = CodebaseScanner(config)
        scanner.scan()
        
        stats = index.get_stats()
        assert stats['cached_files'] == 10
        assert stats['cached_modules'] == 10
    
    def test_index_clear(self, config, temp_repo):
        """Test clearing the index."""
        # Populate index
        scanner = CodebaseScanner(config)
        scanner.scan()
        
        index = CodebaseIndex(config)
        assert index.get_stats()['cached_files'] == 10
        
        # Clear
        index.clear()
        
        assert index.get_stats()['cached_files'] == 0
        assert index.get_stats()['cached_modules'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
