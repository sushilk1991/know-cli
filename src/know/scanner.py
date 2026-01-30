"""Codebase scanner for AST analysis."""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Set, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import os
import logging

from pathspec import PathSpec

from know.exceptions import ParseError, ScanError
from know.logger import get_logger
from know.parsers import ParserFactory
from know.models import FunctionInfo, ClassInfo, ModuleInfo, APIRoute

if TYPE_CHECKING:
    from know.config import Config

# Tree-sitter availability check
try:
    from tree_sitter import Language, Parser
    from tree_sitter_languages import get_language
    _test_lang = get_language("python")
    _test_parser = Parser(_test_lang)
    TREESITTER_AVAILABLE = True
except Exception:
    TREESITTER_AVAILABLE = False


logger = get_logger()

# File extension to language mapping
LANGUAGE_EXTENSIONS = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "typescript",
    ".jsx": "typescript",
    ".go": "go",
}

# Maximum TypeScript files to scan (for performance)
MAX_TS_FILES = 500

# Timeout for parsing individual files (seconds)
FILE_PARSE_TIMEOUT = 30


def parse_file_task(args: Tuple[str, str, str, bool]) -> Optional[Tuple[str, str, Dict]]:
    """Standalone function for process pool parsing.
    
    Must be at module level to be picklable for ProcessPool.
    
    Args:
        args: Tuple of (file_path, root_path, language, use_treesitter)
    
    Returns:
        Tuple of (file_path, language, module_dict) or None on error
    """
    file_path_str, root_path_str, language, use_treesitter = args
    path = Path(file_path_str)
    root = Path(root_path_str)
    
    try:
        parser = ParserFactory.get_parser(language, use_treesitter)
        if not parser:
            return None
        
        module = parser.parse(path, root)
        if not module:
            return None
        
        # Convert to dict for serialization
        module_dict = {
            "path": str(module.path),
            "name": module.name,
            "docstring": module.docstring,
            "functions": [
                {
                    "name": f.name,
                    "line_number": f.line_number,
                    "docstring": f.docstring,
                    "signature": f.signature,
                    "is_async": f.is_async,
                    "is_method": f.is_method,
                    "decorators": f.decorators
                }
                for f in module.functions
            ],
            "classes": [
                {
                    "name": c.name,
                    "line_number": c.line_number,
                    "docstring": c.docstring,
                    "bases": c.bases,
                    "methods": [
                        {
                            "name": m.name,
                            "line_number": m.line_number,
                            "docstring": m.docstring,
                            "signature": m.signature,
                            "is_async": m.is_async,
                            "is_method": m.is_method,
                            "decorators": m.decorators
                        }
                        for m in c.methods
                    ]
                }
                for c in module.classes
            ],
            "imports": module.imports
        }
        
        return str(module.path), language, module_dict
    except ParseError as e:
        logger.debug(f"Parse error in {path}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Unexpected error parsing {path}: {e}")
        return None


class CodebaseScanner:
    """Scans codebase and extracts structure using strategy pattern for parsers."""

    def __init__(self, config: "Config"):
        self.config = config
        self.root = config.root
        self.modules: List[ModuleInfo] = []
        self._pathspec = self._build_pathspec()
        self._index: Optional["CodebaseIndex"] = None
        self._use_processes = True  # Enable ProcessPool by default
    
    def _get_index(self) -> "CodebaseIndex":
        """Lazy load the index."""
        if self._index is None:
            from know.index import CodebaseIndex
            self._index = CodebaseIndex(self.config)
        return self._index

    def _build_pathspec(self) -> PathSpec:
        """Build pathspec for filtering files."""
        patterns = self.config.exclude
        return PathSpec.from_lines("gitwildmatch", patterns)

    def _should_include(self, path: Path) -> bool:
        """Check if path should be included."""
        try:
            relative = path.relative_to(self.root)
        except ValueError:
            return False
        
        relative_str = str(relative)

        if self._pathspec.match_file(relative_str):
            return False

        if self.config.include:
            for pattern in self.config.include:
                pattern_clean = pattern.rstrip("/")
                if relative_str.startswith(pattern_clean + "/") or relative_str == pattern_clean:
                    return True
                for part in relative.parts:
                    if part == pattern_clean:
                        return True
            return False

        return True

    def _discover_files(self) -> Iterator[Tuple[Path, str]]:
        """Discover all source files in a single filesystem walk."""
        ts_files: List[Tuple[Path, str]] = []
        ts_count = 0
        
        logger.debug(f"Scanning filesystem: {self.root}")
        
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            
            if not self._should_include(path):
                continue
            
            suffix = path.suffix.lower()
            language = LANGUAGE_EXTENSIONS.get(suffix)
            
            if not language:
                continue
            
            if language == "typescript":
                if ts_count < MAX_TS_FILES:
                    ts_files.append((path, language))
                    ts_count += 1
                continue
            
            yield path, language
        
        if ts_files:
            ts_files.sort(key=lambda x: (
                "test" in str(x[0]).lower(),
                "spec" in str(x[0]).lower(),
                "__tests__" in str(x[0]).lower()
            ))
            for path, lang in ts_files[:MAX_TS_FILES]:
                yield path, lang

    def scan(self, max_workers: Optional[int] = None, progress_callback=None) -> Dict[str, int]:
        """Scan codebase and return statistics."""
        self.modules = []
        
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1))

        files_to_scan = list(self._discover_files())
        logger.info(f"Found {len(files_to_scan)} files to scan")
        
        return self._scan_incremental(files_to_scan, max_workers, progress_callback)

    def _scan_incremental(
        self, 
        files_to_scan: List[Tuple[Path, str]], 
        max_workers: int,
        progress_callback=None
    ) -> Dict[str, int]:
        """Scan with caching - only parse changed files."""
        from know.index import CodebaseIndex
        
        index = self._get_index()
        
        all_paths = [path for path, _ in files_to_scan]
        changed_paths, cached_modules = index.get_changed_files(all_paths)
        changed_set = set(str(p) for p in changed_paths)
        
        logger.info(f"Files to parse: {len(changed_paths)} new/changed, {len(cached_modules)} from cache")
        
        # Add cached modules
        cached_count = 0
        for cached in cached_modules:
            try:
                module = self._module_from_dict(cached)
                if module:
                    self.modules.append(module)
                    cached_count += 1
            except (KeyError, TypeError) as e:
                logger.debug(f"Failed to load cached module: {e}")
        
        files_to_parse = [(p, lang) for p, lang in files_to_scan if str(p) in changed_set]
        
        file_count = cached_count
        function_count = sum(len(m.functions) for m in self.modules)
        class_count = sum(len(m.classes) for m in self.modules)
        
        for cached in cached_modules:
            try:
                function_count += len(cached.get("functions", []))
                class_count += len(cached.get("classes", []))
                for cls in cached.get("classes", []):
                    function_count += len(cls.get("methods", []))
            except (KeyError, TypeError):
                pass

        # Prepare tasks for parallel processing
        tasks = [
            (str(p), str(self.root), lang, TREESITTER_AVAILABLE)
            for p, lang in files_to_parse
        ]
        
        parsed_count = 0
        failed_count = 0
        
        # Use ProcessPool for CPU-bound parsing (picklable function)
        if self._use_processes and len(tasks) > 5 and max_workers > 1:
            logger.debug(f"Using ProcessPool with {max_workers} workers for {len(tasks)} files")
            executor_class = ProcessPoolExecutor
        else:
            logger.debug(f"Using ThreadPool with {max_workers} workers")
            executor_class = ThreadPoolExecutor
        
        with executor_class(max_workers=max_workers) as executor:
            futures = {
                executor.submit(parse_file_task, task): task 
                for task in tasks
            }
            
            for i, future in enumerate(as_completed(futures)):
                if progress_callback:
                    progress_callback(i + 1, len(tasks))
                
                try:
                    result = future.result(timeout=FILE_PARSE_TIMEOUT)
                    if result:
                        rel_path, lang, module_dict = result
                        abs_path = self.root / rel_path
                        
                        # Cache the result
                        try:
                            index.cache_file(abs_path, lang, module_dict)
                        except Exception as e:
                            logger.debug(f"Failed to cache {rel_path}: {e}")
                        
                        # Convert back to ModuleInfo
                        module = self._module_from_dict(module_dict)
                        if module:
                            self.modules.append(module)
                            file_count += 1
                            function_count += len(module.functions)
                            class_count += len(module.classes)
                            for cls in module.classes:
                                function_count += len(cls.methods)
                            parsed_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    task = futures[future]
                    logger.debug(f"Error processing {task[0]}: {e}")
                    failed_count += 1
        
        if failed_count > 0:
            logger.debug(f"Failed to parse {failed_count} files")
        
        logger.info(f"Scan complete: {file_count} files, {function_count} functions, {class_count} classes")
        
        return {
            "files": file_count,
            "functions": function_count,
            "classes": class_count,
            "modules": len(self.modules),
            "changed_files": len(files_to_parse),
            "cached_files": cached_count,
            "failed_files": failed_count
        }

    def _module_from_dict(self, data: Dict[str, Any]) -> Optional[ModuleInfo]:
        """Convert dict back to ModuleInfo."""
        try:
            return ModuleInfo(
                path=Path(data["path"]),
                name=data["name"],
                docstring=data.get("docstring"),
                functions=[
                    FunctionInfo(
                        name=f["name"],
                        line_number=f["line_number"],
                        docstring=f.get("docstring"),
                        signature=f["signature"],
                        is_async=f.get("is_async", False),
                        is_method=f.get("is_method", False),
                        decorators=f.get("decorators", [])
                    )
                    for f in data.get("functions", [])
                ],
                classes=[
                    ClassInfo(
                        name=c["name"],
                        line_number=c["line_number"],
                        docstring=c.get("docstring"),
                        bases=c.get("bases", []),
                        methods=[
                            FunctionInfo(
                                name=m["name"],
                                line_number=m["line_number"],
                                docstring=m.get("docstring"),
                                signature=m["signature"],
                                is_async=m.get("is_async", False),
                                is_method=m.get("is_method", False),
                                decorators=m.get("decorators", [])
                            )
                            for m in c.get("methods", [])
                        ]
                    )
                    for c in data.get("classes", [])
                ],
                imports=data.get("imports", [])
            )
        except (KeyError, TypeError) as e:
            logger.debug(f"Failed to deserialize module: {e}")
            return None

    def get_structure(self) -> Dict[str, Any]:
        """Get full codebase structure."""
        if not self.modules:
            self.scan()

        total_functions = sum(len(m.functions) for m in self.modules)
        total_classes = sum(len(m.classes) for m in self.modules)

        key_files = []
        for module in self.modules:
            path = str(module.path)
            if any(x in path.lower() for x in ["main", "app", "core", "config", "api", "models"]):
                key_files.append(path)

        return {
            "modules": [
                {
                    "name": m.name,
                    "path": str(m.path),
                    "description": (m.docstring or "")[:200],
                    "function_count": len(m.functions),
                    "class_count": len(m.classes)
                }
                for m in self.modules
            ],
            "key_files": key_files[:20],
            "file_count": len(self.modules),
            "module_count": len(self.modules),
            "function_count": total_functions,
            "class_count": total_classes
        }

    def find_component(self, name: str) -> List[Any]:
        """Find a component by name."""
        if not self.modules:
            self.scan()

        results = []

        for module in self.modules:
            if name.lower() in module.name.lower():
                results.append({
                    "type": "module",
                    "name": module.name,
                    "path": str(module.path),
                    "content": module.docstring or ""
                })

            for func in module.functions:
                if name.lower() in func.name.lower():
                    results.append({
                        "type": "function",
                        "name": func.name,
                        "path": str(module.path),
                        "content": func.docstring or "",
                        "signature": func.signature
                    })

            for cls in module.classes:
                if name.lower() in cls.name.lower():
                    results.append({
                        "type": "class",
                        "name": cls.name,
                        "path": str(module.path),
                        "content": cls.docstring or ""
                    })

        return results

    def extract_api_routes(self) -> List[APIRoute]:
        """Extract API routes from Python codebase."""
        routes = []

        if not self.modules:
            self.scan()

        route_decorators = ["route", "get", "post", "put", "delete", "patch"]

        for module in self.modules:
            # Only process Python files
            if not str(module.path).endswith('.py'):
                continue
                
            try:
                content = (self.root / module.path).read_text()
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue

                    for decorator in node.decorator_list:
                        decorator_name = None

                        if isinstance(decorator, ast.Call):
                            if isinstance(decorator.func, ast.Name):
                                decorator_name = decorator.func.id.lower()
                            elif isinstance(decorator.func, ast.Attribute):
                                decorator_name = decorator.func.attr.lower()

                            if decorator_name in route_decorators:
                                path = "/"
                                if decorator.args:
                                    if isinstance(decorator.args[0], ast.Constant):
                                        path = decorator.args[0].value

                                method = decorator_name.upper()
                                if method == "ROUTE":
                                    method = "GET"

                                routes.append(APIRoute(
                                    method=method,
                                    path=path,
                                    handler=node.name,
                                    file_path=str(module.path),
                                    line_number=node.lineno,
                                    docstring=ast.get_docstring(node)
                                ))
            except Exception as e:
                logger.debug(f"Error extracting routes from {module.path}: {e}")

        return routes

    def scan_files(self, paths: List[Path]) -> Dict[str, int]:
        """Scan specific files incrementally (used by watch mode)."""
        index = self._get_index()
        
        all_changed = set(str(p) for p in paths)
        
        # Load all cached modules first
        all_cached = index.get_all_cached_modules()
        for cached in all_cached:
            try:
                if cached.get("path") in all_changed:
                    continue
                module = self._module_from_dict(cached)
                if module:
                    self.modules.append(module)
            except Exception as e:
                logger.debug(f"Failed to load cached module: {e}")
        
        file_count = len(self.modules)
        function_count = sum(len(m.functions) for m in self.modules)
        class_count = sum(len(m.classes) for m in self.modules)
        
        for path in paths:
            try:
                parser = ParserFactory.get_parser_for_file(path, TREESITTER_AVAILABLE)
                if not parser:
                    continue
                
                module = parser.parse(path, self.root)
                if not module:
                    continue
                
                try:
                    index.cache_file(path, parser.language, self._module_to_dict(module))
                except Exception as e:
                    logger.debug(f"Failed to cache {path}: {e}")
                
                # Replace or append
                existing_idx = None
                for i, m in enumerate(self.modules):
                    if str(m.path) == str(module.path):
                        existing_idx = i
                        break
                
                if existing_idx is not None:
                    self.modules[existing_idx] = module
                else:
                    self.modules.append(module)
                    file_count += 1
                    function_count += len(module.functions)
                    class_count += len(module.classes)
                    for cls in module.classes:
                        function_count += len(cls.methods)
                            
            except ParseError as e:
                logger.debug(f"Parse error: {e}")
            except Exception as e:
                logger.debug(f"Error processing {path}: {e}")
        
        return {
            "files": file_count,
            "functions": function_count,
            "classes": class_count,
            "modules": len(self.modules),
            "changed_files": len(paths)
        }

    def _module_to_dict(self, module: ModuleInfo) -> Dict[str, Any]:
        """Convert ModuleInfo to dict for caching."""
        return {
            "path": str(module.path),
            "name": module.name,
            "docstring": module.docstring,
            "functions": [
                {
                    "name": f.name,
                    "line_number": f.line_number,
                    "docstring": f.docstring,
                    "signature": f.signature,
                    "is_async": f.is_async,
                    "is_method": f.is_method,
                    "decorators": f.decorators
                }
                for f in module.functions
            ],
            "classes": [
                {
                    "name": c.name,
                    "line_number": c.line_number,
                    "docstring": c.docstring,
                    "bases": c.bases,
                    "methods": [
                        {
                            "name": m.name,
                            "line_number": m.line_number,
                            "docstring": m.docstring,
                            "signature": m.signature,
                            "is_async": m.is_async,
                            "is_method": m.is_method,
                            "decorators": m.decorators
                        }
                        for m in c.methods
                    ]
                }
                for c in module.classes
            ],
            "imports": module.imports
        }
