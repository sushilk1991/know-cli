"""Multi-language parsing via Tree-sitter with regex fallback.

Tree-sitter provides accurate AST parsing for 8+ languages. When tree-sitter
grammars are not installed, falls back to regex-based extraction that still
captures function/class signatures for most languages.

Supported languages: Python, TypeScript/JavaScript, Go, Rust, Java, Ruby, C/C++
"""

import ast
import re
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any

from know.exceptions import ParseError
from know.models import FunctionInfo, ClassInfo, ModuleInfo
from know.logger import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------
EXTENSION_TO_LANGUAGE: Dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
}

# Tree-sitter grammar packages for each language
_TS_GRAMMAR_PACKAGES: Dict[str, str] = {
    "python": "tree_sitter_python",
    "typescript": "tree_sitter_typescript",
    "tsx": "tree_sitter_typescript",
    "javascript": "tree_sitter_javascript",
    "jsx": "tree_sitter_javascript",
    "go": "tree_sitter_go",
    "rust": "tree_sitter_rust",
    "java": "tree_sitter_java",
    "ruby": "tree_sitter_ruby",
    "c": "tree_sitter_c",
    "cpp": "tree_sitter_c",  # C parser handles basic C++ too
}

# Node types per language for function/class extraction
_FUNCTION_NODE_TYPES: Dict[str, set] = {
    "python": {"function_definition"},
    "typescript": {"function_declaration", "method_definition", "arrow_function"},
    "tsx": {"function_declaration", "method_definition", "arrow_function"},
    "javascript": {"function_declaration", "method_definition", "arrow_function"},
    "jsx": {"function_declaration", "method_definition", "arrow_function"},
    "go": {"function_declaration", "method_declaration"},
    "rust": {"function_item"},
    "java": {"method_declaration", "constructor_declaration"},
    "ruby": {"method", "singleton_method"},
    "c": {"function_definition"},
    "cpp": {"function_definition"},
}

_CLASS_NODE_TYPES: Dict[str, set] = {
    "python": {"class_definition"},
    "typescript": {"class_declaration"},
    "tsx": {"class_declaration"},
    "javascript": {"class_declaration"},
    "jsx": {"class_declaration"},
    "go": set(),  # Go uses type_declaration + struct
    "rust": {"struct_item", "enum_item"},
    "java": {"class_declaration", "interface_declaration"},
    "ruby": {"class", "module"},
    "c": {"struct_specifier"},
    "cpp": {"struct_specifier", "class_specifier"},
}

_IMPORT_NODE_TYPES: Dict[str, set] = {
    "python": {"import_statement", "import_from_statement"},
    "typescript": {"import_statement"},
    "tsx": {"import_statement"},
    "javascript": {"import_statement"},
    "jsx": {"import_statement"},
    "go": {"import_declaration"},
    "rust": {"use_declaration"},
    "java": {"import_declaration"},
    "ruby": {"call"},  # require/require_relative
    "c": {"preproc_include"},
    "cpp": {"preproc_include"},
}


# ---------------------------------------------------------------------------
# Tree-sitter parser cache
# ---------------------------------------------------------------------------
_parser_cache: Dict[str, Any] = {}
_parser_cache_lock = threading.Lock()


def _get_ts_parser(language: str):
    """Get or create a tree-sitter parser for the given language.

    Returns (parser, Language) or (None, None) if unavailable.
    Thread-safe: uses a lock to protect the shared cache.
    """
    if language in _parser_cache:
        return _parser_cache[language]

    with _parser_cache_lock:
        # Double-check after acquiring lock
        if language in _parser_cache:
            return _parser_cache[language]

        try:
            from tree_sitter import Language, Parser
            import importlib

            pkg_name = _TS_GRAMMAR_PACKAGES.get(language)
            if not pkg_name:
                _parser_cache[language] = (None, None)
                return None, None

            mod = importlib.import_module(pkg_name)

            # Handle TypeScript which has separate language functions
            if language == "typescript":
                lang = Language(mod.language_typescript())
            elif language == "tsx":
                lang = Language(mod.language_tsx())
            else:
                lang = Language(mod.language())

            parser = Parser(lang)
            _parser_cache[language] = (parser, lang)
            return parser, lang

        except (ImportError, AttributeError, Exception) as e:
            logger.debug(f"Tree-sitter not available for {language}: {e}")
            _parser_cache[language] = (None, None)
            return None, None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class BaseParser(ABC):
    """Abstract base class for language parsers."""

    language: str = ""
    extensions: set = set()

    @abstractmethod
    def parse(self, path: Path, root: Path) -> ModuleInfo:
        """Parse a file and return module info."""
        pass

    def _read_file(self, path: Path, strict: bool = False) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="strict" if strict else "replace")
        except UnicodeDecodeError as e:
            raise ParseError(f"Invalid encoding: {e}", str(path))
        except Exception as e:
            raise ParseError(f"Cannot read file: {e}", str(path))


# ---------------------------------------------------------------------------
# Tree-sitter unified parser
# ---------------------------------------------------------------------------
class TreeSitterParser(BaseParser):
    """Multi-language parser using Tree-sitter."""

    def __init__(self, language: str):
        self.language = language
        self.extensions = {ext for ext, lang in EXTENSION_TO_LANGUAGE.items() if lang == language}

    def parse(self, path: Path, root: Path) -> ModuleInfo:
        content = self._read_file(path)
        content_bytes = content.encode("utf-8")

        parser, lang = _get_ts_parser(self.language)
        if parser is None:
            raise ParseError(f"Tree-sitter not available for {self.language}", str(path))

        tree = parser.parse(content_bytes)
        relative_path = path.relative_to(root)
        module_name = str(relative_path.with_suffix("")).replace("/", ".")

        module = ModuleInfo(
            path=relative_path,
            name=module_name,
            docstring=None,
            functions=[],
            classes=[],
            imports=[],
        )

        func_types = _FUNCTION_NODE_TYPES.get(self.language, set())
        class_types = _CLASS_NODE_TYPES.get(self.language, set())
        import_types = _IMPORT_NODE_TYPES.get(self.language, set())

        self._walk(tree.root_node, content_bytes, module, func_types, class_types, import_types)
        return module

    def _walk(self, node, content: bytes, module: ModuleInfo,
              func_types: set, class_types: set, import_types: set,
              is_method: bool = False):
        """Walk tree-sitter AST and extract functions, classes, imports."""
        ntype = node.type

        if ntype in func_types:
            func = self._extract_function(node, content, is_method)
            if func:
                module.functions.append(func)

        elif ntype in class_types:
            cls = self._extract_class(node, content)
            if cls:
                module.classes.append(cls)
            # Walk class body for methods
            for child in node.children:
                if child.type in ("block", "class_body", "body_statement",
                                  "declaration_list", "field_declaration_list"):
                    for grandchild in child.children:
                        self._walk(grandchild, content, module,
                                   func_types, class_types, import_types,
                                   is_method=True)
            return  # Don't recurse further into this node

        elif ntype in import_types:
            import_text = content[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
            # For Ruby, only capture require/require_relative calls
            if self.language == "ruby":
                if "require" in import_text:
                    module.imports.append(import_text.strip())
            else:
                module.imports.append(import_text.strip())

        # Rust impl block — extract methods from its body
        elif ntype == "impl_item" and self.language == "rust":
            for child in node.children:
                if child.type == "declaration_list":
                    for grandchild in child.children:
                        if grandchild.type in func_types:
                            func = self._extract_function(grandchild, content, is_method=True)
                            if func:
                                module.functions.append(func)
            return  # Don't recurse further

        # Go struct detection via type_declaration
        elif ntype == "type_declaration" and self.language == "go":
            cls = self._extract_go_type(node, content)
            if cls:
                module.classes.append(cls)

        for child in node.children:
            self._walk(child, content, module, func_types, class_types, import_types, is_method)

    def _extract_function(self, node, content: bytes, is_method: bool = False) -> Optional[FunctionInfo]:
        """Extract function info from tree-sitter node."""
        name = None
        for child in node.children:
            if child.type in ("identifier", "property_identifier", "name", "field_identifier"):
                name = content[child.start_byte:child.end_byte].decode("utf-8")
                break

        if not name:
            return None

        # Get the full signature line
        sig_line = content[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        # Take first line as signature
        sig = sig_line.split("\n")[0].rstrip("{").rstrip(":").strip()
        if len(sig) > 200:
            sig = sig[:200] + "..."

        # Extract docstring (next sibling or first child string)
        docstring = self._extract_docstring(node, content)

        return FunctionInfo(
            name=name,
            line_number=node.start_point[0] + 1,
            docstring=docstring,
            signature=sig,
            is_async="async" in content[node.start_byte:node.start_byte + 20].decode("utf-8", errors="replace"),
            is_method=is_method,
        )

    def _extract_class(self, node, content: bytes) -> Optional[ClassInfo]:
        """Extract class info from tree-sitter node."""
        name = None
        for child in node.children:
            if child.type in ("identifier", "type_identifier", "name", "constant"):
                name = content[child.start_byte:child.end_byte].decode("utf-8")
                break

        if not name:
            return None

        return ClassInfo(
            name=name,
            line_number=node.start_point[0] + 1,
            docstring=self._extract_docstring(node, content),
            methods=[],
            bases=[],
        )

    def _extract_go_type(self, node, content: bytes) -> Optional[ClassInfo]:
        """Extract Go type (struct/interface) from type_declaration."""
        for child in node.children:
            if child.type == "type_spec":
                name_node = None
                for grandchild in child.children:
                    if grandchild.type == "type_identifier":
                        name_node = grandchild
                        break
                if name_node:
                    name = content[name_node.start_byte:name_node.end_byte].decode("utf-8")
                    return ClassInfo(
                        name=name,
                        line_number=node.start_point[0] + 1,
                        docstring=None,
                        methods=[],
                        bases=[],
                    )
        return None

    def _extract_docstring(self, node, content: bytes) -> Optional[str]:
        """Try to extract a docstring from a node."""
        # Python: first child of body is expression_statement > string
        for child in node.children:
            if child.type in ("block", "body"):
                for grandchild in child.children:
                    if grandchild.type == "expression_statement":
                        for ggchild in grandchild.children:
                            if ggchild.type == "string":
                                doc = content[ggchild.start_byte:ggchild.end_byte].decode("utf-8")
                                return doc.strip("\"'").strip()
                break
        return None


# ---------------------------------------------------------------------------
# Python AST parser (built-in, no tree-sitter needed)
# ---------------------------------------------------------------------------
class PythonParser(BaseParser):
    """Parser for Python files using built-in ast module."""

    language = "python"
    extensions = {".py"}

    def parse(self, path: Path, root: Path) -> ModuleInfo:
        content = self._read_file(path, strict=True)

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            raise ParseError(f"Syntax error: {e}", str(path), e.lineno or 0)

        relative_path = path.relative_to(root)
        module_name = str(relative_path.with_suffix("")).replace("/", ".")

        module = ModuleInfo(
            path=relative_path,
            name=module_name,
            docstring=ast.get_docstring(tree),
            functions=[],
            classes=[],
            imports=[],
        )

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module.imports.append(node.module or "")

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                module.functions.append(self._parse_function(node))
            elif isinstance(node, ast.AsyncFunctionDef):
                module.functions.append(self._parse_function(node, is_async=True))
            elif isinstance(node, ast.ClassDef):
                module.classes.append(self._parse_class(node))

        return module

    def _parse_function(self, node, is_async=False, is_method=False) -> FunctionInfo:
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)

        signature = f"{node.name}({', '.join(args)})"

        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(decorator.attr)
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    decorators.append(f"{decorator.func.id}()")

        return FunctionInfo(
            name=node.name,
            line_number=node.lineno,
            docstring=ast.get_docstring(node),
            signature=signature,
            is_async=is_async or isinstance(node, ast.AsyncFunctionDef),
            is_method=is_method,
            decorators=decorators,
        )

    def _parse_class(self, node) -> ClassInfo:
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(self._parse_function(item, is_method=True))
            elif isinstance(item, ast.AsyncFunctionDef):
                methods.append(self._parse_function(item, is_async=True, is_method=True))

        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(ast.unparse(base))

        return ClassInfo(
            name=node.name,
            line_number=node.lineno,
            docstring=ast.get_docstring(node),
            methods=methods,
            bases=bases,
        )


# ---------------------------------------------------------------------------
# Regex fallback parsers
# ---------------------------------------------------------------------------
class RegexParser(BaseParser):
    """Regex-based parser for when tree-sitter is unavailable."""

    # Override these in subclasses
    _func_pattern: str = ""
    _class_pattern: str = ""
    _import_pattern: str = ""

    def parse(self, path: Path, root: Path) -> ModuleInfo:
        content = self._read_file(path)
        relative_path = path.relative_to(root)
        module_name = str(relative_path.with_suffix("")).replace("/", ".")

        module = ModuleInfo(
            path=relative_path,
            name=module_name,
            docstring=None,
            functions=[],
            classes=[],
            imports=[],
        )

        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            if self._func_pattern:
                match = re.match(self._func_pattern, stripped)
                if match:
                    name = match.group(1)
                    module.functions.append(FunctionInfo(
                        name=name, line_number=i, docstring=None,
                        signature=stripped.rstrip("{:").strip(),
                        is_async="async" in stripped,
                        is_method=False,
                    ))

            if self._class_pattern:
                match = re.match(self._class_pattern, stripped)
                if match:
                    name = match.group(1)
                    module.classes.append(ClassInfo(
                        name=name, line_number=i, docstring=None,
                        methods=[], bases=[],
                    ))

            if self._import_pattern:
                match = re.match(self._import_pattern, stripped)
                if match:
                    module.imports.append(stripped)

        return module


class TypeScriptRegexParser(RegexParser):
    language = "typescript"
    extensions = {".ts", ".tsx", ".js", ".jsx"}
    _func_pattern = r"(?:export\s+)?(?:async\s+)?function\s+(\w+)"
    _class_pattern = r"(?:export\s+)?class\s+(\w+)"
    _import_pattern = r"import\s+"


class GoRegexParser(RegexParser):
    language = "go"
    extensions = {".go"}
    _func_pattern = r"func\s+(?:\([^)]+\)\s+)?(\w+)\s*\("
    _class_pattern = r"type\s+(\w+)\s+struct"
    _import_pattern = r"import\s+"


class RustRegexParser(RegexParser):
    language = "rust"
    extensions = {".rs"}
    _func_pattern = r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)"
    _class_pattern = r"(?:pub\s+)?(?:struct|enum|trait)\s+(\w+)"
    _import_pattern = r"use\s+"


class JavaRegexParser(RegexParser):
    language = "java"
    extensions = {".java"}
    _func_pattern = r"(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)+(\w+)\s*\("
    _class_pattern = r"(?:public\s+)?(?:abstract\s+)?class\s+(\w+)"
    _import_pattern = r"import\s+"


class RubyRegexParser(RegexParser):
    language = "ruby"
    extensions = {".rb"}
    _func_pattern = r"def\s+(?:self\.)?(\w+)"
    _class_pattern = r"(?:class|module)\s+(\w+)"
    _import_pattern = r"require"


class CRegexParser(RegexParser):
    language = "c"
    extensions = {".c", ".h", ".cpp", ".cc", ".cxx", ".hpp"}
    _func_pattern = r"(?:\w+[\s*]+)+(\w+)\s*\([^)]*\)\s*\{"
    _class_pattern = r"(?:struct|class)\s+(\w+)"
    _import_pattern = r"#include"


# ---------------------------------------------------------------------------
# Backwards-compatible aliases (used by tests and external code)
# ---------------------------------------------------------------------------
class TypeScriptParser(TypeScriptRegexParser):
    """TypeScript parser — delegates to tree-sitter or regex."""

    def __init__(self, use_treesitter: bool = True):
        self._delegate = None
        if use_treesitter:
            parser, _ = _get_ts_parser("typescript")
            if parser is not None:
                self._delegate = TreeSitterParser("typescript")

    def parse(self, path: Path, root: Path) -> ModuleInfo:
        if self._delegate:
            return self._delegate.parse(path, root)
        return TypeScriptRegexParser.parse(self, path, root)


class GoParser(GoRegexParser):
    """Go parser — delegates to tree-sitter or regex."""

    def __init__(self, use_treesitter: bool = True):
        self._delegate = None
        if use_treesitter:
            parser, _ = _get_ts_parser("go")
            if parser is not None:
                self._delegate = TreeSitterParser("go")

    def parse(self, path: Path, root: Path) -> ModuleInfo:
        if self._delegate:
            return self._delegate.parse(path, root)
        return GoRegexParser.parse(self, path, root)


# ---------------------------------------------------------------------------
# Parser factory
# ---------------------------------------------------------------------------
_REGEX_PARSERS: Dict[str, type] = {
    "typescript": TypeScriptRegexParser,
    "tsx": TypeScriptRegexParser,
    "javascript": TypeScriptRegexParser,
    "jsx": TypeScriptRegexParser,
    "go": GoRegexParser,
    "rust": RustRegexParser,
    "java": JavaRegexParser,
    "ruby": RubyRegexParser,
    "c": CRegexParser,
    "cpp": CRegexParser,
}


class ParserFactory:
    """Factory for creating language-specific parsers.

    Prefers tree-sitter when available, falls back to regex.
    Python always uses the built-in ast module.
    """

    _parsers: Dict[str, BaseParser] = {}

    @classmethod
    def get_parser(cls, language: str, use_treesitter: bool = True) -> Optional[BaseParser]:
        """Get a parser for the specified language."""
        cache_key = f"{language}:{use_treesitter}"

        if cache_key not in cls._parsers:
            # Python always uses built-in ast
            if language == "python":
                cls._parsers[cache_key] = PythonParser()
            elif language in ("typescript", "tsx", "javascript", "jsx"):
                cls._parsers[cache_key] = TypeScriptParser(use_treesitter)
            elif language == "go":
                cls._parsers[cache_key] = GoParser(use_treesitter)
            elif use_treesitter:
                # Try tree-sitter first for other languages
                parser, _ = _get_ts_parser(language)
                if parser is not None:
                    cls._parsers[cache_key] = TreeSitterParser(language)
                elif language in _REGEX_PARSERS:
                    cls._parsers[cache_key] = _REGEX_PARSERS[language]()
                else:
                    return None
            elif language in _REGEX_PARSERS:
                cls._parsers[cache_key] = _REGEX_PARSERS[language]()
            else:
                return None

        return cls._parsers[cache_key]

    @classmethod
    def get_parser_for_file(cls, path: Path, use_treesitter: bool = True) -> Optional[BaseParser]:
        """Get a parser based on file extension."""
        suffix = path.suffix.lower()
        language = EXTENSION_TO_LANGUAGE.get(suffix)
        if language:
            return cls.get_parser(language, use_treesitter)
        return None

    @classmethod
    def supported_extensions(cls) -> set:
        """Return all supported file extensions."""
        return set(EXTENSION_TO_LANGUAGE.keys())

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the parser cache."""
        cls._parsers.clear()
        _parser_cache.clear()
