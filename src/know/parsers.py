"""Parser strategy pattern for language-specific parsing."""

import ast
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List
import logging

from know.exceptions import ParseError
from know.models import FunctionInfo, ClassInfo, ModuleInfo

logger = logging.getLogger("know")


class BaseParser(ABC):
    """Abstract base class for language parsers."""
    
    language: str = ""
    extensions: set = set()
    
    @abstractmethod
    def parse(self, path: Path, root: Path) -> ModuleInfo:
        """Parse a file and return module info."""
        pass
    
    def _read_file(self, path: Path) -> str:
        """Read file content with proper encoding handling."""
        try:
            return path.read_text(encoding="utf-8", errors="strict")
        except UnicodeDecodeError as e:
            raise ParseError(f"Invalid encoding: {e}", str(path))
        except Exception as e:
            raise ParseError(f"Cannot read file: {e}", str(path))


class PythonParser(BaseParser):
    """Parser for Python files."""
    
    language = "python"
    extensions = {".py"}
    
    def parse(self, path: Path, root: Path) -> ModuleInfo:
        content = self._read_file(path)
        
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
            imports=[]
        )
        
        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module.imports.append(node.module or "")
        
        # Extract top-level functions and classes
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                module.functions.append(self._parse_function(node))
            elif isinstance(node, ast.AsyncFunctionDef):
                module.functions.append(self._parse_function(node, is_async=True))
            elif isinstance(node, ast.ClassDef):
                module.classes.append(self._parse_class(node))
        
        return module
    
    def _parse_function(
        self,
        node: ast.FunctionDef,
        is_async: bool = False,
        is_method: bool = False
    ) -> FunctionInfo:
        """Parse a function definition."""
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
            decorators=decorators
        )
    
    def _parse_class(self, node: ast.ClassDef) -> ClassInfo:
        """Parse a class definition."""
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
                bases.append(f"{ast.unparse(base)}")
        
        return ClassInfo(
            name=node.name,
            line_number=node.lineno,
            docstring=ast.get_docstring(node),
            methods=methods,
            bases=bases
        )


class TypeScriptParser(BaseParser):
    """Parser for TypeScript/JavaScript files."""

    language = "typescript"
    extensions = {".ts", ".tsx", ".js", ".jsx"}

    def __init__(self, use_treesitter: bool = False):
        self.use_treesitter = use_treesitter
        if use_treesitter:
            try:
                from tree_sitter import Parser
                from tree_sitter_languages import get_language
                self._lang = get_language("typescript")
                self._parser = Parser(self._lang)
            except Exception as e:
                logger.debug(f"Tree-sitter not available for TypeScript: {e}")
                self.use_treesitter = False

    def parse(self, path: Path, root: Path) -> ModuleInfo:
        # Always use regex for TSX/JSX files (tree-sitter struggles with JSX syntax)
        if path.suffix in ('.tsx', '.jsx'):
            return self._parse_with_regex(path, root)

        if self.use_treesitter:
            try:
                return self._parse_with_treesitter(path, root)
            except Exception:
                # Fall back to regex on tree-sitter failure
                return self._parse_with_regex(path, root)
        return self._parse_with_regex(path, root)
    
    def _parse_with_treesitter(self, path: Path, root: Path) -> ModuleInfo:
        """Parse using tree-sitter."""
        content = self._read_file(path)
        
        tree = self._parser.parse(bytes(content, "utf-8"))
        root_node = tree.root_node
        
        relative_path = path.relative_to(root)
        module_name = str(relative_path.with_suffix("")).replace("/", ".")
        
        module = ModuleInfo(
            path=relative_path,
            name=module_name,
            docstring=None,
            functions=[],
            classes=[],
            imports=[]
        )
        
        self._walk_tree(root_node, content, module)
        return module
    
    def _walk_tree(self, node, content: str, module: ModuleInfo) -> None:
        """Recursively walk tree-sitter AST."""
        node_type = node.type
        
        if node_type in ("function_declaration", "function_definition", "method_definition"):
            func = self._extract_function(node, content)
            if func:
                module.functions.append(func)
        elif node_type in ("class_declaration", "class_definition"):
            cls = self._extract_class(node, content)
            if cls:
                module.classes.append(cls)
        elif node_type in ("import_statement", "import_declaration"):
            import_text = content[node.start_byte:node.end_byte]
            module.imports.append(import_text.strip())
        
        for child in node.children:
            self._walk_tree(child, content, module)
    
    def _extract_function(self, node, content: str) -> Optional[FunctionInfo]:
        """Extract function info from tree-sitter node."""
        name_node = None
        for child in node.children:
            if child.type in ("identifier", "property_identifier"):
                name_node = child
                break
        
        if not name_node:
            return None
        
        name = content[name_node.start_byte:name_node.end_byte]
        line_num = content[:node.start_byte].count('\n') + 1
        
        return FunctionInfo(
            name=name,
            line_number=line_num,
            docstring=None,
            signature=f"{name}()",
            is_async=False,
            is_method=False
        )
    
    def _extract_class(self, node, content: str) -> Optional[ClassInfo]:
        """Extract class info from tree-sitter node."""
        name_node = None
        for child in node.children:
            if child.type in ("type_identifier", "identifier"):
                name_node = child
                break
        
        if not name_node:
            return None
        
        name = content[name_node.start_byte:name_node.end_byte]
        line_num = content[:node.start_byte].count('\n') + 1
        
        return ClassInfo(
            name=name,
            line_number=line_num,
            docstring=None,
            methods=[],
            bases=[]
        )
    
    def _parse_with_regex(self, path: Path, root: Path) -> ModuleInfo:
        """Fast regex-based parsing fallback."""
        content = self._read_file(path)
        
        relative_path = path.relative_to(root)
        module_name = str(relative_path.with_suffix("")).replace("/", ".")
        
        # Extract JSDoc/docstring
        docstring = None
        doc_match = re.search(r'/\*\*\s*\n([^*]|\*(?!/))*\*/', content)
        if doc_match:
            docstring = doc_match.group(0)
        
        module = ModuleInfo(
            path=relative_path,
            name=module_name,
            docstring=docstring,
            functions=[],
            classes=[],
            imports=[]
        )
        
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            line = line.strip()
            
            if line.startswith('function ') or line.startswith('export function ') or line.startswith('async function '):
                match = re.search(r'function\s+(\w+)', line)
                if match:
                    module.functions.append(FunctionInfo(
                        name=match.group(1),
                        line_number=i,
                        docstring=None,
                        signature=f"{match.group(1)}()",
                        is_async='async' in line,
                        is_method=False
                    ))
            elif line.startswith('class ') or line.startswith('export class '):
                match = re.search(r'class\s+(\w+)', line)
                if match:
                    module.classes.append(ClassInfo(
                        name=match.group(1),
                        line_number=i,
                        docstring=None,
                        methods=[],
                        bases=[]
                    ))
        
        return module


class GoParser(BaseParser):
    """Parser for Go files."""
    
    language = "go"
    extensions = {".go"}
    
    def __init__(self, use_treesitter: bool = False):
        self.use_treesitter = use_treesitter
        if use_treesitter:
            try:
                from tree_sitter import Parser
                from tree_sitter_languages import get_language
                self._lang = get_language("go")
                self._parser = Parser(self._lang)
            except Exception as e:
                logger.debug(f"Tree-sitter not available for Go: {e}")
                self.use_treesitter = False
    
    def parse(self, path: Path, root: Path) -> ModuleInfo:
        if self.use_treesitter:
            return self._parse_with_treesitter(path, root)
        return self._parse_with_regex(path, root)
    
    def _parse_with_treesitter(self, path: Path, root: Path) -> ModuleInfo:
        """Parse using tree-sitter."""
        content = self._read_file(path)
        
        tree = self._parser.parse(bytes(content, "utf-8"))
        root_node = tree.root_node
        
        relative_path = path.relative_to(root)
        module_name = str(relative_path.with_suffix("")).replace("/", ".")
        
        module = ModuleInfo(
            path=relative_path,
            name=module_name,
            docstring=None,
            functions=[],
            classes=[],
            imports=[]
        )
        
        self._walk_tree(root_node, content, module)
        return module
    
    def _walk_tree(self, node, content: str, module: ModuleInfo) -> None:
        """Recursively walk tree-sitter AST."""
        node_type = node.type
        
        if node_type == "function_declaration":
            func = self._extract_function(node, content)
            if func:
                module.functions.append(func)
        elif node_type == "type_declaration":
            cls = self._extract_struct(node, content)
            if cls:
                module.classes.append(cls)
        elif node_type == "import_declaration":
            import_text = content[node.start_byte:node.end_byte]
            module.imports.append(import_text.strip())
        
        for child in node.children:
            self._walk_tree(child, content, module)
    
    def _extract_function(self, node, content: str) -> Optional[FunctionInfo]:
        """Extract function info from tree-sitter node."""
        name_node = None
        for child in node.children:
            if child.type == "identifier":
                name_node = child
                break
        
        if not name_node:
            return None
        
        name = content[name_node.start_byte:name_node.end_byte]
        line_num = content[:node.start_byte].count('\n') + 1
        
        return FunctionInfo(
            name=name,
            line_number=line_num,
            docstring=None,
            signature=f"{name}()",
            is_async=False,
            is_method=False
        )
    
    def _extract_struct(self, node, content: str) -> Optional[ClassInfo]:
        """Extract struct info from tree-sitter node."""
        name_node = None
        for child in node.children:
            if child.type == "type_identifier":
                name_node = child
                break
        
        if not name_node:
            return None
        
        name = content[name_node.start_byte:name_node.end_byte]
        line_num = content[:node.start_byte].count('\n') + 1
        
        return ClassInfo(
            name=name,
            line_number=line_num,
            docstring=None,
            methods=[],
            bases=[]
        )
    
    def _parse_with_regex(self, path: Path, root: Path) -> ModuleInfo:
        """Fast regex-based parsing fallback."""
        content = self._read_file(path)
        
        relative_path = path.relative_to(root)
        module_name = str(relative_path.with_suffix("")).replace("/", ".")
        
        # Extract package doc
        docstring = None
        doc_match = re.search(r'^//\s*(.+?)\n(?://\s*(.+?)\n)*package\s+\w+', content, re.MULTILINE)
        if doc_match:
            lines = doc_match.group(0).split('\n')
            doc_lines = [line.lstrip('/').strip() for line in lines if line.startswith('//')]
            docstring = ' '.join(doc_lines)
        
        module = ModuleInfo(
            path=relative_path,
            name=module_name,
            docstring=docstring,
            functions=[],
            classes=[],
            imports=[]
        )
        
        # Extract imports
        import_block = re.search(r'import\s*\((.*?)\)', content, re.DOTALL)
        if import_block:
            for line in import_block.group(1).split('\n'):
                match = re.search(r'"([^"]+)"', line)
                if match:
                    module.imports.append(match.group(1))
        else:
            single_import = re.search(r'import\s+"([^"]+)"', content)
            if single_import:
                module.imports.append(single_import.group(1))
        
        # Extract functions
        func_pattern = r'func\s+(?:\([^)]+\)\s+)?(\w+)\s*\([^)]*\)'
        for match in re.finditer(func_pattern, content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            
            module.functions.append(FunctionInfo(
                name=name,
                line_number=line_num,
                docstring=None,
                signature=f"{name}()",
                is_async=False,
                is_method=False
            ))
        
        # Extract structs
        struct_pattern = r'type\s+(\w+)\s+struct'
        for match in re.finditer(struct_pattern, content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            
            module.classes.append(ClassInfo(
                name=name,
                line_number=line_num,
                docstring=None,
                methods=[],
                bases=[]
            ))
        
        return module


class ParserFactory:
    """Factory for creating language-specific parsers."""
    
    _parsers: dict = {}
    
    @classmethod
    def get_parser(cls, language: str, use_treesitter: bool = False) -> Optional[BaseParser]:
        """Get a parser for the specified language."""
        cache_key = f"{language}:{use_treesitter}"
        
        if cache_key not in cls._parsers:
            if language == "python":
                cls._parsers[cache_key] = PythonParser()
            elif language == "typescript":
                cls._parsers[cache_key] = TypeScriptParser(use_treesitter)
            elif language == "go":
                cls._parsers[cache_key] = GoParser(use_treesitter)
            else:
                return None
        
        return cls._parsers[cache_key]
    
    @classmethod
    def get_parser_for_file(cls, path: Path, use_treesitter: bool = False) -> Optional[BaseParser]:
        """Get a parser based on file extension."""
        suffix = path.suffix.lower()
        
        if suffix in PythonParser.extensions:
            return cls.get_parser("python", use_treesitter)
        elif suffix in TypeScriptParser.extensions:
            return cls.get_parser("typescript", use_treesitter)
        elif suffix in GoParser.extensions:
            return cls.get_parser("go", use_treesitter)
        
        return None
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the parser cache."""
        cls._parsers.clear()
