"""Codebase scanner for AST analysis."""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from pathspec import PathSpec

if TYPE_CHECKING:
    from know.config import Config

# Tree-sitter imports (optional - falls back to regex if not available)
try:
    from tree_sitter import Language, Parser
    from tree_sitter_languages import get_language, get_parser
    TREESITTER_AVAILABLE = True
except ImportError:
    TREESITTER_AVAILABLE = False


@dataclass
class FunctionInfo:
    name: str
    line_number: int
    docstring: Optional[str]
    signature: str
    is_async: bool = False
    is_method: bool = False
    decorators: List[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    name: str
    line_number: int
    docstring: Optional[str]
    methods: List[FunctionInfo] = field(default_factory=list)
    bases: List[str] = field(default_factory=list)


@dataclass
class ModuleInfo:
    path: Path
    name: str
    docstring: Optional[str]
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)


@dataclass
class APIRoute:
    method: str
    path: str
    handler: str
    file_path: str
    line_number: int
    docstring: Optional[str] = None


class CodebaseScanner:
    """Scans codebase and extracts structure."""
    
    def __init__(self, config: "Config"):
        self.config = config
        self.root = config.root
        self.modules: List[ModuleInfo] = []
        self._pathspec = self._build_pathspec()
    
    def _build_pathspec(self) -> PathSpec:
        """Build pathspec for filtering files."""
        patterns = self.config.exclude
        return PathSpec.from_lines("gitwildmatch", patterns)
    
    def _should_include(self, path: Path) -> bool:
        """Check if path should be included."""
        relative = path.relative_to(self.root)
        relative_str = str(relative)
        
        # Check exclusion patterns
        if self._pathspec.match_file(relative_str):
            return False
        
        # Check inclusion patterns
        if self.config.include:
            for pattern in self.config.include:
                pattern_clean = pattern.rstrip("/")
                # Check if path starts with pattern or is inside it
                if relative_str.startswith(pattern_clean + "/") or relative_str == pattern_clean:
                    return True
                # Check if any parent directory matches
                for part in relative.parts:
                    if part == pattern_clean:
                        return True
            return False
        
        return True
    
    def scan(self, max_workers: Optional[int] = None) -> Dict[str, int]:
        """Scan codebase and return statistics.
        
        Uses parallel processing for better performance on large codebases.
        """
        self.modules = []
        
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) + 4)
        
        # Collect all files to scan
        files_to_scan: List[Tuple[Path, str]] = []
        
        # Python files
        for path in self.root.rglob("*.py"):
            if self._should_include(path):
                files_to_scan.append((path, "python"))
        
        # TypeScript/JavaScript files (limit to avoid overload)
        ts_files = []
        for ext in [".ts", ".tsx", ".js", ".jsx"]:
            for path in self.root.rglob(f"*{ext}"):
                if self._should_include(path):
                    ts_files.append((path, "typescript"))
        
        # Limit TS files for performance (prioritize non-test files)
        ts_files.sort(key=lambda x: ("test" in str(x[0]).lower(), "spec" in str(x[0]).lower()))
        files_to_scan.extend(ts_files[:500])  # Limit to 500 TS files
        
        # Go files
        for path in self.root.rglob("*.go"):
            if self._should_include(path):
                files_to_scan.append((path, "go"))
        
        # Process files in parallel
        file_count = 0
        function_count = 0
        class_count = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self._parse_file_worker, path, lang): (path, lang)
                for path, lang in files_to_scan
            }
            
            for future in as_completed(future_to_file):
                path, lang = future_to_file[future]
                try:
                    module = future.result(timeout=5)  # 5 second timeout per file
                    if module:
                        self.modules.append(module)
                        file_count += 1
                        function_count += len(module.functions)
                        class_count += len(module.classes)
                        for cls in module.classes:
                            function_count += len(cls.methods)
                except Exception:
                    continue
        
        return {
            "files": file_count,
            "functions": function_count,
            "classes": class_count,
            "modules": len(self.modules)
        }
    
    def _parse_file_worker(self, path: Path, language: str) -> Optional[ModuleInfo]:
        """Worker function for parallel parsing."""
        try:
            if language == "python":
                return self._parse_python_file(path)
            elif language == "typescript":
                if TREESITTER_AVAILABLE:
                    return self._parse_with_treesitter(path, language)
                else:
                    return self._parse_typescript_fast(path)
            elif language == "go":
                if TREESITTER_AVAILABLE:
                    return self._parse_with_treesitter(path, language)
                else:
                    return self._parse_go_file(path)
        except Exception:
            return None
        return None
    
    def _parse_with_treesitter(self, path: Path, language: str) -> Optional[ModuleInfo]:
        """Parse file using tree-sitter (fast & accurate)."""
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None
        
        relative_path = path.relative_to(self.root)
        module_name = str(relative_path.with_suffix("")).replace("/", ".")
        
        # Get parser for language
        try:
            parser = get_parser(language)
        except Exception:
            return None
        
        tree = parser.parse(bytes(content, "utf-8"))
        root_node = tree.root_node
        
        module = ModuleInfo(
            path=relative_path,
            name=module_name,
            docstring=None,
            functions=[],
            classes=[],
            imports=[]
        )
        
        # Walk the AST
        self._walk_treesitter_tree(root_node, content, module)
        
        return module
    
    def _walk_treesitter_tree(self, node, content: str, module: ModuleInfo) -> None:
        """Recursively walk tree-sitter AST."""
        node_type = node.type
        
        # Language-agnostic node types
        if node_type in ("function_declaration", "function_definition", "method_definition"):
            func = self._extract_function_from_treesitter(node, content)
            if func:
                module.functions.append(func)
        elif node_type in ("class_declaration", "class_definition", "struct_type"):
            cls = self._extract_class_from_treesitter(node, content)
            if cls:
                module.classes.append(cls)
        elif node_type in ("import_statement", "import_declaration"):
            import_text = content[node.start_byte:node.end_byte]
            module.imports.append(import_text.strip())
        
        # Recurse into children
        for child in node.children:
            self._walk_treesitter_tree(child, content, module)
    
    def _extract_function_from_treesitter(self, node, content: str) -> Optional[FunctionInfo]:
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
    
    def _extract_class_from_treesitter(self, node, content: str) -> Optional[ClassInfo]:
        """Extract class info from tree-sitter node."""
        name_node = None
        for child in node.children:
            if child.type == "type_identifier" or child.type == "identifier":
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
    
    def _parse_typescript_fast(self, path: Path) -> Optional[ModuleInfo]:
        """Fast TypeScript parsing (simplified regex approach)."""
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None
        
        relative_path = path.relative_to(self.root)
        module_name = str(relative_path.with_suffix("")).replace("/", ".")
        
        module = ModuleInfo(
            path=relative_path,
            name=module_name,
            docstring=None,
            functions=[],
            classes=[],
            imports=[]
        )
        
        # Fast line-by-line parsing
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            line = line.strip()
            
            # Function declarations (very fast check)
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
            
            # Class declarations
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
    
    def _parse_python_file(self, path: Path) -> Optional[ModuleInfo]:
        """Parse a Python file and extract information."""
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(content)
        except SyntaxError:
            return None
        
        relative_path = path.relative_to(self.root)
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
        # Build signature
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
        
        signature = f"{node.name}({', '.join(args)})"
        
        # Extract decorators
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
    
    def _parse_typescript_file(self, path: Path) -> Optional[ModuleInfo]:
        """Parse a TypeScript/JavaScript file and extract information."""
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None
        
        relative_path = path.relative_to(self.root)
        module_name = str(relative_path.with_suffix("")).replace("/", ".")
        
        # Extract JSDoc/docstring (comments starting with /**)
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
        
        # Extract imports (ES6 and CommonJS)
        import_patterns = [
            r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'import\s+[\'"]([^\'"]+)[\'"]',
            r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
        ]
        for pattern in import_patterns:
            for match in re.finditer(pattern, content):
                module.imports.append(match.group(1))
        
        # Extract functions (various patterns)
        func_patterns = [
            # function name() {}
            (r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)', False),
            # const name = () => {}
            (r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|\w+)\s*=>', False),
            # export const name = function() {}
            (r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*function', False),
        ]
        
        for pattern, is_async in func_patterns:
            for match in re.finditer(pattern, content):
                name = match.group(1)
                # Skip if already added
                if any(f.name == name for f in module.functions):
                    continue
                
                # Find line number
                line_num = content[:match.start()].count('\n') + 1
                
                func = FunctionInfo(
                    name=name,
                    line_number=line_num,
                    docstring=None,
                    signature=f"{name}()",
                    is_async='async' in match.group(0),
                    is_method=False
                )
                module.functions.append(func)
        
        # Extract classes
        class_pattern = r'(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?'
        for match in re.finditer(class_pattern, content):
            name = match.group(1)
            base_class = match.group(2)
            line_num = content[:match.start()].count('\n') + 1
            
            # Find class body and extract methods
            class_start = match.end()
            brace_count = 0
            class_end = class_start
            for i, char in enumerate(content[class_start:], start=class_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        class_end = i
                        break
            
            class_body = content[class_start:class_end]
            
            # Extract methods
            methods = []
            method_pattern = r'(?:async\s+)?(\w+)\s*\([^)]*\)\s*{'
            for m_match in re.finditer(method_pattern, class_body):
                m_name = m_match.group(1)
                if m_name in ['constructor', 'get', 'set']:
                    continue
                m_line = line_num + class_body[:m_match.start()].count('\n')
                methods.append(FunctionInfo(
                    name=m_name,
                    line_number=m_line,
                    docstring=None,
                    signature=f"{m_name}()",
                    is_async='async' in m_match.group(0),
                    is_method=True
                ))
            
            cls = ClassInfo(
                name=name,
                line_number=line_num,
                docstring=None,
                methods=methods,
                bases=[base_class] if base_class else []
            )
            module.classes.append(cls)
        
        return module
    
    def _parse_go_file(self, path: Path) -> Optional[ModuleInfo]:
        """Parse a Go file and extract information."""
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None
        
        relative_path = path.relative_to(self.root)
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
            
            func = FunctionInfo(
                name=name,
                line_number=line_num,
                docstring=None,
                signature=f"{name}()",
                is_async=False,
                is_method=False
            )
            module.functions.append(func)
        
        # Extract structs
        struct_pattern = r'type\s+(\w+)\s+struct'
        for match in re.finditer(struct_pattern, content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            
            cls = ClassInfo(
                name=name,
                line_number=line_num,
                docstring=None,
                methods=[],
                bases=[]
            )
            module.classes.append(cls)
        
        return module
    
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
    
    def get_structure(self) -> Dict[str, Any]:
        """Get full codebase structure."""
        if not self.modules:
            self.scan()
        
        total_functions = sum(len(m.functions) for m in self.modules)
        total_classes = sum(len(m.classes) for m in self.modules)
        
        # Find key files
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
            # Check module
            if name.lower() in module.name.lower():
                results.append({
                    "type": "module",
                    "name": module.name,
                    "path": str(module.path),
                    "content": module.docstring or ""
                })
            
            # Check functions
            for func in module.functions:
                if name.lower() in func.name.lower():
                    results.append({
                        "type": "function",
                        "name": func.name,
                        "path": str(module.path),
                        "content": func.docstring or "",
                        "signature": func.signature
                    })
            
            # Check classes
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
        """Extract API routes from codebase."""
        routes = []
        
        if not self.modules:
            self.scan()
        
        # Common decorators for API routes
        route_decorators = ["route", "get", "post", "put", "delete", "patch"]
        
        for module in self.modules:
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
                                # Extract path from decorator args
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
            except Exception:
                continue
        
        return routes
