"""Data models for know-cli."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


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
