"""Configuration management for know."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class ProjectConfig:
    name: str = ""
    description: str = ""
    version: str = "1.0.0"


@dataclass
class AIConfig:
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-5-20250929"
    api_key_env: str = "ANTHROPIC_API_KEY"
    generate: dict = field(default_factory=lambda: {
        "summaries": True,
        "architecture": True,
        "api_docs": True,
        "onboarding": True
    })


@dataclass
class OutputConfig:
    format: str = "markdown"
    directory: str = "docs"
    
    @dataclass
    class GitConfig:
        auto_commit: bool = False
        commit_message: str = "docs: update generated documentation"
    
    git: GitConfig = field(default_factory=GitConfig)
    
    @dataclass
    class WatchConfig:
        enabled: bool = True
        debounce_seconds: int = 5
    
    watch: WatchConfig = field(default_factory=WatchConfig)


@dataclass
class DiagramConfig:
    format: str = "mermaid"
    include_dependencies: bool = True
    max_depth: int = 3


@dataclass
class APIConfig:
    frameworks: List[str] = field(default_factory=list)
    include_schemas: bool = True
    include_examples: bool = True


@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    languages: List[str] = field(default_factory=lambda: [
        "python", "javascript", "typescript", "go", "rust"
    ])
    include: List[str] = field(default_factory=lambda: [
        "src/", "lib/", "app/", "packages/", "apps/", "cmd/", "internal/"
    ])
    exclude: List[str] = field(default_factory=lambda: [
        "**/node_modules/**",
        "**/.git/**",
        "**/tests/**",
        "**/__pycache__/**",
        "**/vendor/**",
        "**/.venv/**",
        "**/venv/**",
        "**/.know/**"
    ])
    ai: AIConfig = field(default_factory=AIConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    diagrams: DiagramConfig = field(default_factory=DiagramConfig)
    api: APIConfig = field(default_factory=APIConfig)
    root: Path = field(default_factory=lambda: Path(".").resolve())
    
    @classmethod
    def create_default(cls, root: Path) -> "Config":
        """Create default configuration for a project."""
        config = cls()
        config.root = root.resolve()
        
        # Try to detect project name from git or directory
        git_dir = root / ".git"
        if git_dir.exists():
            config.project.name = root.name
        else:
            config.project.name = root.name
        
        # Auto-detect languages
        detected_langs = cls._detect_languages(root)
        if detected_langs:
            config.languages = detected_langs
        
        # Auto-detect include paths
        config.detect_include_paths()
        
        return config
    
    @staticmethod
    def _detect_languages(root: Path) -> List[str]:
        """Detect programming languages used in the project."""
        languages = []
        
        # Check for Python
        if list(root.glob("*.py")) or list(root.glob("**/requirements.txt")):
            languages.append("python")
        
        # Check for JavaScript/TypeScript
        if (root / "package.json").exists():
            languages.append("javascript")
            if list(root.glob("**/*.ts")):
                languages.append("typescript")
        
        # Check for Go
        if (root / "go.mod").exists():
            languages.append("go")
        
        # Check for Rust
        if (root / "Cargo.toml").exists():
            languages.append("rust")
        
        return languages
    
    def detect_include_paths(self) -> None:
        """Auto-detect common source directories."""
        common_paths = [
            "src", "lib", "app", "packages", "apps", "cmd", "internal",
            "pkg", "api", "web", "server", "client", "frontend", "backend"
        ]
        
        detected = []
        for path in common_paths:
            if (self.root / path).exists() and (self.root / path).is_dir():
                detected.append(f"{path}/")
        
        if detected:
            self.include = detected
    
    def save(self, path: Path) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            "project": {
                "name": self.project.name,
                "description": self.project.description,
                "version": self.project.version
            },
            "languages": self.languages,
            "include": self.include,
            "exclude": self.exclude,
            "ai": {
                "provider": self.ai.provider,
                "model": self.ai.model,
                "api_key_env": self.ai.api_key_env,
                "generate": self.ai.generate
            },
            "output": {
                "format": self.output.format,
                "directory": self.output.directory,
                "git": {
                    "auto_commit": self.output.git.auto_commit,
                    "commit_message": self.output.git.commit_message
                },
                "watch": {
                    "enabled": self.output.watch.enabled,
                    "debounce_seconds": self.output.watch.debounce_seconds
                }
            },
            "diagrams": {
                "format": self.diagrams.format,
                "include_dependencies": self.diagrams.include_dependencies,
                "max_depth": self.diagrams.max_depth
            },
            "api": {
                "frameworks": self.api.frameworks,
                "include_schemas": self.api.include_schemas,
                "include_examples": self.api.include_examples
            }
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        
        config = cls()
        config.root = path.parent.parent.resolve()
        
        if "project" in data:
            config.project = ProjectConfig(**data["project"])
        if "languages" in data:
            config.languages = data["languages"]
        if "include" in data:
            config.include = data["include"]
        if "exclude" in data:
            config.exclude = data["exclude"]
        if "ai" in data:
            config.ai = AIConfig(**{k: v for k, v in data["ai"].items() if k != "generate"})
            if "generate" in data["ai"]:
                config.ai.generate = data["ai"]["generate"]
        if "output" in data:
            config.output = OutputConfig(
                format=data["output"].get("format", "markdown"),
                directory=data["output"].get("directory", "docs"),
            )
            if "git" in data["output"]:
                config.output.git = OutputConfig.GitConfig(**data["output"]["git"])
            if "watch" in data["output"]:
                config.output.watch = OutputConfig.WatchConfig(**data["output"]["watch"])
        if "diagrams" in data:
            config.diagrams = DiagramConfig(**data["diagrams"])
        if "api" in data:
            config.api = APIConfig(**data["api"])
        
        return config


def load_config(path: Optional[Path] = None) -> Config:
    """Load configuration from path or auto-detect."""
    if path:
        return Config.load(path)
    
    # Try to find config file
    current = Path(".").resolve()
    for _ in range(10):  # Max 10 levels up
        config_path = current / ".know" / "config.yaml"
        if config_path.exists():
            return Config.load(config_path)
        
        # Check parent
        parent = current.parent
        if parent == current:
            break
        current = parent
    
    # Return default config
    return Config.create_default(Path(".").resolve())
