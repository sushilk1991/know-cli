"""Configuration management for know."""

from dataclasses import dataclass, field, fields
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
        "python", "javascript", "typescript", "go", "rust", "swift"
    ])
    include: List[str] = field(default_factory=lambda: [
        "src/", "lib/", "app/", "packages/", "apps/", "cmd/", "internal/",
        "benchmark/", "benchmarks/"
    ])
    exclude: List[str] = field(default_factory=lambda: [
        "**/node_modules/**",
        "**/.git/**",
        "**/tests/**",
        "**/__pycache__/**",
        "**/vendor/**",
        "**/.venv/**",
        "**/.venv*/**",
        "**/venv/**",
        "**/venv*/**",
        "**/site-packages/**",
        "**/.cache/**",
        "**/.pytest_cache/**",
        "**/.mypy_cache/**",
        "**/.ruff_cache/**",
        "**/dist/**",
        "**/build/**",
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

        # Check for Swift
        if (root / "Package.swift").exists() or list(root.glob("**/*.swift")):
            languages.append("swift")
        
        return languages
    
    def detect_include_paths(self) -> None:
        """Auto-detect common source directories."""
        common_paths = [
            "src", "lib", "app", "packages", "apps", "cmd", "internal",
            "pkg", "api", "web", "server", "client", "frontend", "backend",
            "benchmark", "benchmarks"
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
            loaded = yaml.safe_load(f)

        # Empty files are a common intermediate state during editor writes.
        # A non-empty scalar/list is malformed, though, and silently treating
        # it as defaults can make know operate with a configuration the user
        # never requested.
        if loaded is None:
            data = {}
        elif not isinstance(loaded, dict):
            raise ValueError("Configuration root must be a YAML mapping")
        else:
            data = loaded

        def _section(name: str) -> dict:
            if name not in data:
                return {}
            value = data[name]
            if not isinstance(value, dict):
                raise ValueError(f"Configuration '{name}' must be a YAML mapping")
            return value

        def _string_list(name: str) -> Optional[List[str]]:
            if name not in data:
                return None
            value = data[name]
            if not isinstance(value, list) or any(
                not isinstance(item, str) for item in value
            ):
                raise ValueError(
                    f"Configuration '{name}' must be a YAML list of strings"
                )
            return value

        def _known(dataclass_type, values: dict, *, exclude=()) -> dict:
            allowed = {item.name for item in fields(dataclass_type)} - set(exclude)
            return {key: value for key, value in values.items() if key in allowed}
        
        config = cls()
        config.root = path.parent.parent.resolve()
        
        project = _section("project")
        if project:
            config.project = ProjectConfig(**_known(ProjectConfig, project))
        languages = _string_list("languages")
        include = _string_list("include")
        exclude = _string_list("exclude")
        if languages is not None:
            config.languages = languages
        if include is not None:
            config.include = include
        if exclude is not None:
            config.exclude = exclude
        ai = _section("ai")
        if ai:
            config.ai = AIConfig(**_known(AIConfig, ai, exclude={"generate"}))
            if "generate" in ai:
                if not isinstance(ai["generate"], dict):
                    raise ValueError("Configuration 'ai.generate' must be a YAML mapping")
                config.ai.generate = ai["generate"]
        output = _section("output")
        if output:
            config.output = OutputConfig(
                **_known(OutputConfig, output, exclude={"git", "watch"}),
            )
            git = output.get("git")
            if "git" in output and not isinstance(git, dict):
                raise ValueError("Configuration 'output.git' must be a YAML mapping")
            if isinstance(git, dict):
                config.output.git = OutputConfig.GitConfig(
                    **_known(OutputConfig.GitConfig, git)
                )
            watch = output.get("watch")
            if "watch" in output and not isinstance(watch, dict):
                raise ValueError("Configuration 'output.watch' must be a YAML mapping")
            if isinstance(watch, dict):
                config.output.watch = OutputConfig.WatchConfig(
                    **_known(OutputConfig.WatchConfig, watch)
                )
        diagrams = _section("diagrams")
        if diagrams:
            config.diagrams = DiagramConfig(**_known(DiagramConfig, diagrams))
        api = _section("api")
        if api:
            config.api = APIConfig(**_known(APIConfig, api))
        
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
