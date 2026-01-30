"""AI integration for intelligent code understanding."""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from rich.console import Console

if TYPE_CHECKING:
    from know.config import Config

console = Console()


@dataclass
class CodeComponent:
    name: str
    type: str  # function, class, module
    file_path: str
    content: str
    docstring: Optional[str] = None
    signature: Optional[str] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class AISummarizer:
    """AI-powered code summarization."""
    
    def __init__(self, config: "Config"):
        self.config = config
        self.provider = config.ai.provider
        self.model = config.ai.model
        self.api_key = os.getenv(config.ai.api_key_env)
        
        if not self.api_key and self.provider == "anthropic":
            console.print("""
[yellow]⚠ ANTHROPIC_API_KEY not set. AI features will be limited.[/yellow]

To enable AI-powered features:
1. Get an API key from https://console.anthropic.com/
2. Set environment variable: [bold]export ANTHROPIC_API_KEY="your-key"[/bold]
3. Add to your ~/.zshrc or ~/.bashrc to make it permanent

See troubleshooting: https://github.com/sushilk1991/know-cli#troubleshooting
""")
    
    def _call_claude(self, prompt: str, max_tokens: int = 4000) -> str:
        """Call Claude API."""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.api_key)
            
            message = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return message.content[0].text
        except ImportError:
            console.print("[red]✗ anthropic package not installed[/red]")
            return ""
        except Exception as e:
            console.print(f"[red]✗ AI call failed: {e}[/red]")
            return ""
    
    def explain_component(self, component: CodeComponent, detailed: bool = False) -> str:
        """Generate explanation for a code component."""
        if not self.api_key:
            return self._fallback_explain(component)
        
        detail_level = "detailed" if detailed else "concise"
        
        prompt = f"""Explain this {component.type} from a codebase:

Name: {component.name}
File: {component.file_path}

```
{component.content[:3000]}
```

Provide a {detail_level} explanation that covers:
1. What this {component.type} does
2. Key functionality
3. How it fits into the codebase
4. Important patterns or design decisions

Keep it clear and technical but accessible."""
        
        return self._call_claude(prompt)
    
    def _fallback_explain(self, component: CodeComponent) -> str:
        """Fallback explanation without AI."""
        lines = [
            f"## {component.name}",
            f"**Type:** {component.type}",
            f"**File:** `{component.file_path}`",
            "",
        ]
        
        if component.docstring:
            lines.append(component.docstring)
            lines.append("")
        
        if component.signature:
            lines.append(f"**Signature:** `{component.signature}`")
            lines.append("")
        
        lines.append("*Install anthropic package and set ANTHROPIC_API_KEY for AI-powered explanations*")
        
        return "\n".join(lines)
    
    def generate_onboarding_guide(
        self,
        structure: Dict[str, Any],
        audience: str
    ) -> str:
        """Generate onboarding guide for new team members."""
        if not self.api_key:
            return self._fallback_onboarding(structure, audience)
        
        # Summarize structure
        modules = structure.get("modules", [])
        key_files = structure.get("key_files", [])
        
        prompt = f"""Create an onboarding guide for a new {audience} joining this codebase.

Project Structure:
- Modules: {', '.join(modules[:10])}
- Key Files: {', '.join(key_files[:10])}

Generate a guide that includes:
1. Project overview
2. Architecture at a glance
3. Key directories and their purposes
4. Important patterns to know
5. Getting started steps
6. Common workflows

Make it welcoming and practical for someone new to the team."""
        
        return self._call_claude(prompt, max_tokens=6000)
    
    def _fallback_onboarding(self, structure: Dict[str, Any], audience: str) -> str:
        """Fallback onboarding without AI."""
        lines = [
            f"# Onboarding Guide for {audience}",
            "",
            "## Project Overview",
            f"This guide will help you get started with the codebase.",
            "",
            "## Project Structure",
            "",
            "### Modules",
        ]
        
        for module in structure.get("modules", [])[:10]:
            lines.append(f"- {module}")
        
        lines.extend([
            "",
            "### Key Files",
        ])
        
        for file in structure.get("key_files", [])[:10]:
            lines.append(f"- {file}")
        
        lines.extend([
            "",
            "*Install anthropic package and set ANTHROPIC_API_KEY for AI-powered guides*"
        ])
        
        return "\n".join(lines)
    
    def generate_llm_digest(
        self,
        structure: Dict[str, Any],
        compact: bool = False
    ) -> str:
        """Generate AI-optimized codebase summary (like GitIngest)."""
        if not self.api_key:
            return self._fallback_digest(structure, compact)
        
        # Build context
        modules_summary = "\n".join([
            f"- {m['name']}: {m.get('description', 'Module')[:100]}"
            for m in structure.get("modules", [])[:50]
        ])
        
        prompt = f"""Create an AI-optimized codebase digest for LLM consumption.

This should be a dense, information-rich summary that helps an AI understand:
1. The overall architecture and patterns
2. Key abstractions and data models
3. Important business logic locations
4. Testing patterns
5. Integration points

Codebase Structure:
{modules_summary}

Format the output as structured markdown with clear sections.
{"Keep it compact and dense." if compact else "Provide comprehensive detail."}"""
        
        return self._call_claude(prompt, max_tokens=8000)
    
    def _fallback_digest(self, structure: Dict[str, Any], compact: bool) -> str:
        """Fallback digest without AI."""
        lines = [
            "# Codebase Digest",
            "",
            "## Project Structure",
            "",
        ]
        
        for module in structure.get("modules", [])[:30]:
            lines.append(f"- {module}")
        
        lines.extend([
            "",
            "## Statistics",
            f"- Files: {structure.get('file_count', 'N/A')}",
            f"- Modules: {structure.get('module_count', 'N/A')}",
            f"- Functions: {structure.get('function_count', 'N/A')}",
            f"- Classes: {structure.get('class_count', 'N/A')}",
            "",
            "*Install anthropic package and set ANTHROPIC_API_KEY for AI-powered digests*"
        ])
        
        return "\n".join(lines)
    
    def generate_summary(self, structure: Dict[str, Any], compact: bool = False) -> str:
        """Generate human-readable summary."""
        return self.generate_llm_digest(structure, compact)
    
    def generate_readme_intro(self, structure: Dict[str, Any]) -> str:
        """Generate README introduction."""
        if not self.api_key:
            return f"""# {self.config.project.name}

{self.config.project.description}

*Generated by [know](https://github.com/vic/know-cli)*
"""
        
        prompt = f"""Write a compelling README introduction for this project:

Name: {self.config.project.name}
Description: {self.config.project.description}
Version: {self.config.project.version}

Files: {structure.get('file_count', 'N/A')}
Modules: {structure.get('module_count', 'N/A')}

Write in a professional, clear style suitable for GitHub.
Include:
1. One-sentence description
2. What problem it solves
3. Key features
4. Quick getting started hint

Keep it under 200 words."""
        
        return self._call_claude(prompt)
