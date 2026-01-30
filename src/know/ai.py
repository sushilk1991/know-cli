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
    
    def _call_claude(self, prompt: str, max_tokens: int = 2000) -> str:
        """Call Claude API with cost optimization."""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.api_key)
            
            # Count input tokens for cost tracking
            input_tokens = client.count_tokens(prompt)
            
            message = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Log usage for transparency
            output_tokens = message.usage.output_tokens
            total_tokens = input_tokens + output_tokens
            
            # Sonnet 4.5 costs: $3/million input, $15/million output
            input_cost = (input_tokens / 1_000_000) * 3.0
            output_cost = (output_tokens / 1_000_000) * 15.0
            total_cost = input_cost + output_cost
            
            if total_cost > 0.01:  # Only show for expensive calls
                console.print(f"[dim]💰 API call: {total_tokens:,} tokens (~${total_cost:.4f})[/dim]")
            
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
        
        # Truncate content to reduce token usage
        max_content = 2000 if detailed else 1000
        content = component.content[:max_content]
        if len(component.content) > max_content:
            content += "\n... [truncated]"
        
        detail_level = "detailed" if detailed else "concise"
        max_tokens = 1500 if detailed else 800
        
        prompt = f"""Explain this {component.type}:

Name: {component.name}
File: {component.file_path}

```
{content}
```

Provide a {detail_level} explanation covering:
1. Purpose and functionality
2. Key inputs/outputs
3. How it fits in the codebase
4. Important patterns

Be concise."""
        
        return self._call_claude(prompt, max_tokens=max_tokens)
    
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
        
        # Limit modules to reduce token usage
        modules = structure.get("modules", [])[:15]  # Limit to 15
        key_files = structure.get("key_files", [])[:10]  # Limit to 10
        
        prompt = f"""Create a concise onboarding guide for {audience}.

Modules: {', '.join(modules[:15])}
Key Files: {', '.join(key_files[:10])}

Include:
1. Project overview (2-3 sentences)
2. Key directories
3. Important patterns
4. Getting started steps
5. Common commands

Keep it under 1000 words. Be practical and direct."""
        
        return self._call_claude(prompt, max_tokens=2000)
    
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
        
        # Build context - limit to reduce tokens
        module_limit = 30 if compact else 50
        modules_summary = "\n".join([
            f"- {m['name']}: {m.get('description', 'Module')[:80]}"
            for m in structure.get("modules", [])[:module_limit]
        ])
        
        max_tokens = 4000 if compact else 6000
        
        prompt = f"""Create a codebase digest for AI consumption.

Structure:
{modules_summary}

Cover:
1. Architecture overview
2. Key abstractions/models
3. Business logic locations
4. Testing patterns
5. Integration points

Format as markdown.
{"Be extremely concise." if compact else "Be comprehensive but focused."}"""
        
        return self._call_claude(prompt, max_tokens=max_tokens)
    
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

*Generated by [know](https://github.com/sushilk1991/know-cli)*
"""
        
        prompt = f"""Write a README intro for {self.config.project.name}.

Description: {self.config.project.description}
Files: {structure.get('file_count', 'N/A')}

Include:
1. One-line pitch
2. Problem it solves (1-2 sentences)
3. 3-4 key features
4. Quick install hint

Keep under 150 words. Professional tone."""
        
        return self._call_claude(prompt, max_tokens=800)
