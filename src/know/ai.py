"""AI integration for intelligent code understanding with advanced token optimization."""

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
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


class TokenOptimizer:
    """Token optimization utilities."""
    
    @staticmethod
    def compress_code(content: str, max_chars: int = 2000) -> str:
        """Compress code by removing comments and excess whitespace."""
        import re
        
        # Remove single-line comments (but not URLs)
        content = re.sub(r'(?<!:)//.*$', '', content, flags=re.MULTILINE)
        
        # Remove multi-line comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Remove docstrings (Python-style)
        content = re.sub(r'""".*?"""', '"""..."""', content, flags=re.DOTALL)
        content = re.sub(r"'''.*?'''", "'''...'''", content, flags=re.DOTALL)
        
        # Remove extra blank lines
        content = re.sub(r'\n\s*\n+', '\n\n', content)
        
        # Remove trailing whitespace
        content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
        
        # Truncate if still too long
        if len(content) > max_chars:
            content = content[:max_chars] + "\n// ... [truncated]"
        
        return content.strip()
    
    @staticmethod
    def extract_key_signatures(content: str, max_items: int = 10) -> str:
        """Extract only function/class signatures without bodies."""
        import re
        
        # Match function signatures
        func_pattern = r'((?:async\s+)?(?:def|function)\s+\w+\s*\([^)]*\)(?:\s*->\s*\w+)?:?)'
        class_pattern = r'((?:export\s+)?(?:class|interface)\s+\w+(?:\s*(?:extends|implements)\s+\w+)?)'
        
        funcs = re.findall(func_pattern, content)[:max_items]
        classes = re.findall(class_pattern, content)[:max_items]
        
        result = []
        if classes:
            result.append("Classes/Interfaces:")
            result.extend(f"  {c}" for c in classes)
        if funcs:
            result.append("Functions:")
            result.extend(f"  {f}" for f in funcs)
        
        return '\n'.join(result) if result else content[:500]


class AIResponseCache:
    """Cache AI responses to avoid duplicate API calls."""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "know-cli"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "ai_cache.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite cache."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    content_hash TEXT PRIMARY KEY,
                    task_type TEXT,
                    model TEXT,
                    response TEXT,
                    tokens_used INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Clean old entries (older than 30 days)
            conn.execute(
                "DELETE FROM cache WHERE created_at < datetime('now', '-30 days')"
            )
    
    def get(self, content: str, task_type: str, model: str) -> Optional[str]:
        """Get cached response if available."""
        content_hash = hashlib.sha256(f"{content}:{task_type}:{model}".encode()).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT response FROM cache WHERE content_hash = ?",
                (content_hash,)
            )
            row = cursor.fetchone()
            if row:
                console.print("[dim]♻️ Using cached response[/dim]")
                return row[0]
        return None
    
    def set(self, content: str, task_type: str, model: str, response: str, tokens_used: int):
        """Cache a response."""
        content_hash = hashlib.sha256(f"{content}:{task_type}:{model}".encode()).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO cache 
                   (content_hash, task_type, model, response, tokens_used)
                   VALUES (?, ?, ?, ?, ?)""",
                (content_hash, task_type, model, response, tokens_used)
            )


class AISummarizer:
    """AI-powered code summarization with aggressive token optimization."""
    
    # Model names - using latest available models
    MODEL_SONNET = "claude-sonnet-4-5-20250929"
    MODEL_HAIKU = "claude-haiku-4-5-20251001"
    
    # Pricing (per million tokens)
    PRICING = {
        MODEL_SONNET: {"input": 3.0, "output": 15.0},
        MODEL_HAIKU: {"input": 1.0, "output": 5.0},
    }
    
    def __init__(self, config: "Config"):
        self.config = config
        self.provider = config.ai.provider
        self.model = config.ai.model
        self.api_key = os.getenv(config.ai.api_key_env, "").strip()
        self.cache = AIResponseCache()
        self.optimizer = TokenOptimizer()
        
        # Only show warning if key is genuinely not set
        if not self.api_key and self.provider == "anthropic":
            console.print("[yellow]⚠ ANTHROPIC_API_KEY not set. AI features will be limited.[/yellow]")
    
    def _call_claude(
        self, 
        prompt: str, 
        max_tokens: int = 2000,
        model: Optional[str] = None,
        cache_key: Optional[str] = None,
        task_type: str = "general"
    ) -> str:
        """Call Claude API with caching and cost optimization."""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.api_key)
            use_model = model or self.model
            
            # Check cache first
            if cache_key:
                cached = self.cache.get(cache_key, task_type, use_model)
                if cached:
                    return cached
            
            # Estimate input tokens (rough approximation: 1 token ≈ 4 chars)
            input_tokens = len(prompt) // 4
            
            # Make API call
            message = client.messages.create(
                model=use_model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Calculate cost
            output_tokens = message.usage.output_tokens
            total_tokens = input_tokens + output_tokens
            
            pricing = self.PRICING.get(use_model, self.PRICING[self.MODEL_SONNET])
            input_cost = (input_tokens / 1_000_000) * pricing["input"]
            output_cost = (output_tokens / 1_000_000) * pricing["output"]
            total_cost = input_cost + output_cost
            
            # Only show cost for non-cached responses
            if not cache_key:
                console.print(f"[dim]({total_tokens:,} tokens)[/dim]")
            
            # Extract text from the response content
            # Anthropic returns a list of content blocks (TextBlock objects)
            content_block = message.content[0]
            response = content_block.text
            
            # Cache the response
            if cache_key:
                self.cache.set(cache_key, task_type, use_model, response, total_tokens)
            
            return response
        except ImportError:
            console.print("[red]✗ anthropic package not installed[/red]")
            return ""
        except Exception as e:
            console.print(f"[red]✗ AI call failed: {e}[/red]")
            return ""
    
    def explain_component(self, component, detailed: bool = False) -> str:
        """Generate explanation with aggressive token optimization.
        
        Args:
            component: Either a CodeComponent object or a dict with name, type, content keys
            detailed: Whether to generate a detailed explanation
        """
        if not self.api_key:
            return self._fallback_explain(component)
        
        # Handle both CodeComponent objects and dicts
        if isinstance(component, dict):
            comp_name = component.get("name", "Unknown")
            comp_type = component.get("type", "component")
            comp_content = component.get("content", "")
            comp_signature = component.get("signature", "")
        else:
            comp_name = component.name
            comp_type = component.type
            comp_content = component.content
            comp_signature = getattr(component, "signature", "")
        
        # Compress code first
        compressed = self.optimizer.compress_code(comp_content, max_chars=1200)
        
        # For simple components, just extract signatures
        if not detailed and len(comp_content) > 2000:
            compressed = self.optimizer.extract_key_signatures(comp_content)
        
        # Include signature if available
        if comp_signature and not detailed:
            compressed = f"{comp_signature}\n\n{compressed}"
        
        detail_level = "detailed" if detailed else "concise"
        max_tokens = 800 if detailed else 400
        
        # Create cache key
        cache_key = f"{comp_name}:{comp_type}:{hashlib.sha256(compressed.encode()).hexdigest()[:16]}"
        
        # Ultra-short prompt
        prompt = f"""Explain {comp_type} `{comp_name}`:

```
{compressed}
```

{detail_level} explanation (2-3 bullet points max):
- Purpose:
- Key functionality:
- Usage pattern:"""
        
        return self._call_claude(
            prompt, 
            max_tokens=max_tokens,
            model=self.MODEL_HAIKU,
            cache_key=cache_key,
            task_type="explain"
        )
    
    def _fallback_explain(self, component) -> str:
        """Fallback explanation without AI.
        
        Args:
            component: Either a CodeComponent object or a dict
        """
        # Handle both CodeComponent objects and dicts
        if isinstance(component, dict):
            comp_name = component.get("name", "Unknown")
            comp_type = component.get("type", "component")
            comp_path = component.get("path", "")
            comp_docstring = component.get("content", "")  # dict uses 'content' for docstring
            comp_signature = component.get("signature", "")
        else:
            comp_name = component.name
            comp_type = component.type
            comp_path = component.file_path
            comp_docstring = component.docstring or ""
            comp_signature = component.signature
        
        lines = [
            f"## {comp_name}",
            f"**Type:** {comp_type}",
            f"**File:** `{comp_path}`",
            "",
        ]
        
        if comp_docstring:
            lines.append(comp_docstring[:500])
            lines.append("")
        
        if comp_signature:
            lines.append(f"**Signature:** `{comp_signature}`")
            lines.append("")
        
        lines.append("*Set ANTHROPIC_API_KEY for AI-powered explanations*")
        
        return "\n".join(lines)
    
    def generate_onboarding_guide(
        self,
        structure: Dict[str, Any],
        audience: str
    ) -> str:
        """Generate onboarding guide with batching."""
        if not self.api_key:
            return self._fallback_onboarding(structure, audience)
        
        # Use abbreviated module names
        modules = [m['name'][:30] for m in structure.get("modules", [])[:15]]
        key_files = structure.get("key_files", [])[:8]
        
        # Create cache key from structure hash
        structure_hash = hashlib.sha256(
            json.dumps(modules, sort_keys=True).encode()
        ).hexdigest()[:16]
        cache_key = f"onboarding:{audience}:{structure_hash}"
        
        # Check cache
        cached = self.cache.get(cache_key, "onboarding", self.MODEL_SONNET)
        if cached:
            return cached
        
        # Concise prompt
        prompt = f"""Onboarding guide for {audience}.

Project modules: {', '.join(modules)}
Key files: {', '.join(key_files)}

Create a 400-word guide:
1. Overview (2 sentences)
2. Directory structure
3. Getting started (3 steps)
4. Common commands"""
        
        return self._call_claude(
            prompt, 
            max_tokens=1200,
            model=self.MODEL_SONNET,
            cache_key=cache_key,
            task_type="onboarding"
        )
    
    def _fallback_onboarding(self, structure: Dict[str, Any], audience: str) -> str:
        """Fallback onboarding without AI."""
        lines = [
            f"# Onboarding: {audience}",
            "",
            "## Structure",
        ]
        
        for module in structure.get("modules", [])[:8]:
            lines.append(f"- {module}")
        
        lines.append("\n*Set ANTHROPIC_API_KEY for AI-powered guides*")
        return "\n".join(lines)
    
    def generate_llm_digest(
        self,
        structure: Dict[str, Any],
        compact: bool = False
    ) -> str:
        """Generate AI-optimized codebase summary with aggressive compression."""
        if not self.api_key:
            return self._fallback_digest(structure, compact)
        
        # Ultra-compact module list
        module_limit = 25 if compact else 40
        modules_summary = ", ".join([
            f"{m['name'][:25]}"  # Just names, no descriptions
            for m in structure.get("modules", [])[:module_limit]
        ])
        
        # Create cache key
        structure_hash = hashlib.sha256(modules_summary.encode()).hexdigest()[:16]
        cache_key = f"digest:{compact}:{structure_hash}"
        
        # Check cache - use the configured model
        use_model = self.model if self.model else self.MODEL_SONNET
        cached = self.cache.get(cache_key, "digest", use_model)
        if cached:
            return cached
        
        max_tokens = 2000 if compact else 3500
        
        prompt = f"""Codebase digest.

Modules ({len(structure.get('modules', []))} total): {modules_summary}
Files: {structure.get('file_count', 'N/A')}

Summarize:
1. Architecture pattern
2. Key modules
3. Data flow
4. Entry points

{"Be brief (800 words)." if compact else "Be thorough (1500 words)."}"""
        
        return self._call_claude(
            prompt, 
            max_tokens=max_tokens,
            model=use_model,
            cache_key=cache_key,
            task_type="digest"
        )
    
    def _fallback_digest(self, structure: Dict[str, Any], compact: bool) -> str:
        """Fallback digest without AI."""
        lines = ["# Codebase Digest\n"]
        
        for module in structure.get("modules", [])[:20]:
            lines.append(f"- {module}")
        
        lines.extend([
            "",
            f"Files: {structure.get('file_count', 'N/A')}",
            f"Modules: {structure.get('module_count', 'N/A')}",
            "",
            "*Set ANTHROPIC_API_KEY for AI digests*"
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
        
        # Create cache key
        cache_key = f"readme:{self.config.project.name}:{hashlib.sha256(self.config.project.description.encode()).hexdigest()[:16]}"
        
        cached = self.cache.get(cache_key, "readme", self.MODEL_HAIKU)
        if cached:
            return cached
        
        # Minimal prompt
        prompt = f"""README for {self.config.project.name}

Desc: {self.config.project.description[:100]}

Write 3 sentences + bullet list of 3 features."""
        
        return self._call_claude(
            prompt, 
            max_tokens=500,
            model=self.MODEL_HAIKU,
            cache_key=cache_key,
            task_type="readme"
        )
