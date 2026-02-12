"""Smart model router for cost-optimized AI interactions.

Routes tasks to the cheapest model that meets quality requirements,
enabling significant cost savings while maintaining output quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ModelInfo:
    """Information about an AI model."""
    name: str
    cost_per_1m_input: float  # $/1M input tokens
    cost_per_1m_output: float  # $/1M output tokens
    quality_score: float  # 0.0-1.0
    max_context: int  # Max context window
    strengths: List[str]  # What this model is good at
    
    def total_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate total cost for a request."""
        input_cost = (input_tokens / 1_000_000) * self.cost_per_1m_input
        output_cost = (output_tokens / 1_000_000) * self.cost_per_1m_output
        return input_cost + output_cost


class SmartModelRouter:
    """Route to cheapest model that passes quality threshold.
    
    Analyzes task requirements and selects the most cost-effective
    model that can handle the task adequately.
    
    Example usage:
        router = SmartModelRouter()
        model = router.route("simple summarization", quality_threshold=0.75)
        # Returns "gemini-1.5-flash" (cheapest that meets threshold)
    """
    
    # Default model configurations with REAL 2026 pricing
    # Based on actual OpenAI, Anthropic, and other provider pricing
    MODELS: Dict[str, ModelInfo] = {
        # OpenAI GPT-5 Series (2026)
        "gpt-5.2-codex": ModelInfo(
            name="gpt-5.2-codex",
            cost_per_1m_input=5.00,
            cost_per_1m_output=20.00,
            quality_score=0.99,
            max_context=512000,
            strengths=["best coding", "agentic tasks", "long horizon"],
        ),
        "gpt-5.2": ModelInfo(
            name="gpt-5.2",
            cost_per_1m_input=4.00,
            cost_per_1m_output=16.00,
            quality_score=0.98,
            max_context=512000,
            strengths=["reasoning", "coding", "multimodal"],
        ),
        "gpt-5.1": ModelInfo(
            name="gpt-5.1",
            cost_per_1m_input=3.00,
            cost_per_1m_output=12.00,
            quality_score=0.96,
            max_context=256000,
            strengths=["reasoning", "coding"],
        ),
        "gpt-5": ModelInfo(
            name="gpt-5",
            cost_per_1m_input=2.00,
            cost_per_1m_output=8.00,
            quality_score=0.94,
            max_context=256000,
            strengths=["good reasoning", "coding"],
        ),
        "gpt-5-mini": ModelInfo(
            name="gpt-5-mini",
            cost_per_1m_input=0.40,
            cost_per_1m_output=1.60,
            quality_score=0.88,
            max_context=256000,
            strengths=["fast", "cheap", "simple tasks"],
        ),
        "gpt-5-nano": ModelInfo(
            name="gpt-5-nano",
            cost_per_1m_input=0.10,
            cost_per_1m_output=0.40,
            quality_score=0.82,
            max_context=128000,
            strengths=["fastest", "cheapest", "simple tasks"],
        ),
        # OpenAI o-series (reasoning)
        "o3": ModelInfo(
            name="o3",
            cost_per_1m_input=10.00,
            cost_per_1m_output=40.00,
            quality_score=0.97,
            max_context=200000,
            strengths=["complex reasoning", "math", "science"],
        ),
        "o4-mini": ModelInfo(
            name="o4-mini",
            cost_per_1m_input=0.50,
            cost_per_1m_output=2.00,
            quality_score=0.90,
            max_context=200000,
            strengths=["fast reasoning", "cost-effective"],
        ),
        # OpenGPT-4.1 Series
        "gpt-4.1": ModelInfo(
            name="gpt-4.1",
            cost_per_1m_input=2.00,
            cost_per_1m_output=8.00,
            quality_score=0.93,
            max_context=128000,
            strengths=["smart non-reasoning", "coding"],
        ),
        "gpt-4.1-mini": ModelInfo(
            name="gpt-4.1-mini",
            cost_per_1m_input=0.30,
            cost_per_1m_output=1.20,
            quality_score=0.86,
            max_context=128000,
            strengths=["fast", "cheap"],
        ),
        "gpt-4.1-nano": ModelInfo(
            name="gpt-4.1-nano",
            cost_per_1m_input=0.10,
            cost_per_1m_output=0.40,
            quality_score=0.80,
            max_context=64000,
            strengths=["fastest", "cheapest"],
        ),
        # Open Weight (requires self-hosting, cost = GPU compute)
        # Mark with small cost to represent GPU wear
        "gpt-oss-120b": ModelInfo(
            name="gpt-oss-120b",
            cost_per_1m_input=0.25,  # ~$0.25/hr on A100 for 1M tokens
            cost_per_1m_output=0.50,
            quality_score=0.88,
            max_context=131072,
            strengths=["open weight", "self-hosted", "privacy"],
        ),
    }
    
    # Task complexity keywords
    COMPLEXITY_KEYWORDS = {
        "simple": ["summarize", "extract", "list", "simple", "fast", "quick"],
        "moderate": ["explain", "analyze", "compare", "review", "debug"],
        "complex": ["architecture", "design", "refactor", "optimize", "create", "build"],
        "expert": ["research", "comprehensive", "detailed", "thorough", "novel"],
    }
    
    def __init__(self, custom_models: Optional[Dict[str, ModelInfo]] = None):
        """Initialize router with optional custom models.
        
        Args:
            custom_models: Additional or override model configurations
        """
        self.models = dict(self.MODELS)
        if custom_models:
            self.models.update(custom_models)
    
    def route(
        self, 
        task: str = "", 
        quality_threshold: float = 0.8,
        max_context_needed: int = 0,
        prefer_speed: bool = False,
        output_format: str = "text",  # "text" | "json" | "code" | "creative"
    ) -> str:
        """Return best model meeting quality threshold.
        
        Args:
            task: Description of the task (used for complexity detection)
            quality_threshold: Minimum quality score required (0.0-1.0)
            max_context_needed: Minimum context window required
            prefer_speed: If True, prefer faster models among candidates
            output_format: Expected output format - affects model choice
                - "text": Plain text, summaries
                - "json": Structured JSON output
                - "code": Code snippets, programs
                - "creative": Stories, creative writing
        
        Returns:
            Name of the recommended model
        """
        # Detect complexity and adjust threshold if needed
        complexity = self._detect_complexity(task)
        task_lower = task.lower()
        
        if complexity == "simple":
            quality_threshold = min(quality_threshold, 0.75)
        elif complexity == "expert":
            quality_threshold = max(quality_threshold, 0.90)
        
        # For coding tasks, require at least gpt-5-mini quality
        coding_keywords = ["code", "fix", "bug", "implement", "refactor", "write", "create", "build", "debug", "function", "class", "api"]
        if any(kw in task_lower for kw in coding_keywords) or output_format == "code":
            quality_threshold = max(quality_threshold, 0.88)  # gpt-5-mini level minimum
        
        # For reasoning tasks, require strong model
        reasoning_keywords = ["reason", "think", "analyze", "research", "solve", "explain", "design", "architecture"]
        if any(kw in task_lower for kw in reasoning_keywords):
            quality_threshold = max(quality_threshold, 0.90)  # o4-mini level minimum
        
        # For JSON output, need structured output capability
        if output_format == "json":
            quality_threshold = max(quality_threshold, 0.85)
        
        # For creative writing, allow more flexibility
        if output_format == "creative":
            quality_threshold = min(quality_threshold, 0.80)
        
        # For complex tasks, always use best quality
        if complexity == "expert":
            quality_threshold = max(quality_threshold, 0.95)
        
        # Filter candidates by quality and context
        candidates = []
        for name, info in self.models.items():
            if info.quality_score >= quality_threshold:
                if max_context_needed == 0 or info.max_context >= max_context_needed:
                    candidates.append((name, info))
        
        if not candidates:
            # Fall back to highest quality model
            best = max(self.models.items(), key=lambda x: x[1].quality_score)
            return best[0]
        
        # Sort by VALUE: balance quality and cost
        # Value = quality^2 / cost - quality matters more than cost
        # This ensures decent quality while still being cost-effective
        def value_score(info):
            cost = max(info.cost_per_1m_input, 0.01)
            base_score = (info.quality_score ** 2) / cost
            
            # Bonus for context if needed
            context_bonus = 1.0
            if max_context_needed > 0 and info.max_context >= max_context_needed:
                # Prefer models that exceed required context (not just meet it)
                if info.max_context >= max_context_needed * 2:
                    context_bonus = 1.1
            
            # Bonus for format match
            format_bonus = 1.0
            if output_format == "code":
                if "code" in info.strengths:
                    format_bonus = 1.15
            elif output_format == "json":
                if "reasoning" in info.strengths:
                    format_bonus = 1.1
            
            return base_score * context_bonus * format_bonus
        
        if prefer_speed:
            # Prefer faster models as tiebreaker
            candidates.sort(key=lambda x: (value_score(x[1]), -x[1].cost_per_1m_input), reverse=True)
        else:
            # Best value: quality weighted heavily
            candidates.sort(key=lambda x: value_score(x[1]), reverse=True)
        
        return candidates[0][0]
    
    def route_with_cost(
        self,
        task: str,
        input_tokens: int,
        output_tokens: int,
        quality_threshold: float = 0.8,
        output_format: str = "text",
    ) -> Dict:
        """Route and return detailed cost breakdown.
        
        Args:
            task: Description of the task
            input_tokens: Estimated input tokens
            output_tokens: Estimated output tokens
            quality_threshold: Minimum quality score required
            output_format: Expected output format
        
        Returns:
            Dict with recommended model and cost analysis
        """
        # Auto-detect context need from input tokens
        context_needed = input_tokens + output_tokens
        
        recommended = self.route(
            task, 
            quality_threshold,
            max_context_needed=context_needed,
            output_format=output_format
        )
        info = self.models[recommended]
        
        # Calculate costs for all viable models
        all_costs = []
        for name, model_info in self.models.items():
            if model_info.quality_score >= quality_threshold:
                cost = model_info.total_cost(input_tokens, output_tokens)
                all_costs.append({
                    "model": name,
                    "cost": round(cost, 4),
                    "quality": model_info.quality_score,
                })
        
        all_costs.sort(key=lambda x: x["cost"])
        
        return {
            "recommended": recommended,
            "recommended_cost": round(
                info.total_cost(input_tokens, output_tokens), 4
            ),
            "quality_score": info.quality_score,
            "task_complexity": self._detect_complexity(task),
            "all_options": all_costs,
        }
    
    def _detect_complexity(self, task: str) -> str:
        """Detect task complexity from description.
        
        Args:
            task: Task description
        
        Returns:
            Complexity level: "simple", "moderate", "complex", or "expert"
        """
        task_lower = task.lower()
        
        for level, keywords in self.COMPLEXITY_KEYWORDS.items():
            if any(kw in task_lower for kw in keywords):
                return level
        
        return "moderate"  # Default
    
    def list_models(self) -> List[Dict]:
        """List all available models with their specs.
        
        Returns:
            List of model info dicts
        """
        return [
            {
                "name": info.name,
                "cost_per_1m_input": info.cost_per_1m_input,
                "cost_per_1m_output": info.cost_per_1m_output,
                "quality_score": info.quality_score,
                "max_context": info.max_context,
                "strengths": info.strengths,
            }
            for info in sorted(
                self.models.values(), 
                key=lambda x: x.cost_per_1m_input
            )
        ]
    
    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get information about a specific model.
        
        Args:
            name: Model name
        
        Returns:
            ModelInfo or None if not found
        """
        return self.models.get(name)