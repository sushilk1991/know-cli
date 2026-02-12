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
    
    # Default model configurations with 2026 pricing
    MODELS: Dict[str, ModelInfo] = {
        # OpenAI 2026
        "gpt-4.5": ModelInfo(
            name="gpt-4.5",
            cost_per_1m_input=3.00,
            cost_per_1m_output=12.00,
            quality_score=0.98,
            max_context=256000,
            strengths=["reasoning", "coding", "multimodal"],
        ),
        "gpt-4.5-mini": ModelInfo(
            name="gpt-4.5-mini",
            cost_per_1m_input=0.20,
            cost_per_1m_output=0.80,
            quality_score=0.85,
            max_context=256000,
            strengths=["fast", "cheap", "simple tasks"],
        ),
        # Anthropic 2026
        "claude-4-opus": ModelInfo(
            name="claude-4-opus",
            cost_per_1m_input=20.00,
            cost_per_1m_output=80.00,
            quality_score=0.99,
            max_context=300000,
            strengths=["expert reasoning", "complex coding", "research"],
        ),
        "claude-4-sonnet": ModelInfo(
            name="claude-4-sonnet",
            cost_per_1m_input=3.50,
            cost_per_1m_output=15.00,
            quality_score=0.97,
            max_context=300000,
            strengths=["coding", "reasoning", "long context"],
        ),
        "claude-4-haiku": ModelInfo(
            name="claude-4-haiku",
            cost_per_1m_input=0.30,
            cost_per_1m_output=1.50,
            quality_score=0.82,
            max_context=300000,
            strengths=["fast", "cheap", "simple tasks"],
        ),
        # Kimi (ByteDance) 2026
        "kimi-k2.5-pro": ModelInfo(
            name="kimi-k2.5-pro",
            cost_per_1m_input=0.25,
            cost_per_1m_output=1.00,
            quality_score=0.92,
            max_context=200000,
            strengths=["fast", "coding", "reasoning"],
        ),
        "kimi-k2.5": ModelInfo(
            name="kimi-k2.5",
            cost_per_1m_input=0.10,
            cost_per_1m_output=0.40,
            quality_score=0.80,
            max_context=200000,
            strengths=["cheap", "fast", "simple tasks"],
        ),
        # Google Gemini 2026
        "gemini-2.5-pro": ModelInfo(
            name="gemini-2.5-pro",
            cost_per_1m_input=1.50,
            cost_per_1m_output=6.00,
            quality_score=0.95,
            max_context=500000,
            strengths=["long context", "multimodal", "reasoning"],
        ),
        "gemini-2.5-flash": ModelInfo(
            name="gemini-2.5-flash",
            cost_per_1m_input=0.10,
            cost_per_1m_output=0.40,
            quality_score=0.82,
            max_context=500000,
            strengths=["fast", "cheap", "simple tasks"],
        ),
        # GLM (Zhipu) 2026
        "glm-5": ModelInfo(
            name="glm-5",
            cost_per_1m_input=0.08,
            cost_per_1m_output=0.30,
            quality_score=0.85,
            max_context=200000,
            strengths=["cheap", "coding", "multilingual"],
        ),
        "glm-5-plus": ModelInfo(
            name="glm-5-plus",
            cost_per_1m_input=0.50,
            cost_per_1m_output=2.00,
            quality_score=0.94,
            max_context=200000,
            strengths=["reasoning", "coding", "long context"],
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
    ) -> str:
        """Return cheapest model meeting quality threshold.
        
        Args:
            task: Description of the task (used for complexity detection)
            quality_threshold: Minimum quality score required (0.0-1.0)
            max_context_needed: Minimum context window required
            prefer_speed: If True, prefer faster models among candidates
        
        Returns:
            Name of the recommended model
        """
        # Detect complexity and adjust threshold if needed
        complexity = self._detect_complexity(task)
        if complexity == "simple":
            quality_threshold = min(quality_threshold, 0.75)
        elif complexity == "expert":
            quality_threshold = max(quality_threshold, 0.90)
        
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
        
        # Sort by cost (prefer speed if requested)
        if prefer_speed:
            # Use a combination of cost and quality as tiebreaker
            candidates.sort(key=lambda x: (x[1].cost_per_1m_input, -x[1].quality_score))
        else:
            candidates.sort(key=lambda x: x[1].cost_per_1m_input)
        
        return candidates[0][0]
    
    def route_with_cost(
        self,
        task: str,
        input_tokens: int,
        output_tokens: int,
        quality_threshold: float = 0.8,
    ) -> Dict:
        """Route and return detailed cost breakdown.
        
        Args:
            task: Description of the task
            input_tokens: Estimated input tokens
            output_tokens: Estimated output tokens
            quality_threshold: Minimum quality score required
        
        Returns:
            Dict with recommended model and cost analysis
        """
        recommended = self.route(task, quality_threshold)
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