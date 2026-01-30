"""Code quality metrics and dashboard."""

from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import json


@dataclass
class CodeMetrics:
    """Metrics for a codebase."""
    total_files: int = 0
    total_lines: int = 0
    total_functions: int = 0
    total_classes: int = 0
    avg_file_length: float = 0.0
    avg_function_length: float = 0.0
    documentation_coverage: float = 0.0
    test_coverage_estimate: float = 0.0
    complexity_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QualityAnalyzer:
    """Analyze code quality metrics."""
    
    def __init__(self, root: Path):
        self.root = root
    
    def analyze_file(self, path: Path) -> Dict[str, Any]:
        """Analyze a single file."""
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")
            
            # Count different types of lines
            code_lines = 0
            comment_lines = 0
            blank_lines = 0
            
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    blank_lines += 1
                elif stripped.startswith("#") or stripped.startswith("//") or stripped.startswith("/*"):
                    comment_lines += 1
                else:
                    code_lines += 1
            
            # Detect docstrings/comments
            has_docstring = '"""' in content or "'''" in content or "/**" in content
            
            return {
                "path": str(path.relative_to(self.root)),
                "total_lines": len(lines),
                "code_lines": code_lines,
                "comment_lines": comment_lines,
                "blank_lines": blank_lines,
                "has_docstring": has_docstring,
                "complexity_estimate": self._estimate_complexity(content)
            }
        except Exception:
            return {}
    
    def _estimate_complexity(self, content: str) -> int:
        """Estimate cyclomatic complexity from content."""
        complexity = 1
        
        # Count branching statements
        complexity += content.count("if ")
        complexity += content.count("elif ")
        complexity += content.count("else:")
        complexity += content.count("for ")
        complexity += content.count("while ")
        complexity += content.count("except")
        complexity += content.count("with ")
        complexity += content.count("and ")
        complexity += content.count("or ")
        
        return complexity
    
    def analyze_structure(self, structure: Dict[str, Any]) -> CodeMetrics:
        """Analyze codebase structure."""
        metrics = CodeMetrics()
        
        metrics.total_files = structure.get("file_count", 0)
        metrics.total_functions = structure.get("function_count", 0)
        metrics.total_classes = structure.get("class_count", 0)
        
        # Analyze files
        total_lines = 0
        documented_files = 0
        test_files = 0
        total_complexity = 0
        
        for module in structure.get("modules", []):
            path = self.root / module.get("path", "")
            if path.exists():
                analysis = self.analyze_file(path)
                if analysis:
                    total_lines += analysis.get("total_lines", 0)
                    if analysis.get("has_docstring"):
                        documented_files += 1
                    if "test" in str(path).lower():
                        test_files += 1
                    total_complexity += analysis.get("complexity_estimate", 0)
        
        metrics.total_lines = total_lines
        
        if metrics.total_files > 0:
            metrics.avg_file_length = total_lines / metrics.total_files
            metrics.documentation_coverage = (documented_files / metrics.total_files) * 100
            metrics.test_coverage_estimate = (test_files / metrics.total_files) * 100
        
        if metrics.total_functions > 0:
            metrics.avg_function_length = total_lines / metrics.total_functions
        
        # Complexity score (0-100)
        if metrics.total_functions > 0:
            avg_complexity = total_complexity / metrics.total_functions
            metrics.complexity_score = min(100, avg_complexity * 10)
        
        return metrics
    
    def find_hot_files(self, structure: Dict[str, Any], top_n: int = 10) -> List[Dict]:
        """Find most complex/largest files."""
        files = []
        
        for module in structure.get("modules", [])[:50]:
            path = self.root / module.get("path", "")
            if path.exists():
                analysis = self.analyze_file(path)
                if analysis:
                    files.append({
                        "path": analysis["path"],
                        "lines": analysis["total_lines"],
                        "complexity": analysis["complexity_estimate"],
                        "score": analysis["total_lines"] * 0.5 + analysis["complexity_estimate"] * 2
                    })
        
        files.sort(key=lambda x: x["score"], reverse=True)
        return files[:top_n]
    
    def find_orphans(self, structure: Dict[str, Any]) -> List[str]:
        """Find potentially orphaned/unused code."""
        orphans = []
        
        # Simple heuristic: files not imported by others
        all_imports = set()
        all_files = set()
        
        for module in structure.get("modules", []):
            path = str(module.get("path", ""))
            all_files.add(path)
            for imp in module.get("imports", []):
                all_imports.add(imp)
        
        # Check for files that might be unused
        # This is a simple heuristic - real analysis would need import resolution
        for f in all_files:
            module_name = f.replace("/", ".").replace(".py", "")
            if module_name not in all_imports and "/test" not in f:
                if len(structure.get("modules", [])) > 20:  # Only in larger projects
                    orphans.append(f)
        
        return orphans[:10]  # Limit results
    
    def generate_dashboard(self, structure: Dict[str, Any]) -> str:
        """Generate a markdown dashboard."""
        metrics = self.analyze_structure(structure)
        hot_files = self.find_hot_files(structure)
        orphans = self.find_orphans(structure)
        
        lines = [
            "# 📊 Code Quality Dashboard",
            "",
            "## Overview",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Files | {metrics.total_files} |",
            f"| Total Lines | {metrics.total_lines:,} |",
            f"| Total Functions | {metrics.total_functions} |",
            f"| Total Classes | {metrics.total_classes} |",
            f"| Avg File Length | {metrics.avg_file_length:.1f} lines |",
            f"| Avg Function Length | {metrics.avg_function_length:.1f} lines |",
            f"| Documentation Coverage | {metrics.documentation_coverage:.1f}% |",
            f"| Test Files | {metrics.test_coverage_estimate:.1f}% |",
            f"| Complexity Score | {metrics.complexity_score:.1f}/100 |",
            "",
            "## 🔥 Hot Files (Most Complex)",
            "",
        ]
        
        for i, f in enumerate(hot_files, 1):
            lines.append(f"{i}. `{f['path']}` - {f['lines']} lines, complexity: {f['complexity']}")
        
        if orphans:
            lines.extend([
                "",
                "## ⚠️ Potential Orphans",
                "",
                "Files that may be unused:",
                ""
            ])
            for f in orphans[:5]:
                lines.append(f"- `{f}`")
        
        lines.extend([
            "",
            "## 📈 Recommendations",
            "",
        ])
        
        if metrics.documentation_coverage < 30:
            lines.append("- ⚠️ Low documentation coverage. Consider adding docstrings.")
        elif metrics.documentation_coverage > 70:
            lines.append("- ✅ Good documentation coverage!")
        
        if metrics.test_coverage_estimate < 10:
            lines.append("- ⚠️ Low test coverage. Consider adding more tests.")
        
        if metrics.complexity_score > 50:
            lines.append("- ⚠️ High complexity detected. Consider refactoring complex functions.")
        
        lines.extend([
            "",
            "---",
            "",
            "*Generated by know-cli*"
        ])
        
        return "\n".join(lines)
    
    def save_dashboard(self, structure: Dict[str, Any], output: Path = None) -> Path:
        """Generate and save dashboard."""
        dashboard = self.generate_dashboard(structure)
        
        if output is None:
            output = self.root / "docs" / "quality-dashboard.md"
        
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(dashboard)
        
        return output
