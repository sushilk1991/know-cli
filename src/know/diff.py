"""Diff mode for tracking architectural changes safely using git worktrees."""

import subprocess
import tempfile
import time
import atexit
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json


class ArchitectureDiff:
    """Track changes in codebase architecture over time using safe git operations."""
    
    def __init__(self, root: Path):
        self.root = root
    
    def _run_git(self, args: List[str], cwd: Optional[Path] = None) -> str:
        """Run a git command and return output."""
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            cwd=str(cwd or self.root)
        )
        return result.stdout.strip()
    
    def _check_git_repo(self) -> bool:
        """Check if the directory is a git repository."""
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            cwd=str(self.root)
        )
        return result.returncode == 0
    
    def get_commit_history(self, since: Optional[str] = None, n: int = 10) -> List[Dict]:
        """Get recent commit history."""
        if not self._check_git_repo():
            return []
        
        # Use null byte separator to handle pipes in commit messages
        args = ["log", "--pretty=format:%H%x00%s%x00%ci", "-n", str(n)]
        
        if since:
            args.extend(["--since", since])
        
        output = self._run_git(args)
        commits = []
        
        for line in output.split("\n"):
            if not line:
                continue
            parts = line.split("\x00")
            if len(parts) >= 3:
                commits.append({
                    "hash": parts[0][:8],
                    "full_hash": parts[0],
                    "message": parts[1],
                    "date": parts[2]
                })
        
        return commits
    
    def get_changed_files(self, commit_hash: str) -> List[str]:
        """Get files changed in a commit."""
        if not self._check_git_repo():
            return []
        
        output = self._run_git(["diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash])
        return [f for f in output.split("\n") if f]
    
    def get_structure_at_commit(self, commit_hash: str) -> Dict[str, Any]:
        """Get codebase structure at a specific commit using git worktree (safe)."""
        if not self._check_git_repo():
            return {"error": "Not a git repository"}
        
        # Create temporary worktree - leaves user's workspace untouched
        with tempfile.TemporaryDirectory() as temp_dir:
            # Include timestamp to avoid conflicts if multiple runs occur
            worktree_path = Path(temp_dir) / f"worktree_{int(time.time())}_{commit_hash[:8]}"
            
            def cleanup():
                try:
                    subprocess.run(
                        ["git", "worktree", "remove", "--force", str(worktree_path)],
                        capture_output=True,
                        cwd=str(self.root)
                    )
                except Exception:
                    pass
            
            # Register atexit in case of sudden termination
            atexit.register(cleanup)
            
            try:
                # Add worktree at the specific commit
                result = subprocess.run(
                    ["git", "worktree", "add", "--detach", str(worktree_path), commit_hash],
                    capture_output=True,
                    text=True,
                    cwd=str(self.root)
                )
                
                if result.returncode != 0:
                    return {"error": f"Failed to create worktree: {result.stderr}"}
                
                # Scan the worktree
                try:
                    from know.config import Config
                    from know.scanner import CodebaseScanner
                    
                    config = Config.create_default(worktree_path)
                    scanner = CodebaseScanner(config)
                    structure = scanner.get_structure()
                    
                    return {
                        "files": structure.get("file_count", 0),
                        "functions": structure.get("function_count", 0),
                        "classes": structure.get("class_count", 0),
                        "modules": [m["name"] for m in structure.get("modules", [])[:20]]
                    }
                    
                except ImportError as e:
                    return {"error": f"Scanner not available: {e}"}
                    
            finally:
                # Always clean up worktree
                cleanup()
                try:
                    atexit.unregister(cleanup)
                except AttributeError:
                    pass # Python < 3 compat
    
    def compare_commits(self, base_hash: str, head_hash: str) -> Dict[str, Any]:
        """Compare two commits and return structural differences."""
        base_structure = self.get_structure_at_commit(base_hash)
        head_structure = self.get_structure_at_commit(head_hash)
        
        if "error" in base_structure or "error" in head_structure:
            return {"error": "Failed to get structure for one or both commits"}
        
        return {
            "base": base_structure,
            "head": head_structure,
            "file_diff": head_structure.get("files", 0) - base_structure.get("files", 0),
            "function_diff": head_structure.get("functions", 0) - base_structure.get("functions", 0),
            "class_diff": head_structure.get("classes", 0) - base_structure.get("classes", 0),
        }
    
    def generate_diff(self, since: str = "1 week ago") -> str:
        """Generate architectural diff report."""
        if not self._check_git_repo():
            return "# Error\n\nNot a git repository."
        
        commits = self.get_commit_history(since=since)
        
        if not commits:
            return f"# 🏗️ Architecture Changes\n\nNo commits found since {since}."
        
        lines = [
            "# 🏗️ Architecture Changes",
            "",
            f"Period: {since}",
            f"Commits: {len(commits)}",
            "",
            "## Summary",
            ""
        ]
        
        # Get stats
        total_files_changed = set()
        for commit in commits:
            files = self.get_changed_files(commit["full_hash"])
            total_files_changed.update(files)
        
        lines.extend([
            f"- **Files modified:** {len(total_files_changed)}",
            f"- **Commits:** {len(commits)}",
            "",
            "## Recent Commits",
            ""
        ])
        
        for commit in commits[:10]:
            lines.append(f"- `{commit['hash']}` {commit['message']}")
        
        lines.extend([
            "",
            "## Modified Files",
            ""
        ])
        
        # Group by type
        py_files = [f for f in total_files_changed if f.endswith(".py")]
        ts_files = [f for f in total_files_changed if f.endswith((".ts", ".tsx"))]
        js_files = [f for f in total_files_changed if f.endswith((".js", ".jsx"))]
        other_files = [f for f in total_files_changed if not f.endswith((".py", ".ts", ".tsx", ".js", ".jsx"))]
        
        if py_files:
            lines.append("### Python Files")
            for f in sorted(py_files)[:10]:
                lines.append(f"- `{f}`")
            if len(py_files) > 10:
                lines.append(f"- ... and {len(py_files) - 10} more")
            lines.append("")
        
        if ts_files:
            lines.append("### TypeScript Files")
            for f in sorted(ts_files)[:10]:
                lines.append(f"- `{f}`")
            if len(ts_files) > 10:
                lines.append(f"- ... and {len(ts_files) - 10} more")
            lines.append("")
        
        if js_files:
            lines.append("### JavaScript Files")
            for f in sorted(js_files)[:10]:
                lines.append(f"- `{f}`")
            if len(js_files) > 10:
                lines.append(f"- ... and {len(js_files) - 10} more")
            lines.append("")
        
        if other_files:
            lines.append("### Other Files")
            for f in sorted(other_files)[:10]:
                lines.append(f"- `{f}`")
            if len(other_files) > 10:
                lines.append(f"- ... and {len(other_files) - 10} more")
        
        # Try to get structural comparison between first and last commit
        if len(commits) >= 2:
            lines.extend([
                "",
                "## Structural Changes",
                ""
            ])
            
            try:
                comparison = self.compare_commits(commits[-1]["full_hash"], commits[0]["full_hash"])
                if "error" not in comparison:
                    file_diff = comparison.get("file_diff", 0)
                    func_diff = comparison.get("function_diff", 0)
                    class_diff = comparison.get("class_diff", 0)
                    
                    file_sign = "+" if file_diff >= 0 else ""
                    func_sign = "+" if func_diff >= 0 else ""
                    class_sign = "+" if class_diff >= 0 else ""
                    
                    lines.extend([
                        f"- Files: {file_sign}{file_diff}",
                        f"- Functions: {func_sign}{func_diff}",
                        f"- Classes: {class_sign}{class_diff}",
                    ])
            except Exception:
                pass  # Structural analysis is optional
        
        lines.extend([
            "",
            "---",
            "",
            "*Generated by know diff*"
        ])
        
        return "\n".join(lines)
    
    def save_diff(self, since: str = "1 week ago", output: Optional[Path] = None) -> Path:
        """Generate and save diff report."""
        diff_content = self.generate_diff(since)
        
        if output is None:
            output = self.root / "docs" / "architecture-diff.md"
        
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(diff_content)
        
        return output
