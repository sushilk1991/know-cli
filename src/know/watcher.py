"""File system watcher for auto-updating documentation."""

import time
from pathlib import Path
from typing import TYPE_CHECKING, Set

from rich.console import Console
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

if TYPE_CHECKING:
    from know.config import Config

console = Console()


class DocUpdateHandler(FileSystemEventHandler):
    """Handles file system events and triggers doc updates."""
    
    def __init__(self, config: "Config"):
        self.config = config
        self.debounce_seconds = config.output.watch.debounce_seconds
        self.last_update = 0
        self.pending_paths: Set[Path] = set()
    
    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification."""
        if event.is_directory:
            return
        
        path = Path(event.src_path)
        
        # Skip certain files
        if self._should_ignore(path):
            return
        
        self.pending_paths.add(path)
        
        # Debounce updates
        current_time = time.time()
        if current_time - self.last_update < self.debounce_seconds:
            return
        
        self._trigger_update()
    
    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation."""
        self.on_modified(event)
    
    def _should_ignore(self, path: Path) -> bool:
        """Check if file should be ignored using path component matching."""
        # Directories/files to ignore (matched against individual path components)
        ignore_dirs = {
            ".git",
            ".know",
            "__pycache__",
            ".venv",
            "venv",
            "node_modules",
            ".idea",
            ".vscode",
        }
        
        # File patterns to ignore (matched against filename)
        ignore_suffixes = {
            ".pyc",
            ".pyo",
            ".swp",
            ".swo",
            "~",
        }
        
        ignore_files = {
            ".DS_Store",
        }
        
        # Check individual path components (prevents "docs" matching "dockerfiles")
        for part in path.parts:
            if part in ignore_dirs:
                return True
        
        # Check filename patterns
        filename = path.name
        if filename in ignore_files:
            return True
        for suffix in ignore_suffixes:
            if filename.endswith(suffix):
                return True
        
        # Check if in output directory
        try:
            path.relative_to(Path(self.config.output.directory).resolve())
            return True
        except ValueError:
            pass
        
        return False
    
    def _trigger_update(self) -> None:
        """Trigger documentation update."""
        self.last_update = time.time()
        
        if not self.pending_paths:
            return
        
        from know.scanner import CodebaseScanner
        from know.generator import DocGenerator
        
        pending_count = len(self.pending_paths)
        console.print(f"\n[dim]Detected changes in {pending_count} files, updating docs...[/dim]")
        
        try:
            scanner = CodebaseScanner(self.config)
            generator = DocGenerator(self.config)
            
            # Filter to only code files we care about
            code_paths = [
                p for p in self.pending_paths 
                if p.suffix in {".py", ".ts", ".tsx", ".js", ".jsx", ".go"}
            ]
            
            if code_paths:
                # Use incremental scan for specific files
                stats = scanner.scan_files(code_paths)
            else:
                # No code files changed, do full scan
                stats = scanner.scan()
            
            structure = scanner.get_structure()
            generator.generate_readme(structure)
            
            # Show efficiency stats
            total = stats.get("files", 0)
            changed = stats.get("changed_files", pending_count)
            cached = total - changed
            
            if cached > 0:
                console.print(
                    f"[green]✓[/green] Documentation updated "
                    f"({changed} changed, {cached} from cache)"
                )
            else:
                console.print(f"[green]✓[/green] Documentation updated ({total} files)")
                
        except Exception as e:
            console.print(f"[red]✗[/red] Update failed: {e}")
        
        self.pending_paths.clear()


class FileWatcher:
    """Watches files and auto-updates documentation."""
    
    def __init__(self, config: "Config"):
        self.config = config
        self.observer: Observer = Observer()
        self.handler = DocUpdateHandler(config)
    
    def run(self) -> None:
        """Run the file watcher (blocking)."""
        # Watch the root directory
        self.observer.schedule(
            self.handler,
            str(self.config.root),
            recursive=True
        )
        
        self.observer.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
        
        self.observer.join()
    
    def stop(self) -> None:
        """Stop the watcher."""
        self.observer.stop()
        self.observer.join()
