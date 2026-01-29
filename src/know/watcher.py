"""File system watcher for auto-updating documentation."""

import time
from pathlib import Path
from typing import TYPE_CHECKING, Set

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

if TYPE_CHECKING:
    from know.config import Config


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
        """Check if file should be ignored."""
        ignore_patterns = [
            ".git",
            ".know",
            "__pycache__",
            ".pyc",
            ".pyo",
            ".venv",
            "venv",
            "node_modules",
            ".idea",
            ".vscode",
            ".DS_Store",
            ".swp",
            ".swo",
            "~",
        ]
        
        path_str = str(path)
        
        for pattern in ignore_patterns:
            if pattern in path_str:
                return True
        
        # Check if in output directory
        if self.config.output.directory in path_str:
            return True
        
        return False
    
    def _trigger_update(self) -> None:
        """Trigger documentation update."""
        self.last_update = time.time()
        
        if not self.pending_paths:
            return
        
        from know.scanner import CodebaseScanner
        from know.generator import DocGenerator
        
        pending_count = len(self.pending_paths)
        print(f"\n[dim]Detected changes in {pending_count} files, updating docs...[/dim]")
        
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
                print(f"[green]✓[/green] Documentation updated ({changed} changed, {cached} from cache)")
            else:
                print(f"[green]✓[/green] Documentation updated ({total} files)")
                
        except Exception as e:
            print(f"[red]✗[/red] Update failed: {e}")
        
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
    
    def run_daemon(self) -> None:
        """Run as a daemon process."""
        import daemon
        
        with daemon.DaemonContext():
            self.run()
    
    def stop(self) -> None:
        """Stop the watcher."""
        self.observer.stop()
        self.observer.join()
