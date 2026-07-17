"""File system watcher for auto-updating documentation."""

import time
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Set

from rich.console import Console
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from know.path_filters import is_hard_excluded_part

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
        self._lock = threading.RLock()
        self._update_lock = threading.Lock()
        self._trailing_timer: threading.Timer | None = None
        self._timer_generation = 0
        self._closed = False
    
    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification."""
        if event.is_directory:
            return
        
        path = Path(event.src_path)
        
        # Skip certain files
        if self._should_ignore(path):
            return
        
        trigger_now = False
        with self._lock:
            if self._closed:
                return
            self.pending_paths.add(path)

            # Keep the leading-edge update for responsiveness. Events arriving
            # during the debounce window also arm a trailing flush, so the last
            # burst is not stranded until some unrelated future event.
            current_time = time.time()
            if current_time - self.last_update >= self.debounce_seconds:
                self._cancel_timer_locked()
                trigger_now = True
            else:
                self._cancel_timer_locked()
                generation = self._timer_generation
                self._trailing_timer = threading.Timer(
                    max(0.0, float(self.debounce_seconds)),
                    self._flush_trailing,
                    args=(generation,),
                )
                self._trailing_timer.daemon = True
                self._trailing_timer.start()

        if trigger_now:
            self._trigger_update()
    
    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation."""
        self.on_modified(event)
    
    def _should_ignore(self, path: Path) -> bool:
        """Check if file should be ignored using path component matching."""
        # Directories/files to ignore (matched against individual path components)
        ignore_dirs = {".idea", ".vscode"}
        
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
            if part in ignore_dirs or is_hard_excluded_part(part):
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
        # Timer callbacks can arrive while a leading-edge scan is still
        # running. Serialize the complete scan/generate cycle; the second
        # caller will drain paths accumulated during the first cycle.
        with self._update_lock:
            self._trigger_update_serialized()

    def _trigger_update_serialized(self) -> None:
        """Run one already-serialized documentation update."""
        with self._lock:
            if self._closed or not self.pending_paths:
                return
            self.last_update = time.time()
            pending_paths = set(self.pending_paths)
            self.pending_paths.clear()
            self._trailing_timer = None
        
        from know.scanner import CodebaseScanner
        from know.generator import DocGenerator
        
        pending_count = len(pending_paths)
        console.print(f"\n[dim]Detected changes in {pending_count} files, updating docs...[/dim]")
        
        try:
            generator = DocGenerator(self.config)
            from know.parsers import ParserFactory

            # Filter to only code files we care about
            supported_exts = ParserFactory.supported_extensions()
            code_paths = [
                p for p in pending_paths
                if p.suffix in supported_exts
            ]
            
            with CodebaseScanner(self.config) as scanner:
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

    def _cancel_timer_locked(self) -> None:
        """Cancel the pending trailing timer while holding ``_lock``."""
        if self._trailing_timer is not None:
            self._trailing_timer.cancel()
            self._trailing_timer = None
        self._timer_generation += 1

    def _flush_trailing(self, generation: int) -> None:
        """Flush a completed debounce burst from the timer thread."""
        # Recheck the generation only after any active scan completes. Events
        # arriving while this callback waits may replace this timer, and a
        # stale callback must not flush their newer debounce burst early.
        with self._update_lock:
            with self._lock:
                if self._closed or generation != self._timer_generation:
                    return
                self._trailing_timer = None
            self._trigger_update_serialized()

    def close(self) -> None:
        """Cancel pending timer work and reject future events."""
        with self._lock:
            self._closed = True
            self._cancel_timer_locked()
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
            pass
        finally:
            self.handler.close()
            self.observer.stop()
            self.observer.join()
    
    def stop(self) -> None:
        """Stop the watcher."""
        self.handler.close()
        self.observer.stop()
        self.observer.join()
