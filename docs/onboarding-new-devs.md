# Onboarding: new team members

## Structure
- {'name': 'src.know.ai', 'path': 'src/know/ai.py', 'description': 'AI integration for intelligent code understanding with advanced token optimization.', 'function_count': 0, 'class_count': 4}
- {'name': 'src.know.diff', 'path': 'src/know/diff.py', 'description': 'Diff mode for tracking architectural changes safely using git worktrees.', 'function_count': 0, 'class_count': 1}
- {'name': 'src.know.knowledge_base', 'path': 'src/know/knowledge_base.py', 'description': 'Knowledge base: cross-session memory for AI agents.\n\nPersists codebase understanding in a project-local SQLite database.\nSupports semantic recall (fastembed) with text-match fallback.\nEach memory is p', 'function_count': 0, 'class_count': 2}
- {'name': 'src.know.index', 'path': 'src/know/index.py', 'description': 'Index management for incremental code scanning.', 'function_count': 1, 'class_count': 1}
- {'name': 'src.know.logger', 'path': 'src/know/logger.py', 'description': 'Logging configuration for know.', 'function_count': 2, 'class_count': 0}
- {'name': 'src.know.watcher', 'path': 'src/know/watcher.py', 'description': 'File system watcher for auto-updating documentation.', 'function_count': 0, 'class_count': 2}
- {'name': 'src.know.models', 'path': 'src/know/models.py', 'description': 'Data models for know-cli.', 'function_count': 0, 'class_count': 4}
- {'name': 'src.know.token_counter', 'path': 'src/know/token_counter.py', 'description': "Token counting for context budget management.\n\nUses tiktoken when available for accurate counting, falls back to\na simple word-based estimation that's fast and dependency-free.\nRoughly approximates GP", 'function_count': 4, 'class_count': 0}

*Set ANTHROPIC_API_KEY for AI-powered guides*