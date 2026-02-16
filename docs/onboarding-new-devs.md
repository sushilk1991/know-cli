# Onboarding: new team members

## Structure
- {'name': 'src.know.daemon', 'path': 'src/know/daemon.py', 'description': 'Background daemon for maintaining hot indexes.\n\nLifecycle:\n- Auto-started on first CLI call (no explicit `know start` needed)\n- Listens on Unix socket at ~/.know/sockets/<project-hash>.sock\n- PID file', 'function_count': 15, 'class_count': 2}
- {'name': 'src.know.ai', 'path': 'src/know/ai.py', 'description': 'AI integration for intelligent code understanding with advanced token optimization.', 'function_count': 0, 'class_count': 4}
- {'name': 'src.know.diff', 'path': 'src/know/diff.py', 'description': 'Diff mode for tracking architectural changes safely using git worktrees.', 'function_count': 0, 'class_count': 1}
- {'name': 'src.know.knowledge_base', 'path': 'src/know/knowledge_base.py', 'description': 'Knowledge base: cross-session memory for AI agents.\n\nThin wrapper around DaemonDB — the single source of truth for all\nproject data. Translates between integer display IDs (used by CLI)\nand text UUIDs', 'function_count': 0, 'class_count': 2}
- {'name': 'src.know.index', 'path': 'src/know/index.py', 'description': 'Index management for incremental code scanning.', 'function_count': 1, 'class_count': 1}
- {'name': 'src.know.logger', 'path': 'src/know/logger.py', 'description': 'Logging configuration for know.', 'function_count': 2, 'class_count': 0}
- {'name': 'src.know.watcher', 'path': 'src/know/watcher.py', 'description': 'File system watcher for auto-updating documentation.', 'function_count': 0, 'class_count': 2}
- {'name': 'src.know.embeddings', 'path': 'src/know/embeddings.py', 'description': 'Centralized embedding model management.\n\nSingle process-wide cache for fastembed models. All consumers\n(context_engine, semantic_search, knowledge_base) use this instead\nof maintaining separate caches', 'function_count': 4, 'class_count': 0}

*Set ANTHROPIC_API_KEY for AI-powered guides*