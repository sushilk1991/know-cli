# Codebase Digest

- {'name': 'src.know.diff', 'path': 'src/know/diff.py', 'description': 'Diff mode for tracking architectural changes safely using git worktrees.', 'function_count': 0, 'class_count': 1}
- {'name': 'src.know.knowledge_base', 'path': 'src/know/knowledge_base.py', 'description': 'Knowledge base: cross-session memory for AI agents.\n\nThin wrapper around DaemonDB — the single source of truth for all\nproject data. Translates between integer display IDs (used by CLI)\nand text UUIDs', 'function_count': 0, 'class_count': 2}
- {'name': 'src.know.daemon', 'path': 'src/know/daemon.py', 'description': 'Background daemon for maintaining hot indexes.\n\nLifecycle:\n- Auto-started on first CLI call (no explicit `know start` needed)\n- Listens on Unix socket at ~/.know/sockets/<project-hash>.sock\n- PID file', 'function_count': 9, 'class_count': 2}
- {'name': 'src.know.ai', 'path': 'src/know/ai.py', 'description': 'AI integration for intelligent code understanding with advanced token optimization.', 'function_count': 0, 'class_count': 4}
- {'name': 'src.know.logger', 'path': 'src/know/logger.py', 'description': 'Logging configuration for know.', 'function_count': 2, 'class_count': 0}
- {'name': 'src.know.embeddings', 'path': 'src/know/embeddings.py', 'description': 'Centralized embedding model management.\n\nSingle process-wide cache for fastembed models. All consumers\n(context_engine, semantic_search, knowledge_base) use this instead\nof maintaining separate caches', 'function_count': 3, 'class_count': 0}
- {'name': 'src.know.models', 'path': 'src/know/models.py', 'description': 'Data models for know-cli.', 'function_count': 0, 'class_count': 4}
- {'name': 'src.know.watcher', 'path': 'src/know/watcher.py', 'description': 'File system watcher for auto-updating documentation.', 'function_count': 0, 'class_count': 2}
- {'name': 'src.know.token_counter', 'path': 'src/know/token_counter.py', 'description': "Token counting for context budget management.\n\nUses tiktoken (cl100k_base) for accurate counting. Applies a calibration\nfactor for Anthropic models since Claude uses a different tokenizer than\nOpenAI'", 'function_count': 4, 'class_count': 0}
- {'name': 'src.know.__init__', 'path': 'src/know/__init__.py', 'description': 'know - Living documentation generator for codebases.', 'function_count': 0, 'class_count': 0}
- {'name': 'src.know.index', 'path': 'src/know/index.py', 'description': 'Index management for incremental code scanning.', 'function_count': 1, 'class_count': 1}
- {'name': 'src.know.stats', 'path': 'src/know/stats.py', 'description': 'Usage statistics tracker for know-cli.\n\nTracks every query, search, and memory operation in a project-local\nSQLite database (`.know/stats.db`).  Powers the `know stats` command.', 'function_count': 0, 'class_count': 1}
- {'name': 'src.know.config', 'path': 'src/know/config.py', 'description': 'Configuration management for know.', 'function_count': 1, 'class_count': 6}
- {'name': 'src.know.import_graph', 'path': 'src/know/import_graph.py', 'description': "Import graph: tracks dependencies between project modules.\n\nDelegates storage to DaemonDB (the single source of truth).\nProvides queries for 'what does X import?' and 'what imports X?'.\n\nUses fully-qu", 'function_count': 0, 'class_count': 1}
- {'name': 'src.know.exceptions', 'path': 'src/know/exceptions.py', 'description': 'Custom exceptions for know-cli.', 'function_count': 0, 'class_count': 8}
- {'name': 'src.know.git_hooks', 'path': 'src/know/git_hooks.py', 'description': 'Git hook management.', 'function_count': 0, 'class_count': 1}
- {'name': 'src.know.generator', 'path': 'src/know/generator.py', 'description': 'Documentation generator.', 'function_count': 0, 'class_count': 1}
- {'name': 'src.know.semantic_search', 'path': 'src/know/semantic_search.py', 'description': 'Semantic code search using real embeddings and vector similarity.\n\nSupports both file-level (v1) and function-level (v2) embeddings.\nFunction-level uses AST to split Python files into individual chunk', 'function_count': 0, 'class_count': 2}
- {'name': 'src.know.mcp_server', 'path': 'src/know/mcp_server.py', 'description': "MCP (Model Context Protocol) server for know-cli.\n\nExposes know-cli's context engine, search, memory, and graph features\nas MCP tools and resources.  Works with Claude Desktop, Cursor, and any\nMCP-com", 'function_count': 5, 'class_count': 0}
- {'name': 'src.know.daemon_db', 'path': 'src/know/daemon_db.py', 'description': 'Unified daemon database — SQLite with FTS5 for search.\n\nStores code chunks, import graph, and cross-session memories in a\nsingle database. FTS5 provides BM25 keyword search as the default;\nvector embe', 'function_count': 0, 'class_count': 1}

Files: 33
Modules: 33

*Set ANTHROPIC_API_KEY for AI digests*