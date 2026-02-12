# Codebase Digest

- {'name': 'src.know.index', 'path': 'src/know/index.py', 'description': 'Index management for incremental code scanning.', 'function_count': 1, 'class_count': 1}
- {'name': 'src.know.knowledge_base', 'path': 'src/know/knowledge_base.py', 'description': 'Knowledge base: cross-session memory for AI agents.\n\nPersists codebase understanding in a project-local SQLite database.\nSupports semantic recall (fastembed) with text-match fallback.\nEach memory is p', 'function_count': 0, 'class_count': 2}
- {'name': 'src.know.diff', 'path': 'src/know/diff.py', 'description': 'Diff mode for tracking architectural changes safely using git worktrees.', 'function_count': 0, 'class_count': 1}
- {'name': 'src.know.logger', 'path': 'src/know/logger.py', 'description': 'Logging configuration for know.', 'function_count': 2, 'class_count': 0}
- {'name': 'src.know.models', 'path': 'src/know/models.py', 'description': 'Data models for know-cli.', 'function_count': 0, 'class_count': 4}
- {'name': 'src.know.watcher', 'path': 'src/know/watcher.py', 'description': 'File system watcher for auto-updating documentation.', 'function_count': 0, 'class_count': 2}
- {'name': 'src.know.token_counter', 'path': 'src/know/token_counter.py', 'description': "Token counting for context budget management.\n\nUses tiktoken when available for accurate counting, falls back to\na simple word-based estimation that's fast and dependency-free.\nRoughly approximates GP", 'function_count': 4, 'class_count': 0}
- {'name': 'src.know.ai', 'path': 'src/know/ai.py', 'description': 'AI integration for intelligent code understanding with advanced token optimization.', 'function_count': 0, 'class_count': 4}
- {'name': 'src.know.__init__', 'path': 'src/know/__init__.py', 'description': 'know - Living documentation generator for codebases.', 'function_count': 0, 'class_count': 0}
- {'name': 'src.know.model_router', 'path': 'src/know/model_router.py', 'description': 'Smart model router for cost-optimized AI interactions.\n\nRoutes tasks to the cheapest model that meets quality requirements,\nenabling significant cost savings while maintaining output quality.', 'function_count': 0, 'class_count': 2}
- {'name': 'src.know.stats', 'path': 'src/know/stats.py', 'description': 'Usage statistics tracker for know-cli.\n\nTracks every query, search, and memory operation in a project-local\nSQLite database (`.know/stats.db`).  Powers the `know stats` command.', 'function_count': 0, 'class_count': 2}
- {'name': 'src.know.config', 'path': 'src/know/config.py', 'description': 'Configuration management for know.', 'function_count': 1, 'class_count': 6}
- {'name': 'src.know.semantic_search', 'path': 'src/know/semantic_search.py', 'description': 'Semantic code search using real embeddings and vector similarity.\n\nSupports both file-level (v1) and function-level (v2) embeddings.\nFunction-level uses AST to split Python files into individual chunk', 'function_count': 0, 'class_count': 2}
- {'name': 'src.know.generator', 'path': 'src/know/generator.py', 'description': 'Documentation generator.', 'function_count': 0, 'class_count': 1}
- {'name': 'src.know.exceptions', 'path': 'src/know/exceptions.py', 'description': 'Custom exceptions for know-cli.', 'function_count': 0, 'class_count': 8}
- {'name': 'src.know.import_graph', 'path': 'src/know/import_graph.py', 'description': "Import graph: tracks dependencies between project modules.\n\nStores import relationships as an adjacency list in index.db.\nProvides queries for 'what does X import?' and 'what imports X?'.", 'function_count': 0, 'class_count': 1}
- {'name': 'src.know.git_hooks', 'path': 'src/know/git_hooks.py', 'description': 'Git hook management.', 'function_count': 0, 'class_count': 1}
- {'name': 'src.know.mcp_server', 'path': 'src/know/mcp_server.py', 'description': "MCP (Model Context Protocol) server for know-cli.\n\nExposes know-cli's context engine, search, memory, and graph features\nas MCP tools and resources.  Works with Claude Desktop, Cursor, and any\nMCP-com", 'function_count': 5, 'class_count': 0}
- {'name': 'src.know.parsers', 'path': 'src/know/parsers.py', 'description': 'Parser strategy pattern for language-specific parsing.', 'function_count': 0, 'class_count': 5}
- {'name': 'src.know.scanner', 'path': 'src/know/scanner.py', 'description': 'Codebase scanner for AST analysis.', 'function_count': 1, 'class_count': 1}

Files: 23
Modules: 23

*Set ANTHROPIC_API_KEY for AI digests*