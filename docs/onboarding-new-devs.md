# Onboarding: new team members

## Structure
- {'name': 'src.know.logger', 'path': 'src/know/logger.py', 'description': 'Logging configuration for know.', 'function_count': 2, 'class_count': 0}
- {'name': 'src.know.token_counter', 'path': 'src/know/token_counter.py', 'description': "Token counting for context budget management.\n\nUses a simple word-based estimation that's fast and dependency-free.\nRoughly approximates GPT/Claude tokenization (~1.3 tokens per word).", 'function_count': 3, 'class_count': 0}
- {'name': 'src.know.semantic_search', 'path': 'src/know/semantic_search.py', 'description': 'Semantic code search using real embeddings and vector similarity.\n\nSupports both file-level (v1) and function-level (v2) embeddings.\nFunction-level uses AST to split Python files into individual chunk', 'function_count': 0, 'class_count': 2}
- {'name': 'src.know.ai', 'path': 'src/know/ai.py', 'description': 'AI integration for intelligent code understanding with advanced token optimization.', 'function_count': 0, 'class_count': 4}
- {'name': 'src.know.import_graph', 'path': 'src/know/import_graph.py', 'description': "Import graph: tracks dependencies between project modules.\n\nStores import relationships as an adjacency list in index.db.\nProvides queries for 'what does X import?' and 'what imports X?'.", 'function_count': 0, 'class_count': 1}
- {'name': 'src.know.watcher', 'path': 'src/know/watcher.py', 'description': 'File system watcher for auto-updating documentation.', 'function_count': 0, 'class_count': 2}
- {'name': 'src.know.cli', 'path': 'src/know/cli.py', 'description': 'CLI entry point for know.', 'function_count': 33, 'class_count': 0}
- {'name': 'src.know.stats', 'path': 'src/know/stats.py', 'description': 'Usage statistics tracker for know-cli.\n\nTracks every query, search, and memory operation in a project-local\nSQLite database (`.know/stats.db`).  Powers the `know stats` command.', 'function_count': 0, 'class_count': 1}

*Set ANTHROPIC_API_KEY for AI-powered guides*