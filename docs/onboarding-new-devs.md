# Onboarding: new team members

## Structure
- {'name': 'src.know.logger', 'path': 'src/know/logger.py', 'description': 'Logging configuration for know.', 'function_count': 2, 'class_count': 0}
- {'name': 'src.know.token_counter', 'path': 'src/know/token_counter.py', 'description': "Token counting for context budget management.\n\nUses tiktoken (cl100k_base) for accurate counting. Applies a calibration\nfactor for Anthropic models since Claude uses a different tokenizer than\nOpenAI'", 'function_count': 5, 'class_count': 0}
- {'name': 'src.know.embeddings', 'path': 'src/know/embeddings.py', 'description': 'Centralized embedding model management.\n\nSingle process-wide cache for fastembed models. All consumers\n(context_engine, semantic_search, knowledge_base) use this instead\nof maintaining separate caches', 'function_count': 4, 'class_count': 0}
- {'name': 'src.know.semantic_search', 'path': 'src/know/semantic_search.py', 'description': 'Semantic code search using real embeddings and vector similarity.\n\nSupports both file-level (v1) and function-level (v2) embeddings.\nFunction-level uses AST to split Python files into individual chunk', 'function_count': 0, 'class_count': 2}
- {'name': 'src.know.ai', 'path': 'src/know/ai.py', 'description': 'AI integration for intelligent code understanding with advanced token optimization.', 'function_count': 0, 'class_count': 4}
- {'name': 'src.know.file_categories', 'path': 'src/know/file_categories.py', 'description': 'File category detection and demotion for search ranking.\n\nClassifies files as source, test, vendor, or generated.\nApplies score multipliers so non-source files rank lower.', 'function_count': 7, 'class_count': 0}
- {'name': 'src.know.import_graph', 'path': 'src/know/import_graph.py', 'description': "Import graph: tracks dependencies between project modules.\n\nDelegates storage to DaemonDB (the single source of truth).\nProvides queries for 'what does X import?' and 'what imports X?'.\n\nUses fully-qu", 'function_count': 6, 'class_count': 1}
- {'name': 'src.know.watcher', 'path': 'src/know/watcher.py', 'description': 'File system watcher for auto-updating documentation.', 'function_count': 0, 'class_count': 2}

*Set ANTHROPIC_API_KEY for AI-powered guides*