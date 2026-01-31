# Onboarding Guide for New Developers

## Overview

Know is a documentation generation and knowledge management system that automatically scans codebases to create searchable documentation. The project uses AI to analyze code structure, generate summaries, and provide semantic search capabilities across your codebase.

## Directory Structure

```
src/know/
├── __init__.py          # Package initialization
├── cli.py               # Command-line interface entry point
├── config.py            # Configuration management and settings
├── models.py            # Core data models (CodeEntity, Documentation)
├── index.py             # Indexing and database operations
├── scanner.py           # Code scanning and file traversal
├── parsers/             # Language-specific code parsers
├── ai.py                # AI integration for code analysis
├── semantic_search.py   # Vector-based search functionality
├── generator.py         # Documentation generation logic
├── watcher.py           # File system monitoring for changes
├── quality.py           # Code quality checks and metrics
├── git_hooks/           # Git integration hooks
├── logger.py            # Logging utilities
└── exceptions.py        # Custom exception classes
```

**Key files to understand:**
- `models.py` defines the data structures representing code entities and documentation
- `config.py` manages all configuration options, environment variables, and default settings

## Getting Started

### Step 1: Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd know

# Install dependencies
pip install -e ".[dev]"

# Set up environment variables (see config.py for options)
export OPENAI_API_KEY="your-api-key"
```

### Step 2: Understand the Flow
The typical execution flow is: **CLI → Scanner → Parsers → AI Analysis → Index → Generator**

1. CLI receives commands
2. Scanner traverses the codebase
3. Parsers extract code entities
4. AI generates summaries and embeddings
5. Index stores data
6. Generator creates documentation

### Step 3: Run Your First Command
```bash
# Initialize documentation for a project
know init /path/to/project

# Scan and update documentation
know update

# Search the knowledge base
know search "function name"
```

## Common Commands

```bash
# Initialize a new project
know init <project-path>

# Update documentation after code changes
know update

# Search codebase semantically
know search "query text"

# Watch for file changes (auto-update)
know watch

# Generate documentation site
know generate

# Run quality checks
know quality check

# View configuration
know config show
```

For debugging, check logs in `.know/logs/`. Start with `scanner.py` and `cli.py` to understand the entry points, then explore `models.py` to grasp the core data structures.