# know

> Living documentation for your codebase. Docs that actually stay current.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**know** is a CLI tool that automatically generates and maintains documentation for your codebase. It uses AST analysis to understand your code structure and AI to generate intelligent summaries that stay in sync as your code evolves.

## ✨ Features

- **🔄 Continuous Sync** - Documentation updates automatically via git hooks
- **🧠 AI-Powered** - Uses Claude/GPT for intelligent code understanding
- **📊 Architecture Diagrams** - Auto-generates C4 and Mermaid diagrams
- **📚 OpenAPI Specs** - Extracts API routes and generates specs
- **🎯 Onboarding Guides** - Creates tailored guides for new team members
- **🤖 LLM-Ready** - Generates AI-optimized codebase digests
- **⚡ Multi-Language** - Supports Python, JavaScript/TypeScript, Go, Rust

## 🚀 Quick Start

```bash
# Install
pip install know-cli

# Initialize in your project
know init

# Generate documentation
know update

# Start watching for changes
know watch

# Explain a specific component
know explain "auth flow"

# Generate API documentation
know api --openapi

# Create onboarding guide
know onboard --for "new devs"

# Generate AI digest
know digest --for-llm
```

## 📖 Commands

### `know init`
Scans your codebase and creates initial documentation structure.

```bash
know init
# Creates:
# - .know/config.yaml
# - README.md (enhanced)
# - docs/architecture.md
# - docs/api.md
```

### `know watch`
Runs in the background, updating docs when files change.

```bash
know watch --daemon
```

### `know explain <component>`
Uses AI to explain specific parts of your codebase.

```bash
know explain "payment processing"
know explain "UserService" --detailed
```

### `know diagram`
Generates architecture diagrams.

```bash
know diagram --architecture  # C4 model
know diagram --components    # Component diagram
know diagram --deps          # Dependency graph
```

### `know api`
Generates API documentation.

```bash
know api --openapi    # OpenAPI 3.0 spec
know api --postman    # Postman collection
know api --markdown   # Markdown docs
```

### `know onboard`
Creates onboarding guides.

```bash
know onboard --for "backend devs"
know onboard --for "new hires" --format pdf
```

### `know digest`
Generates AI-optimized codebase summaries.

```bash
know digest --for-llm     # For feeding to LLMs
know digest --compact     # Compressed summary
know digest --full        # Complete codebase
```

### `know update`
Manually triggers documentation update.

```bash
know update --all
know update --only readme
know update --only diagrams
```

## ⚙️ Configuration

Create `.know/config.yaml` in your project root:

```yaml
project:
  name: "My Project"
  description: "A brief description"
  version: "1.0.0"

# Languages to analyze
languages:
  - python
  - javascript
  - typescript
  - go
  - rust

# Directories to include/exclude
include:
  - "src/"
  - "lib/"
  - "app/"

exclude:
  - "**/node_modules/**"
  - "**/.git/**"
  - "**/tests/**"
  - "**/__pycache__/**"
  - "**/vendor/**"

# AI settings
ai:
  provider: "anthropic"  # or "openai"
  model: "claude-3-sonnet-20240229"
  api_key_env: "ANTHROPIC_API_KEY"
  
  # What to generate
  generate:
    summaries: true
    architecture: true
    api_docs: true
    onboarding: true

# Output settings
output:
  format: "markdown"
  directory: "docs"
  
  # Git integration
  git:
    auto_commit: false
    commit_message: "docs: update generated documentation"
    
  # Watch settings
  watch:
    enabled: true
    debounce_seconds: 5

# Diagram generation
diagrams:
  format: "mermaid"  # or "plantuml", "c4"
  include_dependencies: true
  max_depth: 3

# API documentation
api:
  frameworks:
    - "fastapi"
    - "express"
    - "gin"
  include_schemas: true
  include_examples: true
```

## 🎯 Use Cases

### 1. Team Onboarding
New developers can run `know onboard` to get up to speed quickly.

### 2. AI-Assisted Development
Use `know digest --for-llm` to create codebase summaries for Claude, ChatGPT, or other AI tools.

### 3. Architecture Reviews
Generate C4 diagrams and architecture docs that stay current with your code.

### 4. API Documentation
Auto-generate OpenAPI specs and API docs from your code.

### 5. Knowledge Preservation
Ensure tribal knowledge is captured as documentation that evolves with the codebase.

## 🔧 Git Hooks

Automatically update docs on every commit:

```bash
know hooks install
```

This installs a post-commit hook that runs `know update` after each commit.

## 🏗️ Architecture

```
know/
├── cli.py           # CLI entry point (Click)
├── scanner.py       # AST-based code analysis
├── parser/          # Language-specific parsers
│   ├── python.py
│   ├── javascript.py
│   ├── go.py
│   └── rust.py
├── generator/       # Doc generators
│   ├── markdown.py
│   ├── openapi.py
│   ├── mermaid.py
│   └── c4.py
├── ai.py            # AI integration (Claude/GPT)
├── watcher.py       # File system watcher
├── config.py        # Configuration management
└── git_hooks.py     # Git hook management
```

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

<p align="center">Built with ❤️ by Vic</p>


<!-- KNOW-START -->
# know-cli



*Generated by [know](https://github.com/vic/know-cli)*


## 📁 Project Structure


## 📊 Statistics

- **Files:** 0
- **Modules:** 0
- **Functions:** 0
- **Classes:** 0

---

*This README was generated by [know](https://github.com/vic/know-cli) v1.0.0*

Run `know update` to refresh this documentation.
<!-- KNOW-END -->