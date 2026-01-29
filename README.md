# know

> Living documentation for your codebase. Docs that actually stay current.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/know-cli.svg)](https://pypi.org/project/know-cli/)

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
cd your-project
know init

# Generate documentation
know update

# Start watching for changes
know watch
```

## 📖 Commands

| Command | Description |
|---------|-------------|
| `know init` | Scan codebase, create config, generate initial docs |
| `know watch` | Background process that updates docs on file changes |
| `know explain -c <name>` | AI explains a specific component |
| `know diagram --type architecture` | Generate architecture diagrams |
| `know api --openapi` | Generate OpenAPI spec from API routes |
| `know onboard --for "new devs"` | Create onboarding guide |
| `know digest --for-llm` | Generate AI-optimized codebase summary |
| `know update` | Manually trigger documentation update |
| `know hooks install` | Install git hooks for auto-update |

## 🎯 Use Cases

### 1. **Team Onboarding**
```bash
know onboard --for "backend devs"
# Creates: docs/onboarding-backend-devs.md
```

### 2. **AI-Assisted Development**
```bash
know digest --for-llm
# Creates: docs/digest-llm.md - optimized for Claude/ChatGPT
```

### 3. **Architecture Documentation**
```bash
know diagram --type architecture
# Creates: docs/architecture.md with Mermaid diagrams
```

### 4. **API Documentation**
```bash
know api --openapi
# Creates: docs/openapi.json from your API routes
```

## ⚙️ Configuration

Create `.know/config.yaml`:

```yaml
project:
  name: "My Project"
  description: "A brief description"
  version: "1.0.0"

languages:
  - python
  - typescript
  - go

include:
  - "src/"
  - "packages/"
  - "apps/"

exclude:
  - "**/node_modules/**"
  - "**/.git/**"
  - "**/tests/**"

ai:
  provider: "anthropic"
  model: "claude-3-sonnet-20240229"
  api_key_env: "ANTHROPIC_API_KEY"

output:
  directory: "docs"
```

## 🔧 Installation

```bash
# From PyPI (when published)
pip install know-cli

# From source
git clone https://github.com/sushilk1991/know-cli.git
cd know-cli
pip install -e .
```

## 🏗️ Supported Languages

| Language | Parser | Status |
|----------|--------|--------|
| Python | AST (built-in) | ✅ Full support |
| JavaScript/TypeScript | Regex-based | ✅ Supported |
| Go | Regex-based | ✅ Supported |
| Rust | Planned | 🚧 Coming soon |

## 🔗 Git Hooks

Automatically update docs on every commit:

```bash
know hooks install
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

<p align="center">Built by <a href="https://github.com/sushilk1991">@sushilk1991</a></p>
