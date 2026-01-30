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

### 1. Get an AI API Key (Required for AI Features)

**know** uses AI to generate intelligent documentation. You need an API key from one of these providers:

**Option A: Anthropic (Claude)** ⭐ Recommended
- Sign up at https://console.anthropic.com/
- Get your API key from the dashboard
- Set environment variable: `export ANTHROPIC_API_KEY="your-key-here"`
- Add to your `~/.zshrc` or `~/.bashrc` to make it permanent

**Option B: OpenAI (GPT-4)**
- Get key from https://platform.openai.com/api-keys
- Set: `export OPENAI_API_KEY="your-key-here"`
- Update `.know/config.yaml` to use OpenAI

**💡 Note:** This is different from Claude Code (the CLI tool) or GitHub Copilot. You need a direct API key from the provider.

### 2. Install know

```bash
pip install know-cli

# Or with pipx
pipx install know-cli
```

### 3. Initialize Your Project

```bash
cd your-project
know init

# Follow the prompts to configure
```

### 4. Generate Documentation

```bash
# Create AI-powered documentation
know update

# Generate LLM-optimized digest for AI agents
know digest --for-llm

# Start auto-updating on file changes
know watch
```

### 5. Verify Setup

```bash
# Check if AI is working
know explain -c <component-name>

# Should see AI-generated explanation (not fallback)
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

## 🔧 Troubleshooting

### "ANTHROPIC_API_KEY not set" Error

**Problem:** You're seeing: `⚠ ANTHROPIC_API_KEY not set. AI features will be limited.`

**Solution:**
1. Get an API key from https://console.anthropic.com/
2. Set the environment variable:
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-xxxxx"
   ```
3. Make it permanent by adding to `~/.zshrc` (macOS) or `~/.bashrc` (Linux)

**💡 Important:** This is different from:
- ❌ Claude Code CLI (doesn't provide API keys)
- ❌ GitHub Copilot (uses different authentication)
- ❌ Cursor editor (has its own key management)

You need a direct API key from Anthropic's console.

### "Cannot install tree-sitter-languages" Error

**Problem:** Installation fails with `tree-sitter-languages` errors.

**Cause:** You're using Python 3.13+ which doesn't have pre-built wheels.

**Solution:**
```bash
# Option 1: Use Python 3.10-3.12
pyenv install 3.12.1
pyenv global 3.12.1
pip install know-cli

# Option 2: Install without parser extras (regex fallback)
pip install know-cli
# Tree-sitter features will be disabled, regex-based parsing still works
```

### Syntax Errors When Parsing Files

**Problem:** Seeing `invalid syntax` errors for TypeScript/React files.

**Cause:** Tree-sitter parser isn't installed or the file has syntax errors.

**Solutions:**
1. Install with parser support: `pip install "know-cli[parser]"`
2. Check the file actually compiles (run `tsc --noEmit`)
3. Add problematic files to `.know/config.yaml` exclude list

### AI Explanations Not Working

**Problem:** Getting fallback explanations instead of AI-generated ones.

**Check:**
```bash
# Verify API key is set
echo $ANTHROPIC_API_KEY

# Should output your key (starts with sk-ant-)
```

**If not set:**
```bash
# Add to your shell profile
echo 'export ANTHROPIC_API_KEY="your-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### "Module not found" Errors

**Problem:** `ModuleNotFoundError` when running know.

**Solution:**
```bash
# Reinstall in clean environment
pip uninstall know-cli
pip install --upgrade know-cli

# Or use pipx for isolation
pipx install know-cli
```

## 🤖 Using with AI Agents (Claude Code, Codex, Cursor)

**know** is designed specifically to make AI coding agents more effective. Here's how to integrate it:

### Why AI Agents Love know

- **📚 Complete Context**: Instead of showing agents partial files, they get the full codebase digest
- **🏗️ Architecture Understanding**: Agents understand the big picture before diving into code
- **⚡ Faster Onboarding**: New agents understand your codebase in seconds, not hours
- **🔄 Always Current**: Documentation syncs with code, so agents work with up-to-date info

### Setup for AI Agents

#### 1. Initialize know in Your Project

```bash
cd your-project
know init
```

This creates:
- `.know/config.yaml` - Project configuration
- `docs/` - Documentation directory
- Index of your codebase for fast lookups

#### 2. Generate AI-Optimized Digest

```bash
# Create LLM-friendly digest
know digest --for-llm

# Or create component-specific docs
know explain -c AuthService
know diagram --type architecture
```

#### 3. Configure Git Hooks (Auto-Update)

```bash
# Docs update automatically on every commit
know hooks install
```

### Using with Claude Code

Add this to your Claude Code context:

```markdown
## Project Context

This project uses **know-cli** for documentation. 

**Quick commands:**
- Read `docs/digest-llm.md` for full codebase overview
- Check `docs/architecture.md` for system design
- Use `know explain -c <component>` to understand specific modules

**Current state:**
- Project: [Read docs/project-summary.md]
- API: [Read docs/openapi.json if exists]
- Components: [List from docs/ directory]
```

**In Claude Code, ask:**
```
"Read the digest-llm.md file to understand this codebase, then help me..."
```

### Using with Codex (GitHub Copilot)

Create `.github/copilot-context.md`:

```markdown
# Copilot Context

## Project Overview
[Content from know digest --for-llm]

## Key Components
- AuthService: Handles authentication [docs/auth-service.md]
- API Layer: REST endpoints [docs/openapi.json]
- Database: Schema [docs/database-schema.md]

## Architecture
[Link to docs/architecture.md]

## Code Patterns
- Follow patterns in docs/code-patterns.md
- Check similar implementations using know explain
```

### Using with Cursor

Add to `.cursorrules`:

```markdown
# Cursor Rules

## Before Coding
1. Read `docs/digest-llm.md` for context
2. Check `docs/architecture.md` for design patterns
3. Use `know explain -c <component>` to understand existing code

## Code Standards
- Follow patterns documented in docs/code-patterns.md
- Maintain architecture decisions from docs/adr/
- Update know docs if adding new components

## Helpful Commands
- `know update` - Refresh docs before major changes
- `know explain -c <name>` - Understand any component
```

### Best Practices for AI Agents

#### 1. **Always Start with Digest**

Before asking an AI agent to work on your code:

```bash
# Ensure digest is current
know digest --for-llm
```

Then tell the agent:
```
"Please read docs/digest-llm.md first to understand the codebase structure."
```

#### 2. **Use Component Explanations**

For specific tasks, generate targeted explanations:

```bash
# Before modifying auth system
know explain -c Authentication
# Creates: docs/component-authentication.md

# Give this file to the AI agent
```

#### 3. **Keep Docs in Sync**

With git hooks installed, docs auto-update. But for long sessions:

```bash
# Refresh mid-session if needed
know update
```

#### 4. **Include Architecture Context**

```bash
# Generate architecture docs
know diagram --type architecture
know diagram --type component
```

AI agents work better when they see the visual structure.

### Example Workflow

```bash
# 1. Initialize project
cd my-project
know init

# 2. Generate all AI docs
know digest --for-llm
know diagram --type architecture
know api --openapi

# 3. Install git hooks (auto-update)
know hooks install

# 4. Commit docs
git add docs/ .know/
git commit -m "docs: Add know-cli documentation for AI agents"

# 5. Now AI agents can:
# - Read docs/digest-llm.md for full context
# - Reference docs/architecture.md for design
# - Check docs/openapi.json for API details
```

### Pro Tips

1. **Include docs/ in git**: AI agents need to read these files
2. **Keep ANTHROPIC_API_KEY set**: For AI-powered explanations
3. **Use `--for-llm` flag**: Optimizes output for AI consumption
4. **Generate before PRs**: Run `know update` before asking agents to review

### Troubleshooting

**Agent doesn't understand the codebase?**
```bash
know digest --for-llm --refresh
# This regenerates with latest code
```

**Need specific component context?**
```bash
know explain -c ComponentName --detail high
```

**Architecture unclear?**
```bash
know diagram --type architecture --format mermaid
# Include the diagram in your AI context
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

<p align="center">Built by <a href="https://github.com/sushilk1991">@sushilk1991</a></p>
