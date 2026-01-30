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

1. **Sign up** at https://console.anthropic.com/
2. **Get your API key** from the dashboard (starts with `sk-ant-`)
3. **Add to your shell profile** (choose based on your shell):

   **For Zsh (macOS default, most modern systems):**
   ```bash
   echo 'export ANTHROPIC_API_KEY="sk-ant-xxxxx-your-key-here"' >> ~/.zshrc
   source ~/.zshrc
   ```

   **For Bash (Linux, older macOS):**
   ```bash
   echo 'export ANTHROPIC_API_KEY="sk-ant-xxxxx-your-key-here"' >> ~/.bashrc
   source ~/.bashrc
   ```

   **For Fish shell:**
   ```bash
   set -Ux ANTHROPIC_API_KEY "sk-ant-xxxxx-your-key-here"
   ```

4. **Verify it works:**
   ```bash
   echo $ANTHROPIC_API_KEY
   # Should output: sk-ant-xxxxx-your-key-here
   ```

**Option B: OpenAI (GPT-4)**

1. Get key from https://platform.openai.com/api-keys (starts with `sk-`)
2. Add to your shell profile:

   **Zsh:**
   ```bash
   echo 'export OPENAI_API_KEY="sk-xxxxx-your-key-here"' >> ~/.zshrc
   source ~/.zshrc
   ```

   **Bash:**
   ```bash
   echo 'export OPENAI_API_KEY="sk-xxxxx-your-key-here"' >> ~/.bashrc
   source ~/.bashrc
   ```

3. Update `.know/config.yaml` to use OpenAI:
   ```yaml
   ai:
     provider: "openai"
     model: "gpt-4"
     api_key_env: "OPENAI_API_KEY"
   ```

**💡 Note:** This is different from Claude Code (the CLI tool) or GitHub Copilot. You need a direct API key from the provider.

### Quick Setup Verification

After adding the API key, verify everything is working:

```bash
# 1. Check the key is set
echo $ANTHROPIC_API_KEY

# 2. Test in a project
cd your-project
know init
know explain -c <some-component>

# 3. You should see AI-generated output (not fallback text)
```

**Troubleshooting:**
- If `echo $ANTHROPIC_API_KEY` shows nothing, the key wasn't saved properly
- Try opening a new terminal window and run the echo command again
- Make sure you used `>>` (append) not `>` (overwrite) when adding to your profile

**💰 Cost Estimation (Aggressive Optimization):**
- **Haiku 4.5** (small/fast tasks): $1/million input, $5/million output
- **Sonnet 4.5** (complex tasks): $3/million input, $15/million output
- **Smart Caching**: 60% reduction for repeated scans
- **Code Compression**: 40% token savings
- **Typical project scan**: ~$0.005-0.02
- **Large codebase**: ~$0.15-0.30
- **Daily watch mode**: ~$0.002-0.01/day

### Token Optimization Features

| Feature | Savings | How |
|---------|---------|-----|
| **Response Caching** | ~60% | Caches AI responses in SQLite |
| **Code Compression** | ~40% | Removes comments/whitespace |
| **Smart Truncation** | ~30% | Extracts signatures vs full code |
| **Model Selection** | ~50% | Haiku for simple tasks |
| **Ultra-Short Prompts** | ~20% | Minimal prompt templates |
| **Content Hashing** | ~60% | Skips identical content |

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
# Create AI-powered documentation (generates docs/arc.md by default)
know update

# Generate specific docs only
know update --only system      # docs/arc.md
know update --only diagrams    # docs/architecture.md
know update --only api         # docs/openapi.json

# Generate LLM-optimized digest for AI agents
know digest --for-llm

# Start auto-updating on file changes
know watch
```

**Note:** By default, `know` generates documentation in the `docs/` folder:
- `docs/arc.md` - System overview and project structure
- `docs/architecture.md` - Architecture diagrams
- `docs/onboarding-*.md` - Onboarding guides
- `docs/digest-llm.md` - AI-optimized codebase summary

This keeps generated docs separate from your README.md. If you want to update README.md instead, you can customize the output path in `.know/config.yaml` (see Configuration section).

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
  model: "claude-haiku-4-5-20251022"  # or claude-sonnet-4-5-20251022
  api_key_env: "ANTHROPIC_API_KEY"

output:
  directory: "docs"
  
  # Optional: Customize the main system doc filename
  # Default is "arc.md" in the docs/ folder
  # Set to "README.md" to update your project's README instead
  # system_doc: "arc.md"
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

### 🔄 `know watch` vs Git Hooks: What's the Difference?

Both keep your docs in sync, but work differently:

| Feature | `know watch` | `know hooks install` |
|---------|--------------|----------------------|
| **When it runs** | While you're coding | When you commit code |
| **Process** | Background daemon | Pre-commit hook |
| **Scope** | All file changes | Only staged changes |
| **Best for** | Active development | CI/CD, team workflows |
| **Setup** | Run manually | One-time install |
| **Resource use** | Continuous (light) | Only on commit |

**Use `know watch` when:**
- You're actively developing and want docs updated in real-time
- Working on complex refactors
- Want to see immediate feedback

**Use Git Hooks when:**
- You want docs committed alongside code
- Working in a team (ensures docs are always committed)
- Want CI/CD to have updated docs
- Don't want a background process running

**Can use both together:**
```bash
# Install git hooks for team workflow
know hooks install

# Also run watch during active development
know watch
```

### 🤖 GitHub Action (CI/CD)

For teams using GitHub, there's an official GitHub Action that auto-generates docs on every PR:

**Quick setup:**

1. Add `ANTHROPIC_API_KEY` to your repository secrets (Settings → Secrets)

2. Create `.github/workflows/docs.yml`:
```yaml
name: Documentation
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  docs:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
    steps:
      - uses: actions/checkout@v4
      - uses: sushilk1991/know-cli/.github/actions/know-cli@main
        with:
          api-key: ${{ secrets.ANTHROPIC_API_KEY }}
          comment-on-pr: 'true'
```

**What it does:**
- Generates docs on every PR
- Posts architecture summary as PR comment
- Auto-commits updated docs
- Shows structural changes vs base branch

**Perfect for:**
- Team workflows
- Code reviews (reviewers see architecture context)
- Keeping docs always in sync

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

**know** generates AI-optimized documentation that helps coding agents understand your codebase instantly.

### What Gets Generated

After running `know init` and `know update`, you'll have:

```
docs/
├── arc.md              # Project overview and structure
├── architecture.md     # C4 architecture diagrams
├── digest-llm.md       # AI-optimized codebase summary
└── onboarding-*.md     # Team onboarding guides
```

### Using with Claude Code

**Quick start:**
```bash
# In your project
know init
know digest --for-llm
```

**Then in Claude Code:**
```
Read docs/digest-llm.md to understand this codebase, then help me add a new feature.
```

**Or give Claude a complete brief:**
```markdown
This project uses know-cli for documentation.

Context files:
- docs/digest-llm.md - Full codebase overview (read this first)
- docs/arc.md - Project structure and modules
- docs/architecture.md - System design and patterns

Use "know explain -c <component>" to understand any specific module.
```

### Using with Codex (GitHub Copilot)

Create `.github/copilot-context.md`:

```markdown
# Copilot Context

## Quick Reference
Read docs/digest-llm.md for full codebase context.

## Project Structure
See docs/arc.md for module organization.

## Architecture
Review docs/architecture.md before making structural changes.

## Onboarding
New to this codebase? Read docs/onboarding-developers.md
```

### Using with Cursor

Add to `.cursorrules`:

```markdown
# Cursor Rules

## Before Writing Code
1. Read docs/digest-llm.md for context
2. Check docs/architecture.md for design patterns

## Understanding Components
Use "know explain -c <component-name>" for detailed explanations.

## Keeping Docs Updated
Run "know update" after major changes to keep docs in sync.
```

### Quick Workflow for AI Agents

```bash
# 1. One-time setup
cd your-project
know init
know digest --for-llm

# 2. Tell your AI agent:
# "Read docs/digest-llm.md for context"

# 3. For specific questions:
know explain -c <component-name>
# "Check docs/arc.md for the component explanation"

# 4. Keep docs fresh (optional)
know hooks install  # Auto-update on commit
# or
know watch          # Real-time updates while coding
```

### Pro Tips

1. **Commit docs/** to git so AI agents can read them
2. Run `know digest --for-llm` before asking agents for help
3. Use `know explain -c <name>` for detailed component understanding
4. Install git hooks (`know hooks install`) to keep docs in sync automatically

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

<p align="center">Built by <a href="https://github.com/sushilk1991">@sushilk1991</a></p>
