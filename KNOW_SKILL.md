# know-cli Integration

This project uses [know-cli](https://github.com/vic/know-cli) for intelligent context management. It gives you exactly the code you need — no more, no less.

## Before starting a task

Get smart, token-budgeted context:

```bash
know context "<task description>" --budget 8000 --quiet
```

This returns the most relevant code, imports, tests, and project knowledge — all packed within your token budget.

**Examples:**
```bash
know context "fix the authentication middleware" --budget 8000 --quiet
know context "add pagination to the users API" --budget 4000 --json
know context "refactor the database connection pool" --budget 6000 --quiet
```

## When you learn something about the codebase

Store it as a memory so future sessions benefit:

```bash
know remember "<insight>"
```

**Examples:**
```bash
know remember "The auth system uses JWT tokens stored in Redis"
know remember "Never modify migration files directly — use the migration CLI"
know remember "The config system falls back to env vars when YAML is missing"
```

## To search for specific code

Semantic search — understands meaning, not just keywords:

```bash
know search "<query>" --json
```

**Examples:**
```bash
know search "error handling" --json
know search "database connection" --chunk --json
```

## To understand dependencies

See what imports what:

```bash
know graph <file_path>
```

**Examples:**
```bash
know graph src/auth/middleware.py
know graph src/api/routes.py
```

## To recall stored knowledge

```bash
know recall "<query>"
```

**Examples:**
```bash
know recall "how does auth work?"
know recall "database patterns"
```

## Tips

- Use `--budget` to control context size (default 8000 tokens)
- Use `--json` for machine-readable output
- Use `--quiet` to suppress decorative output
- Use `--time` to see execution timing
- Memories are automatically included in `know context` results
- Run `know reindex` after major refactors to update the search index
