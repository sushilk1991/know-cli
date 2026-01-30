# AGENTS.md - Development Notes

## Critical Learnings

### Anthropic API Models
**ALWAYS** verify model names against the actual API before using them.

**How to check available models:**
```bash
curl -s https://api.anthropic.com/v1/models \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" | \
  python3 -c "import sys,json; d=json.load(sys.stdin); [print(m['id']) for m in d.get('data', [])]"
```

**Current valid models:**
- `claude-sonnet-4-5-20250929` (Sonnet 4.5 - use as default)
- `claude-haiku-4-5-20251001` (Haiku 4.5 - fast/cheap)
- `claude-opus-4-5-20251101` (Opus 4.5 - most capable)

**NEVER invent model names.** Anthropic uses dated versions (YYYYMMDD), not semantic versioning. Names like `claude-sonnet-4-5-20251022` don't exist and will cause 404 errors.

### Testing Checklist
Before shipping AI-related features:
1. ✅ Query actual API for valid model IDs
2. ✅ Test the API call with the model ID
3. ✅ Verify response parsing works correctly
4. ✅ Test both success and error cases

### Response Parsing
Anthropic returns content as a list of `TextBlock` objects:
```python
response = message.content[0].text  # Correct
# NOT: response = message.content[0]["text"]  # Wrong
```

### Version History
- v0.1.22: Fixed model names to use valid IDs (claude-sonnet-4-5-20250929)
- v0.1.15-0.1.21: Various fixes for AI integration bugs
