# Benchmark Task: Farfield Codebase Investigation

## Task

You are investigating the farfield codebase at `/Users/sushil/Code/Github/farfield`.

Answer these 3 questions with specific file paths, function names, and code evidence:

### Question 1: How does the billing/subscription system work?
- What model represents a subscription?
- What are the billing plan tiers and their limits?
- How are sandbox limits enforced?
- Name the specific files, classes, and functions involved.

### Question 2: How does the LLM provider system work?
- How are LLM providers (OpenAI, Anthropic, Gemini etc.) configured and selected?
- What is the `provider_discovery` module doing?
- How are workspace-level model settings handled?
- Name the specific files, classes, and functions involved.

### Question 3: How does the agent execution pipeline work?
- What is the flow from a user message to an agent response?
- What role does LangGraph play?
- How are tools/MCP servers integrated?
- Name the specific files, classes, and functions involved.

## Output Format

Write your findings to a file. Include:
- Specific file paths and line numbers
- Function/class names
- Brief description of what each component does
- How components connect to each other

Be thorough but concise. Focus on accuracy over completeness.
