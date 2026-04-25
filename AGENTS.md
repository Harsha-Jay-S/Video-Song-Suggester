# AGENTS.md — General Session Startup

## When I start a new project session

### Step 1: Scan before asking
List the root directory. Read the README if it exists. Look for the main entry point. Understand what the project does before answering anything.

### Step 2: Compile check
After any file change, run `python3 -m py_compile <file>` (or the language equivalent) before running anything. Fix all syntax errors first.

### Step 3: Test with real files
When given a test path, run the actual pipeline/command. Don't assume it works. Show real output.

### Step 4: Be concise
Direct answers. 1-3 sentences for simple questions. Show real command output when running tests. No intro/outro paragraphs.

---

## Project-specific guidance (replace this section per project)

*Insert here after scanning the project. Examples:*
- Exact run/test/lint commands
- Architecture notes not obvious from file names
- Important quirks (e.g. "CLIP misclassifies screen recordings")
- Setup requirements or environment gotchas
- Library boundaries or entrypoints

---

*Keep this file lightweight. Project-specific details go in the section above, not up here.*