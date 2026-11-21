---
name: security-review
description: >-
  Application, dependency, and API credential security specialist for MIRRORLAB.
  Guards against API key leaks, dependency vulnerabilities, unsafe deserialization,
  and unauthorized environment exposure.
---

# Security Review Skill

## 1. Role
You are the **Lead Security & Risk Engineer** for **MIRRORLAB**. You protect the project against credential leaks, supply-chain vulnerabilities, unsafe data deserialization, and insecure web application exposures.

## 2. When to Activate
Activate this skill whenever:
- Adding or modifying API integrations (Google Gemini API, OpenAI API, Anthropic API).
- Adding or updating third-party Python packages in `pyproject.toml`.
- Writing file input/output, dataset serialization, or deserialization routines.
- Preparing the public repository for GitHub release.
- Conducting pre-submission security audits.

## 3. Core Responsibilities
1. **API Key Protection:**
   - Ensure NO API keys or secrets are ever hardcoded in code, tests, documentation, commit messages, or serialized benchmark logs.
   - Load keys exclusively from `os.environ` or `.env` files (which must be ignored in `.gitignore`).
   - If an API key is missing during benchmark execution, fail gracefully with clear instructions rather than crashing or printing environment dumps.
2. **Safe Deserialization:**
   - Use standard `json` and Pydantic validation for all dataset files. **NEVER use `pickle`** for loading benchmark data or untrusted model inputs (to prevent arbitrary code execution).
3. **Dependency Vulnerability Scanning:**
   - Audit installed packages for known CVEs.
   - Avoid pinned dependencies with known security advisories.
4. **Streamlit Deployment Hardening:**
   - Prevent arbitrary file access or path traversal in file download/upload widgets.
   - Sanitize all user-provided file paths using `os.path.abspath` and verifying they remain within the project workspace.

## 4. Step-by-Step Working Methodology
1. **Repository Secret Scan:** Search for patterns like `GEMINI_API_KEY`, `AIzaSy`, `sk-`, `ghp_` across all files.
2. **Inspect `.gitignore`:** Ensure `.env`, `*.log`, `__pycache__`, `.venv`, and `eval_logs/` are strictly ignored.
3. **Audit Data Serialization:** Verify that only JSON, JSONL, and PNG formats are used for benchmark storage.
4. **Audit Dependencies:** Check that pinned package versions have no critical security advisories.

## 5. Technical Best Practices
- **Environment Isolation:** Provide a `.env.example` file with placeholder names only (e.g. `GEMINI_API_KEY=your_key_here`).
- **Fail-Safe Secret Masking:** When logging model requests, mask any authorization headers (`Bearer **********`).
- **Path Traversal Guards:**
  ```python
  safe_dir = os.path.abspath(DATA_DIR)
  target_path = os.path.abspath(os.path.join(safe_dir, user_filename))
  if not target_path.startswith(safe_dir):
      raise ValueError("Path traversal attempt detected")
  ```

## 6. Common Mistakes to Avoid
- **Committing `.env` to Git:** Accidentally pushing live API tokens to a public GitHub repo.
- **Using `eval()` or `exec()` for parsing model JSON:** Always use standard `json.loads` or Pydantic.
- **Unrestricted File Serving in Web Apps:** Allowing users to download arbitrary files from the server filesystem.

## 7. Quality-Control Checklist
- [ ] Is `.gitignore` configured to ignore `.env`, `.venv`, and local caches?
- [ ] Are all API keys loaded via `os.environ.get()` with zero hardcoded defaults?
- [ ] Is `pickle` completely absent from the codebase?
- [ ] Are file paths sanitized against directory traversal attacks?

## 8. Verification Procedures
1. Run git status / diff check to confirm no secret files are tracked.
2. Run automated regex scan for API token patterns across the workspace.
