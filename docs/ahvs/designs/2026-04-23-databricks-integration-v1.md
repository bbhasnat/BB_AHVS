# AHVS Databricks Integration — Design Document

**Date:** 2026-04-23
**Status:** Draft — awaiting review
**Author:** hasnat@blackbird.ai

---

## 1. Problem

AHVS today assumes the target dataset is a local file (CSV/Parquet/JSONL). For workloads where the data lives on Databricks (Unity Catalog, Delta tables), the current flow forces two expensive steps:

1. **Data egress** — pulling multi-GB tables to the local machine (slow, storage-heavy, often impractical).
2. **Local compute** — running profiling, deduplication, embeddings, and splits on a single laptop instead of on elastic cloud compute.

Both stages are unnecessary: Databricks can run the compute in-place and return only the results AHVS actually needs (profile stats, chosen subsample IDs, report rows).

**Goal:** Let AHVS — primarily `ahvs_data_analyst`, secondarily `eval_command` and `ahvs genesis` — execute data operations *on Databricks* while keeping the orchestration layer (hypothesis generation, worktree splice, Claude Code invocation, EvolutionStore) local.

## 2. Scope

**In scope (this doc):**
- SQL adapter against a serverless SQL Warehouse (exists in workspace).
- Jobs adapter that submits Python tasks to serverless jobs compute (confirmed available).
- Wiring into `ahvs_data_analyst` as the first consumer (vertical slice).
- Config loader for `.databricks.env` with per-line parsing and no value logging.
- CLI surface (`--data-source databricks`, `--table`, `--databricks-mode`).

**Out of scope (deferred):**
- Running the AHVS orchestration loop itself on Databricks (stays local).
- Databricks Connect v2 (interactive remote Spark sessions) — superseded by the Jobs adapter for our use case.
- Asset Bundle packaging for AHVS cycles.
- Unity Catalog volume management, table creation, schema evolution.
- Notebook-style partial-cell re-execution (the earlier "Jupyter-style execution" idea — explicitly deferred).

## 3. Workspace assumptions (verified 2026-04-23)

Probed via the REST API with the project PAT:

| Capability | Status | Source |
|---|---|---|
| SQL Statements execution | ✅ `SELECT 1 → SUCCEEDED` | `POST /api/2.0/sql/statements` |
| SQL Warehouse serverless | ✅ confirmed by user (UI) | — |
| Clusters API | ✅ 200, 7 clusters visible | `/api/2.0/clusters/list` |
| Jobs API | ✅ 200, 25 jobs | `/api/2.1/jobs/list` |
| Cluster policies | ✅ 5 policies (Personal / Power User / Shared / Job / Legacy) | `/api/2.0/policies/clusters/list` |
| Node types | ✅ 612 | `/api/2.0/clusters/list-node-types` |
| Serverless jobs compute | ✅ confirmed by user (UI) | — |
| SQL Warehouses management (list) | ❌ 403 — `sql` scope absent on PAT | Non-blocking (we use warehouse ID from `.databricks.env`) |

Workspace identity is loaded from `.databricks.env` (gitignored, PAT-based). Existing clusters are owned by other users; AHVS will not reuse them — it provisions its own ephemeral compute.

## 4. Architecture

```
ahvs/integrations/databricks/
├── __init__.py           # factory: get_adapter(mode, config) -> DatabricksAdapter
├── auth.py               # loads .databricks.env → DatabricksConfig (never logs values)
├── base.py               # DatabricksAdapter ABC + shared result types
├── sql_adapter.py        # databricks-sql-connector → serverless SQL Warehouse
├── jobs_adapter.py       # databricks-sdk → serverless jobs compute (default)
└── compute_spec.py       # DBR version, node type, worker count, auto-terminate
```

### 4.1 Adapter interface (ABC)

```python
class DatabricksAdapter(ABC):
    @abstractmethod
    def profile_table(self, table: str, columns: list[str] | None = None) -> ProfileResult: ...
    @abstractmethod
    def run_query(self, sql: str) -> pd.DataFrame: ...
    @abstractmethod
    def sample(self, table: str, n: int, stratify_by: str | None = None) -> pd.DataFrame: ...
    @abstractmethod
    def write_table(self, df: pd.DataFrame, target: str, mode: str = "overwrite") -> None: ...
    @abstractmethod
    def run_python(self, script_path: Path, inputs: dict) -> JobResult: ...  # Jobs-only; SQL raises NotImplemented
```

### 4.2 SQL adapter — default for push-downable work

**Transport:** `databricks-sql-connector` or raw REST to `/api/2.0/sql/statements`.
**Compute:** existing serverless SQL Warehouse from `DATABRICKS_HTTP_PATH`.
**Handles:** profiling (`COUNT`, `COUNT DISTINCT`, `APPROX_COUNT_DISTINCT`, percentiles), class-balance (`GROUP BY label`), stratified sampling (`QUALIFY ROW_NUMBER() OVER (PARTITION BY …)`), train/test splits (`TABLESAMPLE` + deterministic hash modulo), basic text stats (`LENGTH`, `REGEXP_COUNT`).
**Cold start:** <10s (serverless). Round-trip for a 100M-row profile: seconds, not minutes.
**Result materialization:** only aggregates / chosen row IDs come back; raw data stays in Databricks.

### 4.3 Jobs adapter — for Python/Spark/embedding work

**Transport:** `databricks-sdk` Python library → `/api/2.1/jobs/runs/submit` with `environments` block (serverless) as the default, `new_cluster` block as fallback.
**Compute:** Databricks-managed serverless jobs compute.
**Handles:** anything that can't be expressed in SQL — embedding-based dedup, semantic clustering, HuggingFace model inference, pandas pipelines, Spark ML.
**Cold start:** ~30s (serverless). Falls back to classic `new_cluster` (~2–5 min) if serverless is unavailable in a given workspace.
**Submission pattern:**
1. AHVS writes the Python task script to a temp location.
2. Uploads to workspace (`/Users/<principal>/.ahvs/runs/<run_id>/task.py`) via Workspace API.
3. Calls `runs/submit` with serverless environment + task pointing at uploaded script.
4. Polls run state; streams logs; downloads output artifact (Parquet / JSON) from Workspace or DBFS.
5. Deletes the temp script and artifact on success.

### 4.4 Auto-selection

`--databricks-mode sql|jobs|auto` (default `auto`).

`auto` rules — SQL first, escalate to Jobs when:
- Operation is not expressible in SQL (embeddings, custom Python model inference, semantic dedup).
- Estimated result size > N rows (default 5M) and the result itself would bloat the local client.
- User explicitly requested a Python-based module (e.g., `--dedup-mode semantic`).
- Operation requires a package beyond SQL UDFs (sentence-transformers, sklearn, torch).

Decision is logged per invocation: `[databricks] mode=sql reason="push-down SQL for class_balance"`.

## 5. Auth and config

### 5.1 Resolution order (highest wins)

1. CLI flags (`--databricks-host`, `--databricks-warehouse-id`, etc.) — rarely used, primarily for overrides.
2. Process env vars (`DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `DATABRICKS_HTTP_PATH`, optional `DATABRICKS_SERVERLESS_ENVIRONMENT_KEY`).
3. `<repo>/.databricks.env` — per-line parsed, never echoed to stdout/logs.
4. `~/.databrickscfg` — standard Databricks CLI profile (fallback for multi-workspace setups).

### 5.2 `DatabricksConfig` dataclass

```python
@dataclass
class DatabricksConfig:
    host: str                    # https://...cloud.databricks.com
    token: str                   # PAT — never repr'd, never logged
    warehouse_http_path: str     # /sql/1.0/warehouses/<id>
    warehouse_id: str            # derived from http_path
    cluster_id: str | None       # optional for all-purpose reuse (unused by default)
    serverless_jobs: bool        # default True
    catalog: str | None          # default Unity Catalog (optional)
    schema: str | None           # default schema (optional)
```

### 5.3 Credential hygiene (mandatory, per CLAUDE.md)

- `__repr__` / `__str__` redact `token`.
- The loader uses per-line parsing — never `cat .databricks.env`, never `env | grep DATABRICKS`.
- Log lines include `host`, `warehouse_id`, `catalog.schema` — **never** the token.
- If the SDK or connector raises with the token in the exception message, we catch and rewrap before logging.
- HTTP errors are logged with method+path+status+requestID only; response body is truncated to 300 chars and scanned for the token substring.

## 6. CLI surface

New flags on `ahvs data_analyst` (first consumer):

```
--data-source {local,databricks}     # default: local
--table CATALOG.SCHEMA.NAME          # required when data-source=databricks
--databricks-mode {sql,jobs,auto}    # default: auto
--databricks-host URL                # override; usually from env
--databricks-warehouse-id ID         # override
--databricks-catalog CATALOG         # default UC catalog for unqualified names
--databricks-schema SCHEMA           # default schema

# --- cost controls (see §8) ---
--databricks-budget-usd FLOAT        # hard per-cycle spend cap (default: 5.00)
--databricks-confirm-each            # prompt Y/N before every remote op (default: on for Jobs, off for SQL)
--databricks-yes                     # non-interactive: pre-accept every cost prompt up to --budget
--databricks-dry-run                 # log every API call, submit nothing, cost = 0
--databricks-max-runtime-sec INT     # per-run timeout for Jobs submissions (default: 1800)
--databricks-max-sql-rows INT        # row cap on any SQL result pulled locally (default: 100000)

# --- artifact destinations (see §9) ---
--databricks-uc-volume PATH          # /Volumes/<cat>/<sch>/<vol>/ahvs_runs; required for Jobs artifacts
--keep-artifacts                     # skip cleanup of remote temp scripts + artifacts (debugging)
```

Example:

```bash
ahvs data_analyst \
  --data-source databricks \
  --table main.marketing.tiktok_posts_q1_2026 \
  --target engagement_label \
  --sample-rows 50000 \
  --dedup-mode hybrid
```

The same flags are later lifted to `ahvs genesis` and to the generated `eval_command` template for Databricks-hosted target repos.

## 7. Integration points (sequenced)

| # | Consumer | What changes | When |
|---|---|---|---|
| 1 | `ahvs_data_analyst` profiling + class balance | SQL adapter push-down | Vertical slice |
| 2 | `ahvs_data_analyst` sampling + split | SQL adapter (TABLESAMPLE / deterministic hash) | Vertical slice |
| 3 | `ahvs_data_analyst` lexical dedup | SQL adapter (MinHash UDF or Python-UDF job) | Vertical slice |
| 4 | `ahvs_data_analyst` semantic dedup + embedding clustering | Jobs adapter | Phase 2 |
| 5 | `eval_command` — Databricks-hosted targets | New eval template that accepts `--table` and runs remote | Phase 3 |
| 6 | `ahvs genesis` — accept Delta table as input | Adapter layer below the KD solver | Phase 4 |

## 8. Cost controls (no unbounded spend)

**Guiding principle:** AHVS must never silently spend money on Databricks. Every remote operation has a pre-flight cost estimate, a per-cycle budget, and an explicit user approval path. Hitting the budget **aborts** the cycle rather than exceeding it.

### 8.1 Per-cycle budget

- Every cycle starts with a budget (`--databricks-budget-usd`, default **$5.00**, overridable by env `AHVS_DATABRICKS_BUDGET_USD`).
- A `spend_tracker.json` in `<cycle_dir>/databricks/` accumulates estimated + actual spend per run.
- On submission, AHVS checks `estimated_cost + spend_to_date <= budget`. If it would exceed, the run is **refused** — not truncated — and the cycle emits a clear abort message.
- Spend-to-date is shown in every progress update: `[databricks] budget=$5.00 spent=$1.23 remaining=$3.77`.

### 8.2 Pre-flight cost estimation

Before any Databricks API call that incurs DBU cost, the adapter emits an estimate:

| Operation | Estimator | Confidence |
|---|---|---|
| SQL query | scanned bytes from `EXPLAIN FORMATTED` × serverless-SQL DBU rate × runtime projection | medium |
| Jobs submit (Python, serverless) | `max_runtime_sec × serverless-jobs DBU rate × worker count` (upper bound) | low (runtime unknown) — always shown as max |
| Jobs submit (new_cluster, non-serverless) | `max_runtime_sec × cluster DBU/hr × node count` | low |

DBU-to-USD rates are workspace-specific; AHVS reads them from `databricks_pricing.json` (shipped defaults for AWS/Azure/GCP) with override `--databricks-dbu-rate-usd`. Estimates are **upper bounds** — actual spend is almost always lower. The distinction (estimated max vs. actual) is always labeled in output.

### 8.3 Interactive confirmation

Three modes, in decreasing verbosity:

| Mode | Behavior |
|---|---|
| `--databricks-confirm-each` | Prompt Y/N before every Jobs submission and every SQL query that `EXPLAIN` says will scan > 1 GB |
| default | Prompt once per cycle for a cumulative budget approval; silent thereafter until budget hit |
| `--databricks-yes` | No prompts; still enforces `--databricks-budget-usd` cap; aborts on cap |

Prompts are **terminal-based** and block the cycle:

```
[databricks] Jobs submit: semantic_dedup on table main.marketing.posts (≈14M rows)
  compute:   serverless-jobs, max 4 workers
  timeout:   1800s
  est. cost: ≤ $0.42 (upper bound)
  budget:    $5.00, spent so far $0.05
Approve this run? [y/N]:
```

For AHVS cycles run non-interactively (e.g., under `/ahvs_multiagent`), the operator must pre-approve the budget at cycle start; the cycle aborts on any run that would exceed it.

### 8.4 Hard aborts and in-flight cancel

- A `KeyboardInterrupt` or budget breach cancels all in-flight Databricks runs via `POST /api/2.1/jobs/runs/cancel` and aborts the SQL statement via `POST /api/2.0/sql/statements/{id}/cancel`.
- AHVS never retries a cost-incurring op after it fails without explicit user re-approval (no silent retries).
- Every adapter method that returns rows has a `max_rows` cap (default 100k); oversize results are truncated server-side via `LIMIT`, with a warning, rather than pulled down and chopped locally.

### 8.5 Dry-run

`--databricks-dry-run` logs every API call to `<cycle_dir>/databricks/dry_run.log` without submitting. Used for:
- Verifying a new hypothesis won't blow budget before committing.
- CI/test runs that shouldn't touch the real workspace.
- Reproducing a failed run's call pattern.

---

## 9. Data and result destinations

**Guiding principle:** aggregated results stay local; intermediate blobs stay on Databricks. The user always knows where everything lives.

### 9.1 Destination matrix

| Artifact | Default location | Why |
|---|---|---|
| Markdown / HTML report | `<target_repo>/reports/` (local) | Readable, committed to git, portable |
| JSON summary (stats, counts, decisions) | `<target_repo>/reports/` (local) | Small, machine-readable |
| Chosen sample row IDs (post-subsampling) | `<target_repo>/reports/samples.parquet` (local) | Usually ≤ few MB |
| Train/test/val split IDs | `<target_repo>/reports/splits.parquet` (local) | Same |
| Full embeddings matrix (if generated) | UC volume: `/Volumes/<cat>/<sch>/<vol>/ahvs_runs/<cycle_id>/embeddings.parquet` | Often GB-scale — too big to ship |
| Dedup candidate pairs (if semantic dedup used) | UC volume, same path | Can be large |
| Temp Python scripts for Jobs submissions | Workspace: `/Users/<principal>/.ahvs/runs/<cycle_id>/` | Lives only during run |
| Job run logs | Downloaded to `<cycle_dir>/databricks/run_<id>.log` on completion | Local for debugging |
| Spend tracker | `<cycle_dir>/databricks/spend_tracker.json` | Always local |
| Artifact manifest | `<cycle_dir>/databricks/artifacts.json` | Index of every remote path AHVS touched |

### 9.2 UC volume requirement

The Jobs adapter **requires** `--databricks-uc-volume <path>` (or `AHVS_DATABRICKS_UC_VOLUME` env). AHVS does not create volumes — the user (or a workspace admin) pre-creates one. On first use, the adapter verifies the volume exists and is writable via a tiny probe file.

**Why require the user to create it:** volume creation needs catalog-level grants that a per-user PAT often lacks, and failing loudly at onboarding is better than failing silently during a run.

### 9.3 Cleanup policy

- **On successful run:** delete the workspace temp script directory and UC-volume intermediates, unless `--keep-artifacts`.
- **On failed run:** keep everything; log the paths so the user can inspect.
- **At cycle end:** regardless of success, write `artifacts.json` listing every remote path AHVS created, with a kept/deleted flag. Users can grep this file to find anything AHVS left behind.
- **Retention cap:** `<cycle_dir>/databricks/` is never auto-deleted; users decide when to prune.

### 9.4 Privacy posture

- Raw rows from a Databricks table **do not** leave Databricks unless explicitly sampled by the user (via `--sample-rows N`). Profiling, class balance, splits, and dedup counts return aggregates only.
- When sample rows come back, they land in `reports/samples.parquet` — predictable path, easy to `.gitignore`.
- The adapter logs `rows_downloaded` for every SQL query; a cycle summary line totals rows ever egressed.

---

## 10. User communication contract

**Guiding principle:** the user is never surprised by what Databricks did, what it cost, or where things went.

### 10.1 Live progress (stdout/stderr)

Every remote operation emits **at least** these events:

```
[databricks] SQL warm-up: warehouse f2561f1c68c0c730 resuming... (elapsed 4s)
[databricks] SQL query #1 (profile): 2.3s, scanned 180 MB, rows returned 1, est. $0.001
[databricks] Jobs submit #1 (semantic_dedup): run_id=4821, state=PENDING → RUNNING (18s)
[databricks] Jobs run #4821: state=TERMINATED, exit=SUCCESS, runtime 142s, DBU 0.12, cost ~$0.08
[databricks] Artifact written: /Volumes/main/analytics/ahvs/ahvs_runs/C042/embeddings.parquet (1.4 GB)
[databricks] Budget: spent $0.08 / $5.00, remaining $4.92
```

State transitions are polled every 5s; a transition worth surfacing is printed immediately.

### 10.2 Persistent run log

`<cycle_dir>/databricks/run_log.md` is appended in real time. Each entry:

```markdown
### [2026-04-23 14:22:05] Jobs run #4821 — semantic_dedup
- Script:        .../ahvs/runs/C042/task_semantic_dedup.py
- Compute:       serverless-jobs, max 4 workers, timeout 1800s
- Input table:   main.marketing.tiktok_posts_q1_2026 (12.8M rows)
- Estimated:     ≤ $0.42
- Actual:        142s runtime, DBU 0.12, $0.08
- Artifacts:     /Volumes/main/analytics/ahvs/ahvs_runs/C042/embeddings.parquet
- Status:        SUCCESS
```

Designed for post-hoc debugging and cost audit.

### 10.3 Cycle summary integration

`<cycle_dir>/cycle_summary.json` gains a `databricks` block:

```json
"databricks": {
  "enabled": true,
  "workspace_host": "dbc-00b83a9a-b378.cloud.databricks.com",
  "runs_submitted": 3,
  "sql_queries": 14,
  "total_cost_estimated_max_usd": 0.87,
  "total_cost_actual_usd": 0.23,
  "budget_usd": 5.00,
  "artifacts_kept": 1,
  "artifacts_deleted": 4,
  "artifact_manifest_path": ".ahvs/cycles/C042/databricks/artifacts.json"
}
```

The AHVS cycle report template is extended to render this block as a human-readable section.

### 10.4 Error format

Databricks-side errors are wrapped into three explicit types before surfacing:

| Class | Trigger | User-facing message |
|---|---|---|
| `DatabricksAuthError` | 401, scope denial, expired PAT | "Databricks auth failed: <reason>. Check `.databricks.env` — PAT may need the `sql`, `clusters`, or `jobs` scope. Host: <host>." |
| `DatabricksPermissionError` | 403, table/catalog denied | "Databricks permission denied on <resource>. Required grant: <grant>. Current identity: <me>." |
| `DatabricksRunError` | Job failure, SQL error, timeout | "Databricks run <id> failed: <cause>. Full log: <cycle_dir>/databricks/run_<id>.log" |

**Mandatory:** every error message is scrubbed for the token substring before logging; if a scrub triggers, the incident is noted in the run log so credential hygiene regressions are visible.

### 10.5 Documentation surface

- `docs/ahvs_data_analyst.md` gains a **"Using Databricks"** section with the three concrete scenarios (SQL-only, SQL + Jobs, dry-run).
- A new doc `docs/ahvs/databricks.md` covers: `.databricks.env` setup, required PAT scopes, UC volume creation walkthrough, cost model, troubleshooting the top five errors.
- `--help` output for every new flag explains the cost implication in one sentence (e.g., `--databricks-mode jobs  # Submits a Python task to serverless compute (~$0.05–$1 per run)`).

---

## 11. Vertical slice — minimum shippable

End-to-end slice, testable on one real table:

1. `DatabricksConfig` loader + env/file precedence.
2. SQL adapter only (`profile_table`, `run_query`, `sample`).
3. `ahvs_data_analyst` plumbed with `--data-source databricks` flag.
4. One real run: `ahvs data_analyst --data-source databricks --table <user-provided> …` producing the same markdown + JSON report that the local path produces today.
5. Integration test that mocks the SQL statements API (no real API calls in CI).

Deliberately out of the vertical slice: Jobs adapter, semantic dedup, embedding clustering. Those ship in Phase 2 after the SQL path is validated end-to-end.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Serverless SQL warehouse suspended on first run → latency surprise | Auto-issue a `SELECT 1` warm-up before the first meaningful query; log wake time. |
| Row-level data leaves Databricks when we least expect it | Every adapter method that returns rows has an explicit `max_rows` cap (default 100k); the profile path only returns aggregates. |
| Long-running Jobs run billed unbounded | Adapter sets `timeout_seconds` on every `runs/submit` (default 30 min); kills runs on local Ctrl-C via `runs/cancel`. |
| PAT leak via SDK error messages | Wrap every SDK call; scrub exception strings for token substrings before logging. |
| Unity Catalog permission denied mid-query | Surface as `DatabricksPermissionError` with table path + required grant; do not retry silently. |
| Two AHVS cycles racing on the same warehouse | The warehouse itself multi-tenants; no locking needed. Cycle IDs are included in every query's comment for observability. |

## 13. Open questions

- **Which table/project is the first vertical-slice target?** We need a real table (catalog.schema.name + approximate row count) to validate the SQL adapter against.
- **Output destination for Jobs adapter artifacts** — Workspace files, DBFS, or Unity Catalog volume? Default recommendation: UC volume under `/Volumes/<catalog>/<schema>/ahvs_runs/` — requires user to pre-create the volume.
- **Service principal migration** — PATs expire and belong to individuals. Long-term, AHVS should support OAuth M2M with a service principal. Deferred to Phase 3.
- **Cost observability** — Databricks tags jobs; AHVS should tag every run with `ahvs.cycle_id`, `ahvs.hypothesis_id`, `ahvs.repo`, so billing can attribute back. In scope for the Jobs adapter; not required for SQL (warehouse-level billing).

## 14. Acceptance criteria for the vertical slice

- `ahvs data_analyst --data-source databricks --table <T>` produces the same report structure as the local path, on a table of at least 1M rows, with no row data leaving Databricks except aggregated stats and the chosen sample rows.
- Cold-start end-to-end run on a warm warehouse: < 60s for profile + class balance + 10k-row stratified sample.
- Unit tests for `DatabricksConfig` loader (file + env precedence, redaction).
- Unit tests for SQL adapter (mocked Statements API).
- Integration smoke test script in `examples/databricks/` (requires real workspace, not in CI).
- `docs/ahvs_data_analyst.md` updated with the Databricks section.

---

## Appendix A — Why not Databricks Connect v2

Earlier draft included a Databricks Connect adapter. Dropped for three reasons:

1. **Requires a pre-provisioned all-purpose cluster** — conflicts with the goal of "no manual cluster management."
2. **DBR version pinning** — client library must match cluster DBR; drift causes obscure failures.
3. **Overlaps with Jobs adapter** — anything Databricks Connect does, a serverless Python job can do, with cleaner lifecycle and no version coupling.

Revisit only if a user need for interactive local Spark DataFrame manipulation emerges.

## Appendix B — References

- `databricks-sql-connector` — PyPI package for SQL adapter (`pip install databricks-sql-connector`).
- `databricks-sdk` — PyPI package for Jobs/Clusters adapter (`pip install databricks-sdk`).
- Serverless jobs compute docs — `/api/2.1/jobs/runs/submit` with `environments` block.
- Unity Catalog volumes — for persistent artifact storage.
