# AHVS Databricks Integration — Design Document (v2)

**Date:** 2026-04-23
**Status:** Draft v2 — awaiting review
**Author:** hasnat@blackbird.ai
**Supersedes:** [2026-04-23-databricks-integration-v1.md](./2026-04-23-databricks-integration-v1.md)

---

## 0. Changelog — v1 → v2

v2 is a structural rewrite prompted by two reviews of v1 (one red-team, one constructive). See [Appendix C](#appendix-c--review-consolidation) for the full consolidation. Summary of changes:

- **Architecture (§4) rewritten around operations + backends.** v1 split by transport (SQL adapter / Jobs adapter) which leaked into every call site. v2 organizes by operation (`ProfileOp`, `SampleOp`, `SplitOp`, `DedupOp`, `EmbedOp`); each op picks its own backend. `run_query(sql) → DataFrame` is removed from the interface — it contradicted v1's own privacy claims and was a credential-holder's escape hatch.
- **New §5.5 — Auth proxy daemon.** v1 assumed the adapter holds the PAT. That made every row cap, budget check, and scrubbing rule cooperative — a hypothesis could bypass them by importing the SDK directly. v2 specifies a dedicated daemon process as the *only* holder of the token, with `SO_PEERCRED` socket auth, `bytearray`/`ctypes.memset` zeroization, and server-side enforcement of every policy. Hypothesis code never sees credentials.
- **§8 Cost controls rewritten** around **always-prompt + least-costly-first + memory-of-approved-max**. `--databricks-yes` is removed (too dangerous given the security posture). Budget ledger lives inside the daemon, race-safe per `(cycle_id, hypothesis_id)`.
- **API-level corrections** in §4 and §8: serverless `environments` block shape (requires `environment_key` + top-level `environments: [...]`), UC volume (not Workspace/DBFS) for script upload, `MOD(abs(xxhash64(id)), 100)` (not `TABLESAMPLE`) for reproducible splits, `system.information_schema.table_storage_usage` (not `EXPLAIN FORMATTED`) for scanned-bytes estimates, `jobs/runs/get-output` for serverless logs.
- **§5.4 Input validation** (identifier grammar) — enforcement is now server-side in the daemon; adapter-side validation is belt-and-suspenders.
- **New §7.5 — Databricks `eval_command` contract.** `.ahvs/eval_spec.yaml` schema for Databricks-hosted eval; prevents a Phase 3 rewrite.
- **§10 Observability expanded.** SQL comment injection (`/* ahvs cycle=C042 hyp=H3 op=profile */`), job `tags`, `x-databricks-org-id` / `x-request-id` header logging, JSONL run log alongside markdown.
- **§13 Cross-cycle memory specified.** `databricks_approvals.jsonl` schema + p50/p95 aggregates seed next-cycle defaults and prompt copy ("last approved $X for this table").
- **§11 + §14 Testing strategy expanded.** VCR-style fixtures, opt-in real-warehouse contract tests, property-based grammar tests, daemon `SO_PEERCRED` subprocess tests, budget-race concurrency regression test.
- **Preserved from v1:** §2 scope fence, Appendix A's reasoned rejection of Databricks Connect v2, the per-artifact destination matrix (now §9.1), SQL-first vertical-slice discipline, acceptance criteria with real-table size threshold.

---

## 1. Problem

AHVS today assumes the target dataset is a local file (CSV/Parquet/JSONL). For workloads where the data lives on Databricks (Unity Catalog, Delta tables), the current flow forces two expensive steps:

1. **Data egress** — pulling multi-GB tables to the local machine (slow, storage-heavy, often impractical).
2. **Local compute** — running profiling, deduplication, embeddings, and splits on a single laptop instead of on elastic cloud compute.

Both stages are unnecessary: Databricks can run the compute in-place and return only the results AHVS actually needs (profile stats, chosen sample IDs, report rows).

**Goal:** Let AHVS — primarily `ahvs_data_analyst`, secondarily `eval_command` and `ahvs genesis` — execute data operations *on Databricks* while keeping the orchestration layer (hypothesis generation, worktree splice, Claude Code invocation, EvolutionStore) local. Do this without giving hypothesis code (which is authored by Claude and therefore not trusted) access to the Databricks PAT or the ability to bypass row caps and budget limits.

## 2. Scope

**In scope (this doc):**
- Auth proxy daemon: dedicated process holding the PAT, enforcing all policies server-side.
- Operation-oriented data plane (`ProfileOp`, `SampleOp`, `SplitOp`, `DedupOp`, `EmbedOp`) with pluggable SQL/Jobs backends.
- Wiring into `ahvs_data_analyst` as the first consumer (vertical slice).
- Config loader for `.databricks.env` with per-line parsing, zeroization handoff, and no value logging.
- CLI surface (`--data-source databricks`, `--table`, `--databricks-op-hint`).
- `eval_spec.yaml` contract sketch for Phase 3 (Databricks-hosted `eval_command`).
- Cross-cycle memory (`databricks_approvals.jsonl`) for propose-least-cost defaults.

**Out of scope (deferred):**
- Running the AHVS orchestration loop itself on Databricks (stays local).
- Databricks Connect v2 (interactive remote Spark sessions) — superseded by the Jobs backend for our use case. See [Appendix A](#appendix-a--why-not-databricks-connect-v2).
- Asset Bundle packaging for AHVS cycles.
- Unity Catalog volume creation, schema evolution, table DDL.
- Notebook-style partial-cell re-execution ("Jupyter-style execution") — explicitly deferred.
- Full service-principal (OAuth M2M) migration — daemon design anticipates it via `reload` signal; implementation is Phase 4.

## 3. Workspace assumptions (verified 2026-04-23)

Probed via REST with the project PAT on workspace `dbc-00b83a9a-b378.cloud.databricks.com`. What was actually verified vs. what was confirmed by the user is distinguished below — v1 was loose with the word "verified."

| Capability | Verification | Confidence |
|---|---|---|
| SQL Statements execution | `POST /api/2.0/sql/statements` → `SELECT 1` succeeded end-to-end | High — actual round trip |
| SQL Warehouse is serverless | User confirmed via UI | Medium — self-report, not API-verified |
| Clusters API | `GET /api/2.0/clusters/list` → 200, 7 clusters visible | High |
| Jobs API | `GET /api/2.1/jobs/list` → 200, 25 jobs, **zero** using serverless | High — but non-use is a weak signal for availability |
| Cluster policies | `GET /api/2.0/policies/clusters/list` → 5 policies | High |
| Node types | `GET /api/2.0/clusters/list-node-types` → 612 types | High |
| Serverless jobs compute | User confirmed via UI (Compute dropdown shows "Serverless") | Medium — self-report |
| SQL Warehouses management list | `GET /api/2.0/sql/warehouses` → 403 (`sql` scope absent on PAT before broadening) | Non-blocking — we use warehouse ID from `.databricks.env`; statements execution uses a different auth path |
| Serverless SQL cold-start latency | Claimed in v1 as "<10s" — **not measured** | None — remove from doc claims |
| 100M-row profile "in seconds" | Claimed in v1 — **not measured** | None — remove from doc claims |

Existing clusters are owned by other workspace users; AHVS will provision its own ephemeral compute (see §4.3). No cluster reuse.

## 4. Architecture — operations + backends

v1 organized the data plane by transport (SQLAdapter / JobsAdapter). That was wrong: the cost/policy/selection logic belongs with the *operation*, not the transport, and the transport split leaked into every caller. v2 inverts it.

### 4.1 Layering

```
┌──────────────────────────────────────────────────────┐
│  AHVS orchestration (hypothesis gen, worktree, ...)  │  local, trusted
└──────────────────────────────────────────────────────┘
                         │
┌──────────────────────────────────────────────────────┐
│  DataPlane — typed operations:                       │
│    ProfileOp  SampleOp  SplitOp  DedupOp  EmbedOp    │  local, trusted
│    each: .select_backend(ctx) .estimate_cost() .run()│
└──────────────────────────────────────────────────────┘
                         │  Unix domain socket (SO_PEERCRED-authenticated)
                         │  no PAT crosses this boundary
                         ▼
┌──────────────────────────────────────────────────────┐
│  AHVS auth proxy daemon (ahvs.proxy)                 │  local, privileged
│  - holds the ONLY copy of the PAT (zeroized on exit) │
│  - enforces: identifier grammar, row caps, budget    │
│  - holds per-(cycle_id,hyp_id) budget ledger         │
│  - speaks Databricks APIs outward                    │
└──────────────────────────────────────────────────────┘
                         │  HTTPS
                         ▼
                 Databricks workspace
                   ├─ serverless SQL Warehouse
                   └─ serverless jobs compute
```

### 4.2 Operations (the `DataPlane` interface)

```python
class DataPlaneOp(Protocol):
    cycle_id: str
    hypothesis_id: str
    def select_backend(self, ctx: OpContext) -> Backend: ...
    def estimate_cost(self, backend: Backend) -> CostEstimate: ...
    def run(self, backend: Backend) -> OpResult: ...
```

| Op | What it does | Typical backend |
|---|---|---|
| `ProfileOp(table)` | Column types, null rates, percentiles, cardinalities | SQL (aggregates only, 1 row out) |
| `SampleOp(table, n, stratify_by=None)` | Pick N row IDs (stratified or uniform), return IDs + optional row payload | SQL (deterministic hash sampling) |
| `SplitOp(table, ratios, seed)` | Train/val/test ID partition via `MOD(abs(xxhash64(id_col, seed)), 100)` | SQL |
| `DedupOp(table, mode)` | Lexical (MinHash UDF in SQL), semantic (Jobs), hybrid | SQL or Jobs |
| `EmbedOp(table, text_col, model)` | Compute embeddings, write to UC volume, return row-count + path | Jobs only |

**Deleted from v1:** `run_query(sql) → DataFrame` on the ABC. It was a credential-holder's escape hatch and it contradicted §9.4's privacy claim. If a future operation needs SQL that isn't expressible through an Op, we add a new Op — not a raw-SQL back door.

### 4.3 Backends (transport executors)

Backends are thin; they know how to talk to one transport, nothing else.

**`SQLBackend`** — speaks `POST /api/2.0/sql/statements` through the proxy daemon.
- Serverless SQL Warehouse from `DATABRICKS_HTTP_PATH`.
- Cold-start: measured on first use per cycle; logged. No claims about latency in this doc until we have real data.
- `wait_timeout` maxes at 50s per API contract; polling past that via `GET /api/2.0/sql/statements/{id}`.
- Cancel via `POST /api/2.0/sql/statements/{id}/cancel` — best-effort, returns 200 even if already finished.

**`JobsBackend`** — speaks `POST /api/2.1/jobs/runs/submit` with a serverless-jobs `environments` block through the proxy daemon.
- Submission payload shape (corrected from v1):
  ```json
  {
    "run_name": "ahvs-C042-H3-embed",
    "tasks": [{
      "task_key": "t0",
      "environment_key": "ahvs_env",
      "spark_python_task": {
        "python_file": "/Volumes/<catalog>/<schema>/ahvs_runs/<cycle_id>/task_embed.py"
      },
      "timeout_seconds": 1800
    }],
    "environments": [{
      "environment_key": "ahvs_env",
      "spec": {
        "client": "2",
        "dependencies": ["sentence-transformers==2.7.0", "pyarrow"]
      }
    }],
    "tags": {
      "ahvs.cycle_id": "C042",
      "ahvs.hypothesis_id": "H3",
      "ahvs.op": "embed"
    }
  }
  ```
- Script upload destination: **UC volume** at `/Volumes/<catalog>/<schema>/<volume>/ahvs_runs/<cycle_id>/task_<op>.py`. Not Workspace API, not DBFS (deprecated on serverless).
- Run state: poll every 5s via `GET /api/2.1/jobs/runs/get`.
- Logs: `GET /api/2.1/jobs/runs/get-output` (serverless) — not DBFS cluster logs.
- Cancel: `POST /api/2.1/jobs/runs/cancel` — also best-effort; see §10.4 orphan handling.
- Fallback path: if serverless jobs is unavailable at run time (error response naming it explicitly), submission is refused with a typed error **and** a user prompt — no silent fall-through to classic `new_cluster`. v1's silent fallback was 20–100× more expensive and failed to re-approve.

### 4.4 Backend selection

Every Op ships with a default backend and a fallback. Selection is a method on the Op, not a tangle of if/elif in the data plane:

```python
class ProfileOp:
    def select_backend(self, ctx) -> Backend:
        return ctx.sql_backend        # always SQL
class EmbedOp:
    def select_backend(self, ctx) -> Backend:
        return ctx.jobs_backend       # SQL can't do transformer inference
class DedupOp:
    def select_backend(self, ctx) -> Backend:
        return ctx.sql_backend if self.mode == "lexical" else ctx.jobs_backend
```

CLI flag `--databricks-op-hint <op>:<backend>` overrides the default per op (e.g., `--databricks-op-hint dedup:jobs` to force Jobs for lexical dedup too). Selection is always logged: `[databricks] op=ProfileOp backend=sql reason=default`.

## 5. Auth, config, and the proxy daemon

**Guiding principle:** the PAT lives in exactly one process's memory for the shortest time possible, and every policy is enforced on the server side of a credential-free boundary.

### 5.1 Resolution order (for daemon startup — not hypothesis processes)

Only the proxy daemon reads these. Hypothesis processes never see any of them.

1. CLI flags passed to `ahvs proxy start` (rarely used; for overrides).
2. Process env vars passed to the daemon (`DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `DATABRICKS_HTTP_PATH`).
3. `<repo>/.databricks.env` — per-line parsed, source file unlinked after load.
4. `~/.databrickscfg` — standard Databricks CLI profile (fallback for multi-workspace setups).

### 5.2 `DatabricksConfig` dataclass (daemon-internal only)

```python
@dataclass
class DatabricksConfig:
    host: str                   # https://...cloud.databricks.com
    _token: bytearray           # mutable; zeroizable via ctypes.memset
    warehouse_http_path: str    # /sql/1.0/warehouses/<id>
    warehouse_id: str           # derived from http_path
    uc_volume: str              # /Volumes/<cat>/<sch>/<vol> — required for Jobs
    catalog: str | None
    schema: str | None
    def __repr__(self) -> str:
        return f"DatabricksConfig(host={self.host!r}, warehouse_id={self.warehouse_id!r}, token=<redacted>)"
```

**Nothing outside the daemon imports or instantiates this class.**

### 5.3 Credential hygiene (mandatory, per CLAUDE.md)

- `__repr__` / `__str__` redact the token; `_token` is `bytearray`, not `str` (Python `str` is immutable → unzeroizable).
- Loader uses per-line parsing — never `cat .databricks.env`, never `env | grep DATABRICKS`.
- Log lines include `host`, `warehouse_id`, `catalog.schema`, request IDs — **never** the token.
- All outbound requests and inbound responses pass through a scrubber that replaces any substring matching the active token with `<REDACTED>` before logging. If the scrubber triggers on an outbound log, the daemon records an incident in `~/.ahvs/proxy/log/incidents.jsonl` — hygiene regressions become visible.
- Exception messages from `databricks-sdk` or `databricks-sql-connector` are caught, the token-substring is scrubbed, and a typed `DatabricksError` is re-raised.

### 5.4 Input validation (enforced server-side in the daemon)

All inputs crossing the socket are validated before touching Databricks:

- **Table identifiers** match `^[A-Za-z_][A-Za-z0-9_]{0,127}(\.[A-Za-z_][A-Za-z0-9_]{0,127}){0,2}$` (1–3 dot-separated parts, alphanumerics + underscore, each part ≤ 128 chars per Unity Catalog limits). SQL builders backtick-quote validated identifiers (`` `cat`.`sch`.`tbl` ``) as belt-and-suspenders.
- **Column names** same grammar as a single-part identifier.
- **Numeric parameters** (row counts, seeds, percentages) range-checked.
- **SQL templates** are fixed strings in the daemon; identifiers are interpolated only after validation. No f-strings over user-supplied SQL. No `exec`-style query submission from the client side — hypotheses choose Ops, not SQL.

Validation failures return a typed `ValidationError` to the client. The daemon never attempts a "best effort" parse of a malformed input.

### 5.5 Auth proxy daemon (`ahvs.proxy`) — new

The daemon is the keystone of the security model. Everything policy-related lives on its server side.

**5.5.1 Lifecycle**

- Install: `ahvs install` writes a systemd user unit (`~/.config/systemd/user/ahvs-proxy.service`) + creates `~/.ahvs/proxy/`.
- Start: `systemctl --user start ahvs-proxy` or fallback `ahvs proxy start` (detached subprocess with PID file).
- Status: `ahvs proxy status` → running/stopped + uptime + open client count.
- Stop: `ahvs proxy stop` → `SIGTERM`, graceful drain, token zeroization, exit.
- Auto-start per cycle (alternative): `ahvs` CLI spawns the daemon on first invocation if not running and tears it down at cycle end. Configurable via `AHVS_PROXY_MODE=persistent|per-cycle`, default persistent.

**5.5.2 Socket**

- Path: `~/.ahvs/proxy/sock`, permissions `0600`.
- Protocol: line-delimited JSON, one request per line, one response per line.
- Auth: `SO_PEERCRED` → check peer uid matches daemon owner; reject otherwise. Peer executable path is checked against an allowlist (`/usr/bin/python3`, `/home/<user>/.claude/…/claude`, the `ahvs` entrypoint). Foreign binaries get `AuthError` and the attempt is logged.
- Handshake: first request per connection is `{"op": "hello", "cycle_id": "…", "hypothesis_id": "…"}`. The daemon opens a per-`(cycle_id, hypothesis_id)` budget ledger. Subsequent requests on the same connection inherit that context.

**5.5.3 Token zeroization**

```python
# Load — once, during daemon startup
path = Path(".databricks.env")
raw = path.read_bytes()
token_start = raw.find(b"DATABRICKS_TOKEN=") + len(b"DATABRICKS_TOKEN=")
token_end = raw.find(b"\n", token_start)
token = bytearray(raw[token_start:token_end])  # mutable, zeroizable
# Zero the source buffer immediately
ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(raw)), 0, len(raw))
del raw
# Optionally unlink the source file if AHVS_PROXY_UNLINK_SOURCE=1
# (off by default; user may want to keep .databricks.env for host/http_path)
```

On `SIGTERM` / graceful exit:
```python
ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(token)), 0, len(token))
```

This is best-effort on a GC'd runtime. Python interns strings; we deliberately avoid `str` for the token. A bytes buffer held in a single `bytearray` reference is the practical ceiling on zeroization in CPython without native extensions.

**5.5.4 Per-client budget ledger**

Each handshake spawns an in-memory ledger:
```python
@dataclass
class BudgetLedger:
    cycle_id: str
    hypothesis_id: str
    approved_budget_usd: float  # set by a BudgetApprovalRequest; 0 until approved
    spent_usd: float = 0.0
    max_rows_approved: int = 0
    max_runtime_sec_approved: int = 0
    approvals: list[Approval]     # append-only audit trail
```

Every cost-incurring request is checked against the ledger **inside a lock** — no race between parallel submits from different hypothesis processes.

**5.5.5 Crash recovery**

- On startup, daemon scans `~/.ahvs/proxy/inflight/*.json` for jobs submitted but not confirmed-complete. For each, it calls `GET /api/2.1/jobs/runs/get` to reconcile state.
- If a run is still running and the owning cycle is no longer alive, the daemon issues `runs/cancel` (user-configurable — default cancel).
- If a run completed, its actual cost is logged to `databricks_approvals.jsonl` and the inflight file is removed.

**5.5.6 Log rotation**

- Logs: `~/.ahvs/proxy/log/daemon.log` via `RotatingFileHandler` (10 MB × 5 files).
- Structured events: `~/.ahvs/proxy/log/events.jsonl` (one line per request/response).
- Both pass through the token scrubber before write.

**5.5.7 PAT rotation (stub)**

- `SIGHUP` → re-read `.databricks.env`, load new token into a new `bytearray`, atomically swap, `memset` the old buffer. In-flight requests use the token reference captured at dispatch, so swap is safe.
- Out of scope: full service-principal OAuth flow. Anticipated in §12.

## 6. CLI surface

New flags on `ahvs data_analyst` (first consumer):

```
--data-source {local,databricks}     # default: local
--table CATALOG.SCHEMA.NAME          # required when data-source=databricks
--databricks-op-hint OP:BACKEND      # override per-op backend; repeatable
--databricks-host URL                # daemon startup only; rarely used
--databricks-catalog CATALOG         # default UC catalog for unqualified names
--databricks-schema SCHEMA           # default schema
--databricks-uc-volume PATH          # /Volumes/<cat>/<sch>/<vol>; required for Jobs

# --- cost controls (§8) ---
--databricks-budget-usd FLOAT        # per-(cycle,hypothesis) approved ceiling; prompt seeds from memory
--databricks-max-runtime-sec INT     # per-run timeout (default 1800)
--databricks-max-sql-rows INT        # row cap on any sample payload (default 100000)
--databricks-dry-run                 # log every API call, submit nothing

# --- artifact destinations (§9) ---
--keep-artifacts                     # skip cleanup of remote temp scripts + artifacts (debug)

# --- daemon management ---
ahvs proxy start | stop | status | reload
```

**Removed from v1:** `--databricks-yes` (auto-approve). Every cost-incurring operation prompts per §8. Users who want non-interactive runs accept a pre-approved budget for the full cycle via the handshake (and have memory-informed defaults seeding the prompt).

Example:

```bash
ahvs data_analyst \
  --data-source databricks \
  --table main.marketing.tiktok_posts_q1_2026 \
  --target engagement_label \
  --sample-rows 50000 \
  --dedup-mode hybrid \
  --databricks-uc-volume /Volumes/main/analytics/ahvs_vol
```

## 7. Integration points (sequenced)

| # | Consumer | What changes | Phase |
|---|---|---|---|
| 1 | `ahvs_data_analyst` — profile + class balance | `ProfileOp` over `SQLBackend` | Vertical slice |
| 2 | `ahvs_data_analyst` — sampling + split | `SampleOp` + `SplitOp` (hash-based) | Vertical slice |
| 3 | `ahvs_data_analyst` — lexical dedup | `DedupOp(mode=lexical)` — MinHash UDF over SQL | Vertical slice |
| 4 | `ahvs_data_analyst` — semantic dedup + embedding | `DedupOp(mode=semantic)` + `EmbedOp` over `JobsBackend` | Phase 2 |
| 5 | `eval_command` for Databricks-hosted targets | New `eval_spec.yaml` contract — see §7.5 | Phase 3 |
| 6 | `ahvs genesis` — Delta table input | Schema probe via `ProfileOp` before solver invocation | Phase 4 |

### 7.5 Databricks `eval_command` contract (Phase 3 sketch)

For target repos whose evaluation runs on Databricks, AHVS needs a contract richer than "print a metric to stdout." Design now to avoid a Phase 3 rewrite.

**Target repo declares** `.ahvs/eval_spec.yaml`:

```yaml
version: 1
runtime: databricks
entrypoint: eval/run_eval.py     # Python file in the repo
inputs:
  - name: test_set
    table: "${params.catalog}.${params.schema}.test_set_v3"
    required_columns: [id, text, label]
outputs:
  - name: metric
    table: "${params.catalog}.${params.schema}.ahvs_eval_metrics"
    metric_column: f1_macro
    filter_column: run_id          # AHVS-generated, isolates this run's metric row
resources:
  uc_volume: "${params.uc_volume}"
  max_runtime_sec: 600
  env:
    - "scikit-learn==1.4.0"
```

**AHVS's behavior:**
1. Read `eval_spec.yaml` during cycle setup.
2. For each hypothesis, submit a Jobs run with the spec's `entrypoint`, `resources.env`, and the generated `run_id` as task param.
3. After completion, read the metric via `SELECT metric FROM <table> WHERE run_id = ?`.
4. Feed metric into the cycle's keep/revert logic exactly as today.

**Why sketch now:** the Op interface in §4.2 is the right shape *if* we know eval_command eventually calls `SubmitEvalJobOp(spec, run_id) → MetricResult`. If we design it without this lens, we'll find in Phase 3 that eval wants raw-SQL or raw-job access the ABC no longer provides, and we'll rewrite. Sketching now keeps the ABC honest.

## 8. Cost controls — always prompt, least-cost first, memory-informed

**Guiding principle:** AHVS proposes the cheapest configuration that meets the task, shows the cost estimate, and prompts the user. No silent spend. Cross-cycle memory informs what "least costly" looks like for this user, this table, this op.

### 8.1 Propose-least-cost

Every Op ships a `propose_config(ctx) → list[ConfigProposal]` method returning 1–3 candidates, cheapest first. Example for `EmbedOp`:

```
[
  { name: "small-serverless", workers: 1, timeout: 600, est_usd_max: 0.08 },
  { name: "default-serverless", workers: 2, timeout: 1200, est_usd_max: 0.22 },
  { name: "large-serverless", workers: 4, timeout: 1800, est_usd_max: 0.62 }
]
```

The daemon returns all three to the client, which renders the prompt (§8.3). Default selection is the smallest that exceeds the user's historical p95 runtime for this op-type (from memory, §13).

### 8.2 Cost estimation (honest about uncertainty)

| Operation | Estimator | Confidence |
|---|---|---|
| SQL profile / aggregate | `system.information_schema.table_storage_usage` → table size bytes × serverless-SQL DBU rate × projected scan fraction | Low-medium (scan fraction is a heuristic per column) |
| SQL sample / split | Same as above, multiplied by sort-related surcharge | Low-medium |
| Jobs submit | `workers × timeout_sec × serverless-jobs DBU rate` (upper bound assuming full timeout and full parallelism) | Low — real cost is almost always lower |

Pricing defaults shipped in `ahvs/integrations/databricks/pricing.json` (AWS/Azure/GCP); override via daemon flag. Estimates are **upper bounds** and always labeled as such in prompts. The v1 claim of "medium confidence" from `EXPLAIN FORMATTED` was wrong — DBSQL `EXPLAIN` doesn't return reliable pre-execution scanned-bytes. Estimates are best-effort, not authoritative.

### 8.3 Interactive confirmation

Every cost-incurring operation prompts, unless the current cycle has a budget already approved via handshake. Prompt shape:

```
[databricks] Op: EmbedOp on main.marketing.tiktok_posts_q1_2026 (~12.8M rows)
  Cheapest proposal:  small-serverless (1 worker, 600s timeout)
  Estimated cost:     ≤ $0.08 (upper bound)
  Historical context: last approved $0.22 for this table 2026-04-21
                      your p95 approved budget: $0.50
  Budget so far this cycle: $0.05 approved, $0.03 spent
  Approve this config? [Y / n / ?show-alternatives]:
```

- `Y`: run with small-serverless.
- `n`: abort op; cycle continues with this hypothesis marked failed.
- `?`: show the other proposals; user picks one; re-prompt.

For non-interactive cycle starts (e.g., `/ahvs_multiagent`), the operator provides `--databricks-budget-usd <X>` at cycle start; the daemon pre-approves anything whose `est_usd_max` fits under the remaining budget. Anything exceeding forces an interactive prompt even in non-interactive mode; if there's no TTY, the cycle aborts with a clear error. No silent auto-approve.

### 8.4 Hard aborts, cancel, and orphan handling

- A `KeyboardInterrupt`, budget breach, or client disconnect triggers best-effort cancel via `runs/cancel` / `statements/cancel`. Cancel is best-effort — success is not guaranteed.
- **Orphan handling:** before every submit, daemon writes an inflight record at `~/.ahvs/proxy/inflight/<run_id>.json` (contains `run_id`, submission time, cycle_id, cost estimate). On daemon startup (§5.5.5), these are reconciled. Orphans are cancelled or logged as "completed-while-untracked" with their actual cost attributed.
- No silent retries after a cost-incurring op fails. Retry requires explicit user re-prompt.

### 8.5 Dry-run (hard transport-isolated)

`--databricks-dry-run` sets a daemon-level flag that causes every HTTP egress to short-circuit at the scrubber/logger boundary (a single `if dry_run: log_and_return_stub()` guard wrapping the outbound client). The real HTTP client is never invoked. A unit test asserts no network activity in dry-run mode via `socket.socket` mock.

## 9. Data and result destinations

### 9.1 Destination matrix

| Artifact | Default location | Why |
|---|---|---|
| Markdown / HTML report | `<target_repo>/reports/` (local) | Readable, committed to git, portable |
| JSON summary | `<target_repo>/reports/` (local) | Small, machine-readable |
| Chosen sample row IDs | `<target_repo>/reports/samples.parquet` (local) | ≤ few MB |
| Train/test/val split IDs | `<target_repo>/reports/splits.parquet` (local) | Same |
| Full embeddings matrix | UC volume: `/Volumes/<cat>/<sch>/<vol>/ahvs_runs/<cycle_id>/embeddings.parquet` | Often GB-scale |
| Dedup candidate pairs | UC volume, same path | Can be large |
| Jobs submission scripts | UC volume: `/Volumes/<cat>/<sch>/<vol>/ahvs_runs/<cycle_id>/task_<op>.py` | Serverless-compatible (not DBFS) |
| Job run logs | Downloaded to `<cycle_dir>/databricks/run_<id>.log` on completion | Local for debugging |
| Spend ledger | `<cycle_dir>/databricks/spend_ledger.jsonl` | Always local; mirrors daemon ledger on completion |
| Artifact manifest | `<cycle_dir>/databricks/artifacts.json` | Index of every remote path AHVS touched |

### 9.2 UC volume requirement

The Jobs backend **requires** `--databricks-uc-volume <path>` (or `AHVS_DATABRICKS_UC_VOLUME`). AHVS does not create volumes — the user (or workspace admin) pre-creates one. On first use, the daemon verifies writability via a probe file. Volume creation needs catalog-level grants that a per-user PAT often lacks; failing loudly at onboarding beats failing mid-cycle.

### 9.3 Cleanup policy

- **On success:** delete the UC-volume script file and any non-manifest intermediates, unless `--keep-artifacts`. Kept artifacts (embeddings, dedup pairs) stay per cleanup policy config.
- **On failure:** keep everything. Log paths so the user can inspect.
- **At cycle end:** always write `artifacts.json` listing every remote path AHVS created, with kept/deleted flag.
- **Retention cap:** `<cycle_dir>/databricks/` never auto-deleted. User prunes.

### 9.4 Privacy posture (strengthened from v1)

- Raw rows do not leave Databricks **as a consequence of adapter design**, not just convention. The only Op that returns rows is `SampleOp`, and it enforces `max_rows` server-side in the daemon; attempts to sample more result in a `ValidationError` before any SQL runs.
- `ProfileOp`, `SplitOp`, `DedupOp(lexical)` return only aggregates or ID lists.
- `DedupOp(semantic)` and `EmbedOp` write their large artifacts to UC volume (§9.1); only summary stats and the UC volume path come back.
- Daemon logs `rows_returned` per SQL statement; cycle summary totals rows ever egressed.
- UC row filters and column masks are detected on first touch of a table; their presence is logged and surfaced in the report so cycle lessons don't misattribute observed distributions. (Mitigation stub in §12.)

## 10. User communication contract

### 10.1 Live progress

Every remote op emits structured events to stdout, mirrored to files in §10.2:

```
[databricks C042/H3] SQL warm-up: warehouse wake 4.2s
[databricks C042/H3] ProfileOp on main.marketing.posts: 2.3s, rows_returned=1, est=$0.001
[databricks C042/H3] EmbedOp submit: run_id=4821, state=PENDING → RUNNING (18s)
[databricks C042/H3] EmbedOp run 4821: TERMINATED, exit=SUCCESS, 142s, DBU 0.12, $0.08
[databricks C042/H3] Artifact: /Volumes/main/analytics/ahvs_vol/ahvs_runs/C042/embeddings.parquet (1.4 GB)
[databricks C042/H3] Budget: approved $1.00, spent $0.08, remaining $0.92
```

Polling interval: 5s; state transitions surface immediately.

### 10.2 Persistent logs (markdown + JSONL)

- `<cycle_dir>/databricks/run_log.md` — human-readable, one block per op.
- `<cycle_dir>/databricks/run_log.jsonl` — grep-able, same events as stdout, one JSON object per line with `ts`, `cycle_id`, `hypothesis_id`, `op`, `event`, `payload`.

Every entry includes: Databricks `x-request-id` and `x-databricks-org-id` headers (what support asks for at 2am), `run_id` or `statement_id`, full traceback (in the JSONL `payload.traceback` on failures), and the op's cost estimate + actual.

### 10.3 Correlation — SQL comment injection + job tags

Every SQL statement is issued with a leading comment:

```sql
/* ahvs cycle=C042 hyp=H3 op=ProfileOp */
SELECT COUNT(*) FROM `main`.`marketing`.`tiktok_posts`;
```

Every Jobs submit carries tags:

```json
"tags": {"ahvs.cycle_id": "C042", "ahvs.hypothesis_id": "H3", "ahvs.op": "EmbedOp"}
```

This lets Databricks admins attribute cost and trace issues from their side, and lets AHVS cycle lessons join against `system.query.history` for historical analysis.

### 10.4 Cycle summary integration

`<cycle_dir>/cycle_summary.json` gains a `databricks` block:

```json
"databricks": {
  "enabled": true,
  "workspace_host_fingerprint": "sha256:e3b0c442...",
  "workspace_host": "dbc-00b83a9a-b378.cloud.databricks.com",
  "ops_executed": 7,
  "total_cost_estimated_max_usd": 0.87,
  "total_cost_actual_usd": 0.23,
  "budget_usd": 5.00,
  "rows_returned_total": 50000,
  "artifacts_kept": 1,
  "artifacts_deleted": 4,
  "artifact_manifest_path": ".ahvs/cycles/C042/databricks/artifacts.json"
}
```

Cycle report template gains a human-readable Databricks section.

### 10.5 Error format

Three typed error classes, rendered with actionable messages:

| Class | Trigger | Message pattern |
|---|---|---|
| `DatabricksAuthError` | 401, scope denial, expired PAT | `"Databricks auth failed on <host>: <reason>. Check .databricks.env — PAT may need scopes [sql, clusters, jobs]. Request ID: <id>."` |
| `DatabricksPermissionError` | 403, UC grant denied | `"Databricks permission denied on <resource>. Required grant: <grant>. Current identity: <me>. Request ID: <id>. If the grant is owned by a different admin, escalate via <your workspace's access-request process>."` |
| `DatabricksRunError` | Job failure, SQL error, timeout, orphan reconciliation | `"Databricks run <id> failed: <cause>. Full log: <cycle_dir>/databricks/run_<id>.log. Request ID: <id>."` |

All messages are scrubbed for the token substring; scrubber triggers are logged as hygiene incidents (§5.3).

### 10.6 Documentation surface

- `docs/ahvs_data_analyst.md` gains a **"Using Databricks"** section.
- New doc `docs/ahvs/databricks.md` covers: `.databricks.env` setup, required PAT scopes, proxy daemon install/start, UC volume creation walkthrough, cost model, top-5 error troubleshooting.
- `--help` for every new flag explains cost implications in one sentence.

## 11. Vertical slice — minimum shippable

End-to-end on one real table, SQL-only (no Jobs backend, no semantic dedup, no embeddings):

1. `DatabricksConfig` + `.databricks.env` loader with zeroization.
2. Auth proxy daemon — core features: SO_PEERCRED auth, identifier grammar, budget ledger, inflight tracking.
3. `SQLBackend` only.
4. `ProfileOp`, `SampleOp`, `SplitOp`, `DedupOp(lexical)`.
5. `ahvs_data_analyst` plumbed with `--data-source databricks` flag.
6. One real run on a user-provided table (≥ 1M rows) producing the same markdown + JSON report that the local path produces today.
7. Test pack — see §11.1.

Deliberately out of the vertical slice: JobsBackend, EmbedOp, DedupOp(semantic), eval_spec.yaml contract. Phase 2+.

### 11.1 Testing strategy

| Test | Purpose | Fixture approach |
|---|---|---|
| Daemon startup / shutdown | Token load + zeroize round trip | `tempfile` + `ctypes.string_at` probe to assert bytes cleared post-memset |
| Daemon SO_PEERCRED reject | Foreign-uid connection refused | Subprocess forked under a different uid (or mocked `getsockopt`) |
| Identifier grammar | Strict accept/reject | Property-based (`hypothesis` library) over random strings |
| SQL Statements API shape | Happy + 4 failure modes | VCR-style recorded JSON fixtures in `tests/fixtures/databricks/` |
| Budget ledger race | Concurrent clients can't exceed budget | `pytest` + two threads through the ledger; invariant check at end |
| SQL comment injection | Every statement carries `/* ahvs cycle=... */` | Mock the SQL client, assert every call's SQL starts with the comment |
| Cost scrubber | Token never appears in logs | Inject token into synthetic log payload, assert scrubber replaces |
| Dry-run isolation | No network activity | Mock `socket.socket`, run full flow, assert zero connects |
| Contract test (opt-in, `AHVS_DATABRICKS_CONTRACT=1`) | Response shapes match reality | Live against workspace; gated out of CI |

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Serverless SQL warehouse cold-start surprise | Warm-up query logged on cycle start; latency recorded in memory for future estimates |
| Row-level data egress via bug or rogue Op | Row caps enforced server-side in daemon; every egress logged; `SampleOp.max_rows` is the only entrypoint |
| Unbounded Jobs runtime | `timeout_seconds` on every submit; cancel on budget breach; inflight reconciliation on daemon restart |
| PAT leak via SDK error messages | Scrubber at every log boundary; hygiene incidents recorded |
| UC permission denied mid-run | Typed `DatabricksPermissionError` with grant + identity; no silent retry |
| UC row filters / column masks silently altering stats | Daemon queries `information_schema.row_filters` / `column_masks` on first touch per table; logs presence; report flags "stats reflect filtered view" |
| PAT rotation without daemon restart | `SIGHUP` → swap + memset old buffer (§5.5.7) |
| Orphan Jobs run (cycle died post-submit) | `~/.ahvs/proxy/inflight/*.json`; reconciled on daemon startup |
| Dependency CVE in `databricks-sdk` / `databricks-sql-connector` | Pin minor version; `pip-audit` in CI; quarterly review |
| Workspace-host leak into portable cycle lessons | Fingerprint (`sha256`) instead of raw host in `evolution/lessons.jsonl`; raw host only in cycle-local `cycle_summary.json` |
| Two cycles race on same `cycle_id` | Daemon rejects duplicate handshake; `cycle_id` includes wall-clock timestamp by construction |
| TTY-less cycle hits a cost prompt | Abort with clear error + guidance to pass `--databricks-budget-usd`; never silent-block |

## 13. Open questions and cross-cycle memory

### 13.1 `databricks_approvals.jsonl` schema

Lives at `<target_repo>/.ahvs/memory/databricks_approvals.jsonl`. Append-only. One record per user decision:

```json
{
  "ts": "2026-04-23T14:22:05Z",
  "cycle_id": "C042",
  "hypothesis_id": "H3",
  "op": "EmbedOp",
  "table": "main.marketing.tiktok_posts_q1_2026",
  "estimated_max_usd": 0.42,
  "actual_usd": 0.08,
  "compute": {"backend": "jobs", "workers": 2, "runtime_sec": 142},
  "rows_scanned": 12800000,
  "approved": {"budget_usd": 1.00, "max_runtime_sec": 1800, "max_rows": 100000},
  "outcome": "success"
}
```

Seeds next-cycle defaults via:
- **p95 × 1.2** for budget per op-type per table.
- **Historical max approved** surfaced in prompt copy ("last approved $X for this table").
- **Per-table scanned-bytes** short-circuits `EXPLAIN`-based estimates.
- **Cold-start latency** per `warehouse_id` makes ETAs honest.
- **Grant patterns** resolved for `DatabricksPermissionError` events → onboarding prompt pre-suggests them.

### 13.2 Open questions

- **Service principal migration** — daemon can reload on `SIGHUP` (§5.5.7); full OAuth M2M flow deferred to Phase 4.
- **Cost attribution at the workspace level** — tags on Jobs work today; SQL needs `SET STATEMENT_TAG` or comment-based attribution; confirm with workspace admin which is canonical.
- **Multi-workspace cycles** — memory is per-`workspace_host_fingerprint`; design allows it but no current use case.
- **UC volume lifecycle** — we assume user provides; what about shared volumes with retention policies clobbering AHVS artifacts? Deferred — surface as a friction log entry if it happens.

## 14. Acceptance criteria for the vertical slice

- `ahvs data_analyst --data-source databricks --table <T>` produces the same report structure as the local path, on a table of at least 1M rows.
- Cold-start end-to-end (profile + class balance + 10k-row stratified sample) completes on a warm serverless warehouse within 2 minutes (honest about cold-start variance; number from actual measurement, not a pre-commit claim).
- No row data leaves Databricks except aggregates + the explicitly-requested sample rows (≤ `--databricks-max-sql-rows`).
- Total cycle cost ≤ user-approved budget, verified against daemon ledger and Databricks billing (sampled).
- Unit tests per §11.1 all pass in CI (contract test gated out).
- Proxy daemon: zeroization verified via `ctypes.string_at` probe; SO_PEERCRED reject verified in subprocess test; budget race passes invariant.
- `databricks_approvals.jsonl` populated after first cycle; second cycle's prompt cites the first's approval.
- `docs/ahvs/databricks.md` published with setup walkthrough and top-5 errors.

---

## Appendix A — Why not Databricks Connect v2

Earlier draft included a Databricks Connect adapter. Dropped for three reasons:

1. **Requires a pre-provisioned all-purpose cluster** — conflicts with the goal of "no manual cluster management."
2. **DBR version pinning** — client library must match cluster DBR; drift causes obscure failures.
3. **Overlaps with Jobs backend** — anything Databricks Connect does, a serverless Python job can do, with cleaner lifecycle and no version coupling.

Revisit only if a user need for interactive local Spark DataFrame manipulation emerges.

## Appendix B — References

- `databricks-sql-connector` — PyPI (`pip install databricks-sql-connector`); pinned in the daemon only.
- `databricks-sdk` — PyPI (`pip install databricks-sdk`); pinned in the daemon only.
- Serverless jobs compute — `POST /api/2.1/jobs/runs/submit` with `environments` block (shape in §4.3).
- SQL Statements API — `POST /api/2.0/sql/statements`, `wait_timeout` ≤ 50s.
- Unity Catalog volumes — user-provisioned; daemon verifies writability.
- Python `ctypes.memset` for bytearray zeroization — practical ceiling on CPython credential hygiene without native extensions.

## Appendix C — Review consolidation

v2 incorporates findings from two reviews of v1 on 2026-04-23:

**Red-team review (adversarial pass):** 6 🔴 blockers + 8 🟡 issues. Key findings addressed in v2:
- Cooperative enforcement → daemon enforces server-side (§5.5).
- `run_query → DataFrame` privacy contradiction → removed from interface (§4.2).
- Budget race → per-`(cycle_id, hypothesis_id)` ledger inside daemon, lock-guarded (§5.5.4).
- Orphan jobs → inflight tracking + startup reconciliation (§5.5.5, §8.4).
- SQL injection → strict identifier grammar, daemon-side (§5.4).
- TTY-blocking prompts in non-interactive mode → pre-approved budget path + explicit no-TTY abort (§8.3).
- `--databricks-yes` → removed (too dangerous under cooperative security).

**Constructive review (Codex second pass):** complementary, API-level and architectural findings:
- Operation-oriented decomposition → §4.
- Serverless `environments` block shape → §4.3 payload sketch.
- UC volume (not Workspace API, not DBFS) for script upload → §4.3.
- `MOD(abs(xxhash64(id, seed)), 100)` for deterministic splits → §4.2 SplitOp.
- `information_schema.table_storage_usage` for scanned-bytes → §8.2.
- `jobs/runs/get-output` for serverless logs → §4.3.
- SQL comment + job tags for correlation → §10.3.
- `databricks_approvals.jsonl` schema → §13.1.
- `eval_spec.yaml` contract for Databricks-hosted eval → §7.5.
- Proxy daemon lifecycle, zeroization, SO_PEERCRED, reload → §5.5.
- VCR fixtures + opt-in contract tests + property-based grammar + subprocess SO_PEERCRED test + budget race test → §11.1.
- UC row filters / column masks awareness → §12.
- PAT rotation via SIGHUP → §5.5.7.

Both reviews converged on the privacy contradiction (finding #2) and cooperative-enforcement weakness (finding #3) as the deepest issues. Both are resolved by the daemon architecture in §5.5.
