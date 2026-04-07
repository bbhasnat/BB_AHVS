"""AHVS CLI — run an adaptive hypothesis-validation cycle."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import cast


def _parse_hypothesis_ops(args: argparse.Namespace) -> list[dict]:
    """Parse --add/--edit/--insert-hypothesis flags into a list of ops."""
    ops: list[dict] = []
    for raw in getattr(args, "add_hypotheses", []) or []:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"Error: --add-hypothesis: invalid JSON: {e}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(data, dict) or "description" not in data:
            print("Error: --add-hypothesis requires at least 'description' field", file=sys.stderr)
            sys.exit(1)
        data["op"] = "add"
        data.setdefault("type", "code_change")
        ops.append(data)

    for raw in getattr(args, "edit_hypotheses", []) or []:
        m = re.match(r"^(H\d+)\s*:\s*(.+)$", raw, re.IGNORECASE | re.DOTALL)
        if not m:
            print(f"Error: --edit-hypothesis: expected 'H<N>:{{...}}', got: {raw!r}", file=sys.stderr)
            sys.exit(1)
        hyp_id = m.group(1).upper()
        try:
            fields = json.loads(m.group(2))
        except json.JSONDecodeError as e:
            print(f"Error: --edit-hypothesis: invalid JSON after {hyp_id}: {e}", file=sys.stderr)
            sys.exit(1)
        ops.append({"op": "edit", "id": hyp_id, "fields": fields})

    for raw in getattr(args, "insert_hypotheses", []) or []:
        m = re.match(r"^(\d+)\s*:\s*(.+)$", raw, re.DOTALL)
        if not m:
            print(f"Error: --insert-hypothesis: expected 'POS:{{...}}', got: {raw!r}", file=sys.stderr)
            sys.exit(1)
        pos = int(m.group(1))
        try:
            data = json.loads(m.group(2))
        except json.JSONDecodeError as e:
            print(f"Error: --insert-hypothesis: invalid JSON: {e}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(data, dict) or "description" not in data:
            print("Error: --insert-hypothesis requires at least 'description' field", file=sys.stderr)
            sys.exit(1)
        data["op"] = "insert"
        data["position"] = pos
        data.setdefault("type", "code_change")
        ops.append(data)

    return ops


def cmd_ahvs(args: argparse.Namespace) -> int:
    """Run one AHVS hypothesis-validation cycle."""
    from pathlib import Path as _Path
    from ahvs.config import AHVSConfig
    from ahvs.runner import execute_ahvs_cycle, read_ahvs_checkpoint
    from ahvs.stages import AHVSStage, StageStatus

    from ahvs.registry import resolve as _resolve_repo

    resolved = _resolve_repo(args.repo)
    if resolved is None:
        print(
            f"Error: '{args.repo}' is not a valid path and not a registered AHVS repo.\n"
            f"  Hint: run 'ahvs list' to see registered repos, or pass a full path.",
            file=sys.stderr,
        )
        return 1
    repo_path = resolved

    # Resolve domain pack paths (--domain overrides --prompts/--skill-registry)
    _prompts_path = _Path(args.prompts).resolve() if args.prompts else None
    _skills_path = _Path(args.skill_registry).resolve() if args.skill_registry else None
    domain = getattr(args, "domain", None)
    if domain and domain != "llm":
        _pack_dir = _Path(__file__).resolve().parent / "domain_packs"
        if not _prompts_path:
            _prompts_path = _pack_dir / f"{domain}_prompts.yaml"
        if not _skills_path:
            _skills_path = _pack_dir / f"{domain}_skills.yaml"

    hypothesis_ops = _parse_hypothesis_ops(args)

    config = AHVSConfig(
        repo_path=repo_path,
        question=args.question,
        max_hypotheses=args.max_hypotheses,
        regression_guard_path=_Path(args.regression_guard).resolve() if args.regression_guard else None,
        apply_best=getattr(args, "apply_best", False),
        skill_registry_path=_skills_path,
        prompts_override_path=_prompts_path,
        llm_provider=args.provider or "anthropic",
        llm_base_url=getattr(args, "base_url", "") or "",
        llm_model=args.model or "claude-opus-4-6",
        llm_api_key_env=args.api_key_env or "ANTHROPIC_API_KEY",
        run_dir=_Path(args.run_dir).resolve() if args.run_dir else None,
        acp_agent=getattr(args, "acp_agent", "claude") or "claude",
        acpx_command=getattr(args, "acpx_command", "") or "",
        acp_session_name=getattr(args, "acp_session_name", "ahvs") or "ahvs",
        acp_timeout_sec=getattr(args, "acp_timeout_sec", 1800) or 1800,
        eval_timeout_sec=getattr(args, "eval_timeout_sec", 600) or 600,
        cache_enabled=not getattr(args, "no_cache", False),
        hypothesis_ops=hypothesis_ops,
    )

    from_stage: AHVSStage | None = None
    if args.from_stage:
        try:
            from_stage = AHVSStage[args.from_stage.upper()]
        except KeyError:
            valid = [s.name for s in AHVSStage]
            print(
                f"Error: unknown stage '{args.from_stage}'. Valid: {', '.join(valid)}",
                file=sys.stderr,
            )
            return 1
    elif args.resume:
        # Auto-detect latest cycle dir if --run-dir was not provided
        if not args.run_dir:
            cycles_root = repo_path / ".ahvs" / "cycles"
            if cycles_root.is_dir():
                cycle_dirs = sorted(
                    [d for d in cycles_root.iterdir() if d.is_dir()],
                    key=lambda d: d.name,
                    reverse=True,
                )
                if cycle_dirs:
                    config.run_dir = cycle_dirs[0]
                    print(f"[AHVS] Auto-detected latest cycle: {cycle_dirs[0].name}")
                else:
                    print(
                        "Error: --resume requires a previous cycle, but no cycles found "
                        f"under {cycles_root}",
                        file=sys.stderr,
                    )
                    return 1
            else:
                print(
                    "Error: --resume requires a previous cycle, but "
                    f"{cycles_root} does not exist. Run a cycle first or pass --run-dir.",
                    file=sys.stderr,
                )
                return 1

        resumed = read_ahvs_checkpoint(config.run_dir)
        if resumed is not None:
            # Advance one stage past the last completed checkpoint
            from ahvs.stages import AHVS_NEXT_STAGE
            nxt = AHVS_NEXT_STAGE.get(resumed)
            if nxt is not None:
                from_stage = nxt
                print(f"[AHVS] Resuming from checkpoint: {nxt.name}")
            else:
                print("[AHVS] Checkpoint shows cycle already complete.")
                return 0
        else:
            print(
                f"Error: no checkpoint found in {config.run_dir}. "
                "Cannot resume — start a new cycle instead.",
                file=sys.stderr,
            )
            return 1

    # Pre-write selection.json if --selection was provided
    if getattr(args, "selection", None):
        sel_ids = [
            s.strip().upper()
            for s in re.split(r"[,\s]+", args.selection)
            if s.strip()
        ]
        if sel_ids:
            sel_data = {
                "selected": sel_ids,
                "rationale": "CLI --selection flag",
                "approved_by": "caller",
            }
            config.run_dir.mkdir(parents=True, exist_ok=True)
            (config.run_dir / "selection.json").write_text(
                json.dumps(sel_data, indent=2), encoding="utf-8"
            )
            print(f"[AHVS] Pre-specified selection: {', '.join(sel_ids)}")

    until_stage: AHVSStage | None = None
    if getattr(args, "until_stage", None):
        try:
            until_stage = AHVSStage[args.until_stage.upper()]
        except KeyError:
            valid = [s.name for s in AHVSStage]
            print(
                f"Error: unknown stage '{args.until_stage}'. Valid: {', '.join(valid)}",
                file=sys.stderr,
            )
            return 1

    results = execute_ahvs_cycle(
        config,
        auto_approve=args.auto_approve,
        from_stage=from_stage,
        until_stage=until_stage,
    )

    failed = [r for r in results if r.status != StageStatus.DONE]
    if failed:
        return 1

    # Post-cycle: apply best hypothesis if requested
    if config.apply_best:
        return _apply_best(config)

    return 0


def _apply_best(config: "AHVSConfig") -> int:
    """Apply the best hypothesis patch from a completed cycle.

    Reads cycle_summary.json, applies the best patch via ``git apply``,
    and updates baseline_metric.json with the new metric value.
    """
    import json
    import subprocess
    from datetime import datetime, timezone

    summary_path = config.run_dir / "cycle_summary.json"
    if not summary_path.exists():
        print(f"Error: cycle_summary.json not found at {summary_path}", file=sys.stderr)
        return 1

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    best_id = summary.get("best_hypothesis")
    if not best_id:
        print("[AHVS] No improving hypothesis this cycle — nothing to apply.")
        return 0

    patch_rel = summary.get("kept_patch")
    if not patch_rel:
        print(
            f"[AHVS] Best hypothesis {best_id} has no patch file — "
            "cannot apply (was it sandbox-only?).",
            file=sys.stderr,
        )
        return 1

    patch_path = config.run_dir / patch_rel
    if not patch_path.exists():
        print(f"Error: patch file not found: {patch_path}", file=sys.stderr)
        return 1

    # Dry-run check, then apply
    try:
        subprocess.run(
            ["git", "apply", "--check", str(patch_path)],
            cwd=str(config.repo_path),
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "apply", str(patch_path)],
            cwd=str(config.repo_path),
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        print(
            f"Error: git apply failed for {patch_path}:\n{exc.stderr}",
            file=sys.stderr,
        )
        return 1

    print(f"[AHVS] Applied patch from {best_id}: {patch_path.name}")

    # Update baseline_metric.json
    best_metric = summary.get("best_metric_value")
    if best_metric is not None:
        baseline_path = config.baseline_path
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        else:
            baseline = {}

        metric_key = baseline.get("primary_metric", "")
        if metric_key:
            baseline[metric_key] = best_metric
        baseline["recorded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # Record provenance: HEAD is the pre-patch commit because
        # --apply-best only runs git-apply (no auto-commit).  We record
        # both the base commit and the patch so later cycles can verify.
        try:
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(config.repo_path),
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            baseline["commit"] = head
            baseline["commit_note"] = (
                "pre-patch commit; working tree includes unapplied patch from "
                f"hypothesis {best_id} — commit the changes to update this field"
            )
        except subprocess.CalledProcessError:
            pass

        baseline["applied_from_cycle"] = config.run_dir.name
        baseline["applied_hypothesis"] = best_id
        baseline["applied_patch"] = patch_rel

        baseline_path.write_text(
            json.dumps(baseline, indent=2), encoding="utf-8"
        )
        print(
            f"[AHVS] Updated {baseline_path.name}: "
            f"{metric_key}={best_metric} (from {best_id}, cycle {config.run_dir.name})"
        )

    return 0


def _cmd_list_repos() -> int:
    """Print all registered AHVS repos."""
    from ahvs.registry import list_repos

    repos = list_repos()
    if not repos:
        print("No repos registered. Onboard a repo first with the ahvs_onboarding skill.")
        return 0

    for name, info in repos.items():
        metric = info.get("primary_metric", "?")
        value = info.get("baseline_value", "?")
        last = info.get("last_cycle", "none")
        path = info.get("path", "?")
        print(f"  {name:20s}  metric={metric}={value}  last_cycle={last}")
        print(f"  {'':20s}  path={path}")
    return 0


def _cmd_unregister(name: str) -> int:
    """Remove a repo from the AHVS registry."""
    from ahvs.registry import unregister

    if unregister(name):
        print(f"Unregistered '{name}' from AHVS registry.")
        return 0
    print(f"'{name}' is not in the registry.", file=sys.stderr)
    return 1


def cmd_genesis(argv: list[str]) -> int:
    """Run the genesis subcommand — create a new project from data."""
    parser = argparse.ArgumentParser(
        prog="ahvs genesis",
        description="Genesis — bootstrap a new AHVS project from raw data",
    )
    parser.add_argument(
        "--problem", "-p", required=True,
        help="Natural language problem description (e.g. 'Classify customer emails by intent')",
    )
    parser.add_argument(
        "--data", "-d", required=True,
        help="Path to input data file (CSV, TSV, or Parquet)",
    )
    parser.add_argument(
        "--target-metric", "-m", default="f1_weighted",
        help="Metric to optimize (default: f1_weighted)",
    )
    parser.add_argument(
        "--output-dir", "-o", required=True,
        help="Output directory for the new project (REQUIRED — never auto-generated)",
    )
    parser.add_argument(
        "--solver", "-s", default=None,
        help="Solver name (default: auto-detect from problem description)",
    )
    parser.add_argument(
        "--mode", default="pipeline",
        choices=["pipeline", "agent"],
        help=(
            "Execution mode. 'pipeline' (default): deterministic, generates config/spec "
            "and calls the KD pipeline directly. 'agent': uses the KD Agent (claude-agent-sdk) "
            "to inspect data and drive all stages autonomously."
        ),
    )
    parser.add_argument(
        "--solver-registry",
        help="Path to custom solvers.yaml (default: built-in registry)",
    )
    parser.add_argument(
        "--classes", nargs="+",
        help="Classification classes (e.g. --classes positive negative neutral)",
    )
    parser.add_argument(
        "--input-column", default="text",
        help="Name of the text column in the data file (default: text)",
    )
    parser.add_argument(
        "--annotation-model", default=None,
        help="LLM model for annotation (default: gpt-4.1-mini). Never use gpt-4o.",
    )

    args = parser.parse_args(argv)

    from pathlib import Path as _Path
    from ahvs.genesis.registry import SolverRegistry
    from ahvs.genesis.router import ProblemRouter

    # Load solver
    registry = SolverRegistry(args.solver_registry)
    if args.solver:
        try:
            solver = registry.get(args.solver)
        except KeyError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    else:
        router = ProblemRouter(registry)
        solver_name = router.route(args.problem)
        solver = registry.get(solver_name)
        print(f"[Genesis] Auto-selected solver: {solver_name}")

    # Build config overrides
    overrides: dict = {"mode": args.mode}
    if args.classes:
        overrides["classes"] = args.classes
    if args.input_column != "text":
        overrides["input_column"] = args.input_column
    if args.annotation_model:
        overrides["annotation_model"] = args.annotation_model

    print(f"[Genesis] Problem: {args.problem}")
    print(f"[Genesis] Data: {args.data}")
    print(f"[Genesis] Mode: {args.mode}")
    print(f"[Genesis] Output: {args.output_dir}")
    print()

    # Run solver
    result = solver.solve(
        problem=args.problem,
        data_path=args.data,
        target_metric=args.target_metric,
        output_dir=args.output_dir,
        config_overrides=overrides,
    )

    if not result.success:
        print("Genesis FAILED:", file=sys.stderr)
        for err in result.errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    # Register the project
    from ahvs.registry import register as _register_repo
    metric_val = result.baseline_metric.get(args.target_metric)
    short_name = _register_repo(
        result.project_dir,
        primary_metric=args.target_metric,
        baseline_value=metric_val,
    )

    print(f"[Genesis] Success!")
    print(f"  Project:  {result.project_dir}")
    print(f"  Metric:   {args.target_metric} = {metric_val}")
    print(f"  Model:    {result.model_path or 'N/A'}")
    print(f"  Registry: registered as '{short_name}'")
    print()
    print("Next step — run AHVS optimization:")
    print(f"  ahvs --repo {short_name} --question 'improve {args.target_metric} to <target>'")

    return 0


def cmd_install(argv: list[str]) -> int:
    """Run the install subcommand."""
    parser = argparse.ArgumentParser(
        prog="ahvs install",
        description="Install AHVS skills globally and initialize ~/.ahvs/",
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force reinstall even if skills are up to date",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args(argv)

    from ahvs.installer import run_install
    return run_install(force=args.force, quiet=args.quiet)


def cmd_update(argv: list[str]) -> int:
    """Run the update subcommand."""
    parser = argparse.ArgumentParser(
        prog="ahvs update",
        description="Update AHVS skills to match the current package version",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args(argv)

    from ahvs.installer import run_update
    return run_update(quiet=args.quiet)


def cmd_uninstall(argv: list[str]) -> int:
    """Run the uninstall subcommand."""
    parser = argparse.ArgumentParser(
        prog="ahvs uninstall",
        description="Remove AHVS skills from ~/.claude/skills/",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args(argv)

    from ahvs.installer import run_uninstall
    return run_uninstall(quiet=args.quiet)


def main(argv: list[str] | None = None) -> int:
    # Intercept subcommands before main argparse
    effective_argv = argv if argv is not None else sys.argv[1:]
    if effective_argv and effective_argv[0] == "genesis":
        return cmd_genesis(effective_argv[1:])
    if effective_argv and effective_argv[0] == "install":
        return cmd_install(effective_argv[1:])
    if effective_argv and effective_argv[0] == "update":
        return cmd_update(effective_argv[1:])
    if effective_argv and effective_argv[0] == "uninstall":
        return cmd_uninstall(effective_argv[1:])

    parser = argparse.ArgumentParser(
        prog="ahvs",
        description="AHVS — Adaptive Hypothesis Validation System",
    )

    parser.add_argument(
        "--list-repos", action="store_true",
        help="List all registered AHVS repos and exit",
    )
    parser.add_argument(
        "--unregister",
        help="Remove a repo from the registry by name and exit",
    )
    parser.add_argument(
        "--repo", "-r",
        help="Path or registered short name of target repository (see --list-repos)",
    )
    parser.add_argument(
        "--question", "-q",
        help="Cycle question (e.g. 'How can we improve answer_relevance by 5%%?')",
    )
    parser.add_argument(
        "--max-hypotheses", type=int, default=3,
        help="Maximum hypotheses to generate per cycle (default: 3, hard cap: 5)",
    )
    parser.add_argument(
        "--max-lesson-cycles", type=int, default=5,
        help="Load lessons from last K complete cycles only (default: 5, 0 = unlimited)",
    )
    parser.add_argument(
        "--regression-guard",
        help="Path to regression guard shell script (optional)",
    )
    parser.add_argument(
        "--auto-approve", action="store_true",
        help="Skip interactive gate and run all generated hypotheses",
    )
    parser.add_argument(
        "--selection",
        help=(
            "Pre-specify which hypotheses to run (e.g. 'H1,H3'). "
            "For conversational/agent-driven mode: the caller writes "
            "selection.json into the cycle dir before the gate stage runs. "
            "This flag is a convenience shortcut that writes selection.json "
            "from the command line."
        ),
    )
    parser.add_argument(
        "--from-stage",
        help="Start from a specific stage (e.g. AHVS_HYPOTHESIS_GEN)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the last written checkpoint in the cycle directory",
    )
    parser.add_argument(
        "--domain",
        choices=["llm", "ml"],
        help=(
            "Load a domain pack that sets prompts and skills for the target domain. "
            "'llm' (default) uses LLM/RAG-focused prompts and skills. "
            "'ml' uses traditional ML-focused prompts (feature engineering, "
            "hyperparameter tuning, algorithm selection) and ML skill templates. "
            "Equivalent to --prompts + --skill-registry with the domain pack YAMLs."
        ),
    )
    parser.add_argument(
        "--skill-registry",
        help="Path to custom skill registry YAML file (optional)",
    )
    parser.add_argument(
        "--prompts",
        help="Path to AHVS prompts override YAML file (optional)",
    )
    parser.add_argument(
        "--model", default="claude-opus-4-6",
        help="LLM model ID (default: claude-opus-4-6)",
    )
    parser.add_argument(
        "--api-key-env", default="ANTHROPIC_API_KEY",
        help="Environment variable holding the LLM API key (default: ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--base-url", default="",
        help="Override LLM base URL (required for --provider openai-compatible)",
    )
    parser.add_argument(
        "--provider", default="anthropic",
        choices=["anthropic", "openai", "openai-compatible", "openrouter", "deepseek", "acp"],
        help="LLM provider for AHVS orchestration (default: anthropic). Use 'acp' for local agent (Claude Code, Codex)",
    )
    parser.add_argument(
        "--acp-agent", default="claude",
        help="ACP agent CLI name (default: claude). Only used with --provider acp",
    )
    parser.add_argument(
        "--acpx-command", default="",
        help="Path to acpx binary (auto-detected if omitted). Only used with --provider acp",
    )
    parser.add_argument(
        "--acp-session-name", default="ahvs",
        help="ACP session name (default: ahvs). Only used with --provider acp",
    )
    parser.add_argument(
        "--acp-timeout", type=int, default=1800, dest="acp_timeout_sec",
        help="ACP per-prompt timeout in seconds (default: 1800). Only used with --provider acp",
    )
    parser.add_argument(
        "--eval-timeout", type=int, default=600, dest="eval_timeout_sec",
        help="Timeout in seconds for eval_command execution (default: 600). Also settable via 'eval_timeout' in baseline_metric.json",
    )
    parser.add_argument(
        "--apply-best", action="store_true",
        help="Auto-apply the best improving hypothesis patch to the working tree and update baseline",
    )
    parser.add_argument(
        "--run-dir",
        help="Override cycle output directory (default: <repo>/.ahvs/cycles/<timestamp>)",
    )
    parser.add_argument(
        "--until-stage",
        help=(
            "Stop after this stage and exit (e.g. 'AHVS_HYPOTHESIS_GEN'). "
            "Useful for running only hypothesis generation, then resuming with "
            "--from-stage after GUI selection."
        ),
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable LLM response cache (also controllable via LLM_CACHE_ENABLED=false)",
    )

    # ── Hypothesis modification flags ────────────────────────────────────
    parser.add_argument(
        "--add-hypothesis", action="append", default=[], dest="add_hypotheses",
        metavar="JSON",
        help=(
            "Add a custom hypothesis (repeatable). JSON object with at least "
            "'type' and 'description'. Example: "
            """'{"type":"code_change","description":"Refactor tokenizer","rationale":"..."}' """
            "Appended after LLM-generated hypotheses with the next available ID."
        ),
    )
    parser.add_argument(
        "--edit-hypothesis", action="append", default=[], dest="edit_hypotheses",
        metavar="ID:JSON",
        help=(
            "Edit fields of a generated hypothesis (repeatable). Format: "
            "'H2:{\"description\":\"new desc\"}'. Merges the JSON fields into the "
            "existing hypothesis, leaving other fields unchanged."
        ),
    )
    parser.add_argument(
        "--insert-hypothesis", action="append", default=[], dest="insert_hypotheses",
        metavar="POS:JSON",
        help=(
            "Insert a custom hypothesis at a specific position (repeatable). "
            "Format: '2:{\"type\":\"code_change\",\"description\":\"...\"}'. "
            "1-indexed position; hypotheses after the insertion point are renumbered."
        ),
    )

    args = parser.parse_args(argv)

    # Short-circuit commands that don't need --repo / --question
    if args.list_repos:
        return _cmd_list_repos()
    if args.unregister:
        return _cmd_unregister(args.unregister)

    # Validate required args for cycle run
    if not args.repo:
        parser.error("--repo is required for running a cycle (or use --list-repos)")
    if not args.question:
        parser.error("--question is required for running a cycle")

    return cmd_ahvs(args)


if __name__ == "__main__":
    sys.exit(main())
