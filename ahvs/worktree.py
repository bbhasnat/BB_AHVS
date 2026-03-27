"""HypothesisWorktree — manages a git worktree for one AHVS hypothesis.

Each hypothesis gets a detached worktree at repo HEAD where generated files
are applied, the eval_command is run, and a diff/patch is captured.
The worktree is cleaned up unless it produced the best improvement.
"""

from __future__ import annotations

import ast
import logging
import os
import signal
import subprocess
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared path-safety utility (used by both worktree writes and tool_runs writes)
# ---------------------------------------------------------------------------


def validate_safe_relpath(relpath: str, root: Path) -> None:
    """Reject *relpath* if it would escape *root* when joined.

    Checks performed (in order):
    1. Absolute path (e.g. ``/tmp/evil``)
    2. ``..`` component anywhere in the path
    3. Resolved destination is not a descendant of *root*
       (catches symlink escapes and edge cases)

    Uses :meth:`Path.is_relative_to` for the containment check — immune
    to the string-prefix false-positive where ``/tmp/wt2/f`` appears to
    be inside ``/tmp/wt``.

    Raises:
        ValueError: with a descriptive message when the path is unsafe.
    """
    from pathlib import PurePosixPath

    pure = PurePosixPath(relpath)

    if pure.is_absolute():
        raise ValueError(
            f"Refusing absolute path from Claude Code output: {relpath!r}"
        )

    if ".." in pure.parts:
        raise ValueError(
            f"Refusing path with '..' traversal from Claude Code output: {relpath!r}"
        )

    dest_resolved = (root / relpath).resolve()
    root_resolved = root.resolve()
    if not dest_resolved.is_relative_to(root_resolved):
        raise ValueError(
            f"Path escapes boundary: {relpath!r} "
            f"resolves to {dest_resolved}, outside {root_resolved}"
        )


# ---------------------------------------------------------------------------
# AST-based partial-file splicing
# ---------------------------------------------------------------------------


def splice_functions(original_src: str, partial_src: str) -> str:
    """Merge *partial_src* (containing only modified definitions) into
    *original_src*, returning the full updated file.

    The Claude Code is instructed to output only the functions, classes, and
    constants it modifies — not the entire file.  This function splices those
    fragments into the original source by:

    1. Parsing both sources with ``ast.parse``.
    2. For each top-level definition (``FunctionDef``, ``AsyncFunctionDef``,
       ``ClassDef``, ``Assign``, ``AnnAssign``) in *partial_src*, finding the
       matching definition in *original_src* (by name) and replacing the
       original's source lines with the partial's source lines.
    3. Appending any NEW definitions (not present in the original) at the end.
    4. Collecting any new ``import`` / ``from ... import`` lines from the
       partial and prepending them after the original's existing imports.

    Falls back to returning *partial_src* unchanged if either source fails to
    parse (safety valve — worst case is the old full-file behaviour).
    """
    try:
        orig_tree = ast.parse(original_src)
    except SyntaxError:
        logger.warning("splice_functions: original source has SyntaxError — "
                       "returning partial as-is")
        return partial_src

    try:
        part_tree = ast.parse(partial_src)
    except SyntaxError:
        logger.warning("splice_functions: partial source has SyntaxError — "
                       "returning original unchanged")
        return original_src

    orig_lines = original_src.splitlines(keepends=True)
    part_lines = partial_src.splitlines(keepends=True)

    # --- Build index of original top-level definitions ---
    # Maps name → (start_line_0idx, end_line_0idx) for each top-level node.
    def _node_name(node: ast.AST) -> str | None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return node.name
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    return target.id
            return None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            return node.target.id
        return None

    def _node_end(node: ast.AST, source_lines: list[str]) -> int:
        """Return the 0-indexed end line of *node*.  Uses end_lineno if
        available (Python 3.8+), otherwise scans for the next non-blank,
        non-indented line."""
        if hasattr(node, "end_lineno") and node.end_lineno is not None:
            return node.end_lineno - 1  # 0-indexed
        # Fallback: scan forward until we hit a line at the same or lesser
        # indentation (or EOF).
        start = node.lineno - 1
        base_indent = len(source_lines[start]) - len(source_lines[start].lstrip())
        end = start
        for i in range(start + 1, len(source_lines)):
            line = source_lines[i]
            stripped = line.strip()
            if not stripped:
                end = i
                continue
            indent = len(line) - len(line.lstrip())
            if indent <= base_indent and stripped:
                break
            end = i
        return end

    orig_index: dict[str, tuple[int, int]] = {}
    for node in orig_tree.body:
        name = _node_name(node)
        if name is not None:
            orig_index[name] = (node.lineno - 1, _node_end(node, orig_lines))

    # --- Collect partial definitions and imports ---
    partial_defs: list[tuple[str, int, int]] = []  # (name, start, end) in part_lines
    new_import_lines: list[str] = []

    # Collect existing import lines from original for dedup
    orig_import_set: set[str] = set()
    for node in orig_tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for ln in range(node.lineno - 1, _node_end(node, orig_lines) + 1):
                if ln < len(orig_lines):
                    orig_import_set.add(orig_lines[ln].strip())

    for node in part_tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for ln in range(node.lineno - 1, _node_end(node, part_lines) + 1):
                if ln < len(part_lines):
                    line_stripped = part_lines[ln].strip()
                    if line_stripped and line_stripped not in orig_import_set:
                        new_import_lines.append(part_lines[ln])
            continue

        name = _node_name(node)
        if name is not None:
            partial_defs.append((name, node.lineno - 1, _node_end(node, part_lines)))

    # --- Apply replacements in reverse order (to preserve line offsets) ---
    # Sort by position in original (or append position) so we can apply
    # replacements bottom-up.
    replacements: list[tuple[int, int, list[str]]] = []  # (start, end, new_lines)
    appends: list[str] = []

    for name, p_start, p_end in partial_defs:
        new_lines = part_lines[p_start:p_end + 1]
        if name in orig_index:
            o_start, o_end = orig_index[name]
            replacements.append((o_start, o_end, new_lines))
        else:
            # New definition — append at end
            appends.extend(["\n", "\n"])
            appends.extend(new_lines)

    # Apply replacements bottom-up so line offsets remain valid
    replacements.sort(key=lambda r: r[0], reverse=True)
    result_lines = list(orig_lines)
    for o_start, o_end, new_lines in replacements:
        result_lines[o_start:o_end + 1] = new_lines

    # Prepend new imports after the last import block in the original
    if new_import_lines:
        last_import_line = 0
        for node in orig_tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                last_import_line = _node_end(node, orig_lines) + 1
        for i, imp_line in enumerate(new_import_lines):
            result_lines.insert(last_import_line + i, imp_line)

    # Append new definitions at the end
    if appends:
        result_lines.extend(appends)

    merged = "".join(result_lines)

    # Sanity check: the merged result should parse
    try:
        compile(merged, "<splice_result>", "exec")
    except SyntaxError as exc:
        logger.warning(
            "splice_functions: merged result has SyntaxError (%s) — "
            "returning original unchanged", exc.msg,
        )
        return original_src

    return merged


@dataclass(frozen=True)
class EvalResult:
    """Outcome of running eval_command inside a hypothesis worktree."""

    returncode: int
    stdout: str
    stderr: str
    elapsed_sec: float
    timed_out: bool = False


class HypothesisWorktree:
    """Lifecycle manager for a single hypothesis git worktree."""

    def __init__(self, repo_path: Path, worktree_path: Path) -> None:
        self.repo_path = repo_path
        self.worktree_path = worktree_path
        self._created = False
        # eval_cwd is set in create() to handle the case where repo_path is a
        # subdirectory of the git root (the worktree is always rooted at the
        # git root, so eval_command must cd into the subdir before running).
        self.eval_cwd: Path = worktree_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _terminate_eval_process_group(
        self,
        proc: subprocess.Popen[str],
        *,
        grace_sec: float = 3.0,
    ) -> None:
        """Best-effort teardown of the full eval process group.

        Eval commands often launch child Python/DataLoader workers. Killing only
        the shell parent can leave descendants holding CPU/GPU memory. By
        running evals in a dedicated session and terminating the entire process
        group, AHVS avoids leaking resources between hypotheses.
        """
        if proc.poll() is not None:
            return

        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return

        deadline = time.monotonic() + grace_sec
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.1)

        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            return

        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Timed out waiting for eval process group %s to exit after SIGKILL",
                proc.pid,
            )

    def create(self) -> None:
        """Create a detached worktree at repo HEAD.

        When ``repo_path`` is a subdirectory of the git root (e.g.
        ``monorepo/autoqa``), the worktree is created at the git root level
        and ``eval_cwd`` is set to the corresponding subdir inside the
        worktree so that eval commands run from the correct directory.

        Raises RuntimeError with a diagnostic message if the worktree cannot
        be created or the expected subdir is missing after checkout.
        """
        # Determine git root *before* creating the worktree so we can give a
        # clear diagnostic if repo_path is not inside a git repo.
        git_root_result = self._run_git(
            ["rev-parse", "--show-toplevel"],
            cwd=self.repo_path,
        )
        if git_root_result is None or git_root_result.returncode != 0:
            raise RuntimeError(
                f"Cannot determine git root for repo_path={self.repo_path}. "
                f"Is it inside a git repository?"
            )
        git_root = Path(git_root_result.stdout.strip()).resolve()
        repo_resolved = self.repo_path.resolve()
        is_subdir = repo_resolved != git_root
        subdir = None

        if is_subdir:
            try:
                subdir = repo_resolved.relative_to(git_root)
            except ValueError:
                raise RuntimeError(
                    f"repo_path {repo_resolved} is not under git root {git_root}. "
                    f"This should not happen — check your --repo argument."
                )
            logger.info(
                "repo_path is a git subdir: git_root=%s, subdir=%s",
                git_root, subdir,
            )

        self.worktree_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove stale worktree from a previous run at the same path.
        # This happens when a hypothesis is re-run after an earlier failure.
        if self.worktree_path.exists():
            logger.info(
                "Removing stale worktree at %s before re-creating",
                self.worktree_path,
            )
            remove_result = self._run_git(
                ["worktree", "remove", "--force", str(self.worktree_path)],
                cwd=self.repo_path,
            )
            if remove_result is None or remove_result.returncode != 0:
                # git worktree remove failed — try pruning and removing dir
                logger.warning(
                    "git worktree remove failed (%s), pruning and removing dir",
                    remove_result.stderr if remove_result else "unknown",
                )
                self._run_git(["worktree", "prune"], cwd=self.repo_path)
                import shutil
                shutil.rmtree(self.worktree_path, ignore_errors=True)

        result = self._run_git(
            ["worktree", "add", "--detach", str(self.worktree_path)],
            cwd=self.repo_path,
        )
        if result is None or result.returncode != 0:
            stderr = result.stderr if result else "unknown error"
            raise RuntimeError(f"Failed to create worktree: {stderr}")
        self._created = True
        logger.info("Created worktree at %s", self.worktree_path)

        # Set eval_cwd to the subdir within the worktree
        if subdir is not None:
            self.eval_cwd = self.worktree_path / subdir
            logger.info("eval_cwd set to subdir: %s", self.eval_cwd)

        # Verify eval_cwd exists after checkout.  If it's missing the worktree
        # was created but the expected subdir is absent — surface this clearly
        # rather than letting it fail with a confusing ENOENT later.
        if not self.eval_cwd.exists():
            raise RuntimeError(
                f"Worktree created at {self.worktree_path} but expected "
                f"eval_cwd {self.eval_cwd} does not exist.  "
                f"Git root: {git_root}, subdir: {subdir}. "
                "Check that the repo subdir is tracked on the current HEAD "
                "and that Claude Code did not delete it during generation."
            )

        # Symlink untracked data directories (checkpoints, ground-truth,
        # build artifacts) from the live repo into the worktree so that
        # eval commands have access to the same data files.  Git worktrees
        # only contain tracked files — without this, evals that read
        # gitignored data (e.g. parquet checkpoints) will crash.
        self._symlink_untracked_dirs(git_root, self.worktree_path)

    def read_file(self, relpath: str) -> str | None:
        """Read a file from the worktree by repo-relative path.

        Returns None if the file does not exist.
        """
        dest = self._resolve_dest(relpath)
        if dest is not None and dest.exists():
            return dest.read_text(encoding="utf-8")
        return None

    def apply_files(
        self,
        files: dict[str, str],
        *,
        splice: bool = False,
    ) -> list[Path]:
        """Write Claude Code-generated files into the worktree.

        *files* maps repo-relative paths to file contents.  When repo_path is
        a subdirectory of the git root, paths are resolved relative to
        ``eval_cwd`` (the repo subdir within the worktree) so that
        Claude Code-generated paths like ``src/autoqa/parsing.py`` land at
        ``{worktree}/{repo_subdir}/src/autoqa/parsing.py`` rather than the
        wrong location at the worktree root.

        When Claude Code generates a bare filename (e.g. ``parsing.py``)
        that doesn't match an existing file at ``{eval_cwd}/parsing.py`` but
        DOES match exactly one existing file deeper in the tree (e.g.
        ``{eval_cwd}/src/autoqa/parsing.py``), the file is written to the
        existing location instead.  This handles the common case where the
        Claude Code strips directory prefixes from filenames.

        If *splice* is True, the content is treated as **partial output**
        containing only modified functions/classes/constants.  The partial
        content is merged into the existing file via :func:`splice_functions`
        rather than replacing it wholesale.  This avoids LLM output
        truncation issues when modifying large files.

        Returns list of absolute paths written.

        Raises ValueError if any path would escape the repo boundary
        (absolute paths, ``..`` traversal, or symlink escape).
        """
        written: list[Path] = []
        base = self.eval_cwd
        base_resolved = base.resolve()

        # Pre-build an index of existing .py files for smart matching
        existing_py: dict[str, list[Path]] | None = None

        for relpath, content in files.items():
            self._validate_relpath(relpath, base_resolved)
            dest = (base / relpath).resolve()

            # Smart matching: if dest doesn't exist but there's exactly one
            # existing file with the same basename deeper in the tree, use that.
            if not dest.exists():
                basename = Path(relpath).name
                if existing_py is None:
                    # Build index lazily on first miss
                    existing_py = {}
                    for p in base_resolved.rglob("*.py"):
                        if "__pycache__" not in p.parts:
                            existing_py.setdefault(p.name, []).append(p)
                    # Also index .yaml and .json files
                    for ext in ("*.yaml", "*.yml", "*.json"):
                        for p in base_resolved.rglob(ext):
                            if "__pycache__" not in p.parts:
                                existing_py.setdefault(p.name, []).append(p)

                candidates = existing_py.get(basename, [])
                if len(candidates) == 1:
                    matched = candidates[0]
                    logger.info(
                        "apply_files: remapped %r → %s (smart match — "
                        "Claude Code used bare filename instead of full path)",
                        relpath, matched.relative_to(base_resolved),
                    )
                    dest = matched
                elif len(candidates) > 1:
                    logger.warning(
                        "apply_files: %r has %d matches in repo — "
                        "writing to literal path (ambiguous): %s",
                        relpath, len(candidates),
                        [str(c.relative_to(base_resolved)) for c in candidates],
                    )

            # Splice partial output into existing file when requested
            if splice and dest.exists() and relpath.endswith(".py"):
                original_src = dest.read_text(encoding="utf-8")
                content = splice_functions(original_src, content)
                logger.info(
                    "apply_files: spliced partial output into %s "
                    "(%d→%d lines)",
                    relpath,
                    original_src.count("\n") + 1,
                    content.count("\n") + 1,
                )

            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            written.append(dest)
        return written

    def _resolve_dest(self, relpath: str) -> Path | None:
        """Resolve a repo-relative path to an absolute worktree path,
        applying smart matching for bare filenames."""
        base = self.eval_cwd
        base_resolved = base.resolve()
        self._validate_relpath(relpath, base_resolved)
        dest = (base / relpath).resolve()
        if dest.exists():
            return dest
        # Try smart match
        basename = Path(relpath).name
        candidates = [
            p for p in base_resolved.rglob(f"*/{basename}")
            if "__pycache__" not in p.parts
        ]
        if len(candidates) == 1:
            return candidates[0]
        return dest  # Return the literal path even if it doesn't exist

    @staticmethod
    def _validate_relpath(relpath: str, worktree_root: Path) -> None:
        """Reject paths that escape the worktree boundary.

        Delegates to the module-level :func:`validate_safe_relpath` so
        the same logic is reused for ``tool_runs`` writes in the executor.
        """
        validate_safe_relpath(relpath, worktree_root)

    def run_eval_command(
        self, cmd: str, timeout: int = 300
    ) -> EvalResult:
        """Run *cmd* (shell) inside the worktree and return the result.

        If *cmd* starts with ``cd /absolute/path && ...`` or
        ``cd /absolute/path ; ...``, the ``cd`` prefix is stripped because
        ``eval_cwd`` already points to the correct directory inside the
        worktree.  An absolute ``cd`` would escape the worktree and run
        against the live repo — defeating isolation entirely.
        """
        import re

        # Strip leading 'cd /absolute-path && ' or 'cd /absolute-path ; '
        # to prevent the shell from escaping the worktree.  eval_cwd already
        # handles the correct directory context.
        m = re.match(r'^cd\s+(/\S+)\s*(&&|;)\s*', cmd)
        if m:
            logger.info(
                "run_eval_command: stripped 'cd %s' from command to prevent "
                "worktree escape — running from eval_cwd=%s instead",
                m.group(1), self.eval_cwd,
            )
            cmd = cmd[m.end():]

        if not self.eval_cwd.exists():
            msg = (
                f"eval_cwd does not exist: {self.eval_cwd}. "
                f"worktree_path={self.worktree_path}. "
                "The git worktree may not have checked out the expected subdir, "
                "or Claude Code may have deleted it."
            )
            logger.error(msg)
            return EvalResult(
                returncode=-1,
                stdout="",
                stderr=msg,
                elapsed_sec=0.0,
            )
        t0 = time.monotonic()
        try:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                cwd=str(self.eval_cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            stdout, stderr = proc.communicate(timeout=timeout)
            elapsed = time.monotonic() - t0
            # Cap captured output to avoid unbounded memory from chatty evals
            _MAX_OUTPUT = 1_000_000  # 1 MB
            return EvalResult(
                returncode=proc.returncode,
                stdout=stdout[-_MAX_OUTPUT:] if len(stdout) > _MAX_OUTPUT else stdout,
                stderr=stderr[-_MAX_OUTPUT:] if len(stderr) > _MAX_OUTPUT else stderr,
                elapsed_sec=round(elapsed, 2),
            )
        except subprocess.TimeoutExpired as exc:
            self._terminate_eval_process_group(proc)
            elapsed = time.monotonic() - t0
            stderr = f"eval_command timed out after {timeout}s"
            if exc.stderr:
                stderr += f"\n\nPartial stderr:\n{exc.stderr[:2000]}"
            return EvalResult(
                returncode=-1,
                stdout=exc.stdout or "",
                stderr=stderr,
                elapsed_sec=round(elapsed, 2),
                timed_out=True,
            )
        except Exception:
            if "proc" in locals():
                self._terminate_eval_process_group(proc)
            raise

    def capture_diff(self) -> str:
        """Return `git diff` output from the worktree (staged + unstaged)."""
        # Stage everything so diff captures new files too
        self._run_git(["add", "-A"], cwd=self.worktree_path)
        result = self._run_git(
            ["diff", "--cached"],
            cwd=self.worktree_path,
        )
        if result is None:
            return ""
        return result.stdout

    def save_patch(self, dest: Path) -> Path:
        """Write the worktree diff to *dest* as a .patch file."""
        diff = self.capture_diff()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(diff, encoding="utf-8")
        logger.info("Saved patch (%d bytes) to %s", len(diff), dest)
        return dest

    def cleanup(self) -> None:
        """Remove the worktree from the main repo."""
        if not self._created:
            return
        result = self._run_git(
            ["worktree", "remove", "--force", str(self.worktree_path)],
            cwd=self.repo_path,
        )
        if result and result.returncode == 0:
            logger.info("Removed worktree %s", self.worktree_path)
        else:
            logger.warning(
                "Failed to remove worktree %s — manual cleanup may be needed",
                self.worktree_path,
            )
        self._created = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _symlink_untracked_dirs(live_root: Path, wt_root: Path) -> None:
        """Symlink untracked directories from the live repo into the worktree.

        Git worktrees only contain tracked files.  Data directories that are
        gitignored (checkpoints, ground-truth parquets, build artefacts) won't
        exist in the worktree, causing eval commands to fail.

        This scans two levels deep:
        - Level 0: top-level dirs in *live_root* missing from *wt_root*
        - Level 1: subdirs within tracked parents (e.g.
          ``ground_truth_builder/checkpoints/`` where the parent is tracked
          but the subdir is not)

        Only directories are symlinked; regular files are ignored (they are
        either tracked or irrelevant).  Hidden directories (starting with
        ``'.'``) are skipped to avoid symlinking ``.git``, ``.ahvs``, etc.
        """
        if not live_root.is_dir() or not wt_root.is_dir():
            return
        for item in live_root.iterdir():
            if item.name.startswith(".") or not item.is_dir():
                continue
            wt_item = wt_root / item.name
            if not wt_item.exists() and not wt_item.is_symlink():
                # Entirely untracked directory — symlink it
                wt_item.symlink_to(item.resolve())
                logger.info(
                    "Symlinked untracked dir into worktree: %s → %s",
                    wt_item, item.resolve(),
                )
            elif wt_item.is_dir() and not wt_item.is_symlink():
                # Tracked parent — check one level deeper for untracked subdirs
                for sub in item.iterdir():
                    if sub.name.startswith(".") or not sub.is_dir():
                        continue
                    wt_sub = wt_item / sub.name
                    if not wt_sub.exists() and not wt_sub.is_symlink():
                        wt_sub.symlink_to(sub.resolve())
                        logger.info(
                            "Symlinked untracked subdir into worktree: %s → %s",
                            wt_sub, sub.resolve(),
                        )

    @staticmethod
    def _run_git(
        args: list[str], cwd: Path
    ) -> subprocess.CompletedProcess[str] | None:
        try:
            logger.debug("Running git command: git %s (cwd=%s)", " ".join(args), cwd)
            return subprocess.run(
                ["git", *args],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Git operation failed (%s): %s", " ".join(args), exc)
            return None
