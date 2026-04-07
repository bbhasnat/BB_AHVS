"""AHVS installer — copies skills globally and initializes ~/.ahvs/."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Skills bundled with the package (relative to this file's parent)
_PACKAGE_ROOT = Path(__file__).resolve().parent
_BUNDLED_SKILLS_DIR = _PACKAGE_ROOT.parent / ".claude" / "skills"
_BUNDLED_COMMANDS_DIR = _PACKAGE_ROOT.parent / ".claude" / "commands"

# Global destinations
_CLAUDE_SKILLS_DIR = Path.home() / ".claude" / "skills"
_CLAUDE_COMMANDS_DIR = Path.home() / ".claude" / "commands"
_AHVS_HOME = Path.home() / ".ahvs"

# Skill directories to install
SKILL_NAMES = ["ahvs_genesis", "ahvs_multiagent", "ahvs_onboarding"]

# Command directories to install (subcommand registration)
# Each maps to ~/.claude/commands/<dir_name>/gui.md
COMMAND_DIRS = ["ahvs_genesis", "ahvs_multiagent", "ahvs_onboarding"]


def _check_python_version() -> list[str]:
    """Check Python >= 3.11."""
    if sys.version_info < (3, 11):
        return [f"Python >= 3.11 required (found {sys.version})"]
    return []


def _check_git() -> list[str]:
    """Check git is available."""
    import shutil
    import subprocess

    git_bin = shutil.which("git") or "/usr/bin/git"
    try:
        subprocess.run(
            [git_bin, "--version"],
            capture_output=True, check=True, text=True,
        )
        return []
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ["git is not installed or not in PATH"]


def preflight() -> list[str]:
    """Run prerequisite checks. Returns list of error strings (empty = OK)."""
    errors: list[str] = []
    errors.extend(_check_python_version())
    errors.extend(_check_git())
    return errors


def install_skills(force: bool = False, quiet: bool = False) -> list[str]:
    """Copy bundled skills to ~/.claude/skills/.

    Returns list of installed skill names.
    """
    installed: list[str] = []

    # Find skills — try bundled location first, then package data
    skills_src = _BUNDLED_SKILLS_DIR
    if not skills_src.is_dir():
        # Installed via pip: skills are in ahvs/_skills/ (package data)
        skills_src = _PACKAGE_ROOT / "_skills"
    if not skills_src.is_dir():
        print(
            f"Warning: skill source directory not found at {_BUNDLED_SKILLS_DIR} "
            f"or {_PACKAGE_ROOT / '_skills'}",
            file=sys.stderr,
        )
        return installed

    _CLAUDE_SKILLS_DIR.mkdir(parents=True, exist_ok=True)

    for name in SKILL_NAMES:
        src = skills_src / name
        if not src.is_dir():
            if not quiet:
                print(f"  skip {name} (not found in source)")
            continue

        dst = _CLAUDE_SKILLS_DIR / name

        if dst.exists() and not force:
            # Check if update is needed by comparing SKILL.md content
            src_skill = src / "SKILL.md"
            dst_skill = dst / "SKILL.md"
            if (
                src_skill.exists()
                and dst_skill.exists()
                and src_skill.read_text() == dst_skill.read_text()
            ):
                if not quiet:
                    print(f"  {name}: up to date")
                installed.append(name)
                continue

        # Remove old version and copy fresh
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        installed.append(name)
        if not quiet:
            print(f"  {name}: {'updated' if dst.exists() else 'installed'}")

    return installed


def install_commands(force: bool = False, quiet: bool = False) -> list[str]:
    """Copy bundled command directories to ~/.claude/commands/.

    Command directories register skill subcommands (e.g., ahvs_genesis:gui).
    Uses the directory structure: commands/<skill_name>/gui.md
    Returns list of installed command directory names.
    """
    installed: list[str] = []

    # Find commands — try bundled location first, then package data
    cmds_src = _BUNDLED_COMMANDS_DIR
    if not cmds_src.is_dir():
        cmds_src = _PACKAGE_ROOT / "_commands"
    if not cmds_src.is_dir():
        if not quiet:
            print("  no command files found in source (subcommands skipped)")
        return installed

    _CLAUDE_COMMANDS_DIR.mkdir(parents=True, exist_ok=True)

    for dirname in COMMAND_DIRS:
        src = cmds_src / dirname
        if not src.is_dir():
            if not quiet:
                print(f"  skip {dirname} (not found)")
            continue

        dst = _CLAUDE_COMMANDS_DIR / dirname

        if dst.exists() and not force:
            # Check if update needed by comparing gui.md
            src_cmd = src / "gui.md"
            dst_cmd = dst / "gui.md"
            if (
                src_cmd.exists()
                and dst_cmd.exists()
                and src_cmd.read_text() == dst_cmd.read_text()
            ):
                if not quiet:
                    print(f"  {dirname}: up to date")
                installed.append(dirname)
                continue

        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        installed.append(dirname)
        if not quiet:
            print(f"  {dirname}: installed")

    return installed


def init_ahvs_home(quiet: bool = False) -> None:
    """Initialize ~/.ahvs/ directory structure."""
    _AHVS_HOME.mkdir(parents=True, exist_ok=True)

    registry_path = _AHVS_HOME / "registry.json"
    if not registry_path.exists():
        registry_path.write_text(
            json.dumps({"repos": {}}, indent=2), encoding="utf-8"
        )
        if not quiet:
            print(f"  created {registry_path}")
    else:
        if not quiet:
            print(f"  {registry_path}: exists")

    # Global evolution store
    evo_dir = _AHVS_HOME / "global" / "evolution"
    evo_dir.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"  created {evo_dir}")


def write_install_receipt() -> None:
    """Write a receipt file for tracking install state."""
    receipt_path = _AHVS_HOME / "install_receipt.json"
    try:
        from ahvs import __version__
    except (ImportError, AttributeError):
        __version__ = "0.1.0"

    receipt = {
        "installed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": __version__,
        "python": sys.version,
        "skills": SKILL_NAMES,
        "skills_dir": str(_CLAUDE_SKILLS_DIR),
        "ahvs_home": str(_AHVS_HOME),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")


def run_install(force: bool = False, quiet: bool = False) -> int:
    """Full install: preflight, skills, commands, init, receipt. Returns exit code."""
    if not quiet:
        print("AHVS Installer")
        print("=" * 40)

    # Preflight
    if not quiet:
        print("\n[1/5] Preflight checks...")
    errors = preflight()
    if errors:
        for e in errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        return 1
    if not quiet:
        print("  all checks passed")

    # Install skills
    if not quiet:
        print("\n[2/5] Installing skills to ~/.claude/skills/...")
    installed = install_skills(force=force, quiet=quiet)
    if not installed:
        print("  WARNING: no skills were installed", file=sys.stderr)

    # Install commands (subcommand registration)
    if not quiet:
        print("\n[3/5] Installing commands to ~/.claude/commands/...")
    install_commands(force=force, quiet=quiet)

    # Init ~/.ahvs/
    if not quiet:
        print("\n[4/5] Initializing ~/.ahvs/...")
    init_ahvs_home(quiet=quiet)

    # Receipt
    if not quiet:
        print("\n[5/5] Writing install receipt...")
    write_install_receipt()

    if not quiet:
        print("\n" + "=" * 40)
        print("AHVS installed successfully!")
        print()
        print("Skills installed:")
        for name in installed:
            print(f"  /ahvs_{name.removeprefix('ahvs_')}")
            if name == "ahvs_genesis":
                print(f"  /ahvs_genesis:gui")
            elif name == "ahvs_multiagent":
                print(f"  /ahvs_multiagent:gui")
            elif name == "ahvs_onboarding":
                print(f"  /ahvs_onboarding:gui")
        print()
        print("Quick start:")
        print("  ahvs --help              # CLI reference")
        print("  ahvs --list-repos        # show registered repos")
        print("  /ahvs_onboarding         # onboard a repo (in Claude Code)")
        print("  /ahvs_genesis:gui        # create project from data (browser form)")

    return 0


def run_update(quiet: bool = False) -> int:
    """Update: reinstall skills + commands (force) + refresh receipt. Returns exit code."""
    if not quiet:
        print("AHVS Update")
        print("=" * 40)
        print("\n[1/3] Updating skills...")

    installed = install_skills(force=True, quiet=quiet)

    if not quiet:
        print("\n[2/3] Updating commands...")
    cmds = install_commands(force=True, quiet=quiet)

    if not quiet:
        print(f"\n[3/3] Writing install receipt...")
    write_install_receipt()

    if not quiet:
        print(f"\n{'=' * 40}")
        print(f"Updated {len(installed)} skill(s), {len(cmds)} command(s).")

    return 0


def run_uninstall(quiet: bool = False) -> int:
    """Remove installed skills and commands. Returns exit code."""
    if not quiet:
        print("AHVS Uninstall")
        print("=" * 40)

    removed = 0
    for name in SKILL_NAMES:
        dst = _CLAUDE_SKILLS_DIR / name
        if dst.exists():
            shutil.rmtree(dst)
            removed += 1
            if not quiet:
                print(f"  removed skill: {name}")

    cmds_removed = 0
    for dirname in COMMAND_DIRS:
        dst = _CLAUDE_COMMANDS_DIR / dirname
        if dst.exists():
            shutil.rmtree(dst)
            cmds_removed += 1
            if not quiet:
                print(f"  removed command: {dirname}")

    receipt = _AHVS_HOME / "install_receipt.json"
    if receipt.exists():
        receipt.unlink()

    if not quiet:
        print(f"\nRemoved {removed} skill(s), {cmds_removed} command(s).")
        print("Note: ~/.ahvs/ (registry, evolution data) was preserved.")
        print("      To remove it: rm -rf ~/.ahvs/")

    return 0
