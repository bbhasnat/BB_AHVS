#!/usr/bin/env bash
# AHVS Installer — one-command setup
# Usage:
#   curl -sSL <url>/install.sh | bash          # from remote
#   ./install.sh                                # from cloned repo
#   ./install.sh --dev                          # editable install (for contributors)

set -euo pipefail

REPO_URL="https://github.com/bbhasnat/BB_AHVS.git"
MIN_PYTHON="3.11"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[AHVS]${NC} $*"; }
warn()  { echo -e "${YELLOW}[AHVS]${NC} $*"; }
error() { echo -e "${RED}[AHVS]${NC} $*" >&2; }

# --- Check Python version ---
check_python() {
    local py=""
    for candidate in python3 python; do
        if command -v "$candidate" &>/dev/null; then
            local ver
            ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
            if "$candidate" -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
                py="$candidate"
                break
            fi
        fi
    done

    if [ -z "$py" ]; then
        error "Python >= $MIN_PYTHON is required but not found."
        error "Install it: https://www.python.org/downloads/"
        exit 1
    fi

    echo "$py"
}

# --- Check git ---
check_git() {
    if ! command -v git &>/dev/null; then
        error "git is required but not found."
        exit 1
    fi
}

# --- Main ---
main() {
    local dev_mode=false
    if [[ "${1:-}" == "--dev" || "${1:-}" == "-d" ]]; then
        dev_mode=true
    fi

    info "AHVS Installer"
    echo "========================================"

    # Prerequisites
    info "Checking prerequisites..."
    check_git
    PYTHON=$(check_python)
    info "Using: $PYTHON ($($PYTHON --version 2>&1))"

    # Determine install source
    if [ -f "pyproject.toml" ] && grep -q 'name = "ahvs"' pyproject.toml 2>/dev/null; then
        # Running from inside the repo
        INSTALL_DIR="$(pwd)"
        info "Installing from local repo: $INSTALL_DIR"
    else
        # Clone the repo to a temp dir
        INSTALL_DIR="$(mktemp -d)"
        info "Cloning AHVS to $INSTALL_DIR..."
        git clone --depth 1 "$REPO_URL" "$INSTALL_DIR"
    fi

    # Install the package
    if $dev_mode; then
        info "Installing in editable (dev) mode..."
        $PYTHON -m pip install -e "$INSTALL_DIR[dev]" --quiet
    else
        info "Installing AHVS package..."
        $PYTHON -m pip install "$INSTALL_DIR" --quiet
    fi

    # Copy skills bundled in the repo to the package data dir for non-dev installs
    # For editable installs, the installer reads directly from .claude/skills/
    if ! $dev_mode; then
        SKILLS_SRC="$INSTALL_DIR/.claude/skills"
        # Find the installed package location
        PKG_DIR=$($PYTHON -c "import ahvs; print(ahvs.__file__.rsplit('/', 1)[0])")
        SKILLS_DST="$PKG_DIR/_skills"
        if [ -d "$SKILLS_SRC" ]; then
            info "Bundling skills into package..."
            rm -rf "$SKILLS_DST"
            mkdir -p "$SKILLS_DST"
            for skill in ahvs_genesis ahvs_multiagent ahvs_onboarding; do
                if [ -d "$SKILLS_SRC/$skill" ]; then
                    cp -r "$SKILLS_SRC/$skill" "$SKILLS_DST/$skill"
                fi
            done
        fi
    fi

    # Run the AHVS installer (copies skills globally, inits ~/.ahvs/)
    info "Running ahvs install..."
    $PYTHON -m ahvs.cli install

    echo ""
    echo "========================================"
    info "Done! Try these commands:"
    echo "  ahvs --help              # CLI reference"
    echo "  ahvs --list-repos        # show registered repos"
    echo ""
    info "In Claude Code, use these skills:"
    echo "  /ahvs_onboarding         # onboard a repo"
    echo "  /ahvs_onboarding:gui     # onboard via browser form"
    echo "  /ahvs_genesis            # create project from data"
    echo "  /ahvs_genesis:gui        # genesis via browser form"
    echo "  /ahvs_multiagent         # run multi-agent cycle"
    echo "  /ahvs_multiagent:gui     # multi-agent via browser form"
}

main "$@"
