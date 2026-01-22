"""Version information for MeshView."""

import subprocess
from pathlib import Path

__version__ = "3.0.3"
__release_date__ = "2026-1-15"


def get_git_revision():
    """Get the current git revision hash."""
    try:
        repo_dir = Path(__file__).parent.parent
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_dir,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_git_revision_short():
    """Get the short git revision hash."""
    try:
        repo_dir = Path(__file__).parent.parent
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_dir,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_git_branch():
    """Get the current git branch name."""
    try:
        repo_dir = Path(__file__).parent.parent
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_dir,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_display_version():
    """Get version string with branch suffix for non-main branches."""
    branch = get_git_branch()
    base_version = __version__

    # If on a claude/ branch, append branch suffix
    if branch.startswith("claude/"):
        # Extract meaningful part of branch name
        branch_suffix = branch.replace("claude/", "").replace("-", "")
        return f"{base_version}-azmsh-claude-{branch_suffix}"

    # If on main/master or unknown, return base version
    return base_version


def get_version_info():
    """Get complete version information."""
    return {
        "version": get_display_version(),
        "release_date": __release_date__,
        "git_revision": get_git_revision(),
        "git_revision_short": get_git_revision_short(),
        "git_branch": get_git_branch(),
    }


# Cache git info at import time for performance
_git_revision = get_git_revision()
_git_revision_short = get_git_revision_short()

# Full version string for display - use the branch-aware version
__version_string__ = f"{get_display_version()} ~ {__release_date__}"
