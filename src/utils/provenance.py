"""Where a model came from: a UTC timestamp and the git commit of the code that produced it."""

import subprocess
from datetime import UTC, datetime

from src.utils.config import PROJECT_ROOT


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def git_short_sha() -> str:
    """Short commit hash, with '-dirty' when the working tree has uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "nogit"  # not a git checkout, or git is not installed
    return f"{sha}-dirty" if status else sha
