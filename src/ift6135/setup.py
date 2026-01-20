"""Environment setup utilities for IFT 6135 notebooks.

Handles both local development and Google Colab environments seamlessly.
"""

import sys
from pathlib import Path

REPO_URL = "https://github.com/trvslhlt/ift_6135_representation_learning.git"
COLAB_REPO_PATH = Path("/content/ift_6135_representation_learning")


def is_colab() -> bool:
    """Check if running in Google Colab."""
    return "google.colab" in sys.modules


def get_repo_root() -> Path:
    """Find the repository root directory.

    Returns:
        Path to the repository root (contains pyproject.toml).

    Raises:
        FileNotFoundError: If repository root cannot be found.
    """
    if is_colab():
        return COLAB_REPO_PATH

    # Walk up from current directory to find pyproject.toml
    current = Path.cwd()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent

    raise FileNotFoundError(
        "Could not find repository root. "
        "Make sure you're running from within the repository."
    )


def setup_environment() -> Path:
    """Set up the environment for notebook execution.

    In Google Colab:
        - Clones or updates the repository
        - Installs dependencies from pyproject.toml
        - Adds src to Python path

    Locally:
        - Verifies repository structure
        - Adds src to Python path (dependencies managed by uv)

    Returns:
        Path to the repository root.
    """
    import subprocess

    if is_colab():
        if COLAB_REPO_PATH.exists():
            print("Updating repository...")
            subprocess.run(
                ["git", "-C", str(COLAB_REPO_PATH), "pull"],
                check=True,
                capture_output=True,
            )
        else:
            print("Cloning repository...")
            subprocess.run(
                ["git", "clone", REPO_URL, str(COLAB_REPO_PATH)],
                check=True,
                capture_output=True,
            )

        print("Installing dependencies...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-e", str(COLAB_REPO_PATH)],
            check=True,
            capture_output=True,
        )

    repo_root = get_repo_root()

    # Add src to path for imports
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    env_name = "Google Colab" if is_colab() else "Local"
    print(f"Environment: {env_name}")
    print(f"Repository: {repo_root}")

    return repo_root


def get_data_path(filename: str) -> Path:
    """Get the path to a data file.

    Args:
        filename: Name of the file in the data/ directory.

    Returns:
        Full path to the data file.
    """
    return get_repo_root() / "data" / filename
