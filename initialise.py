#!/usr/bin/env python3
"""Project setup script that initialises a Python project."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def check_uv_installed() -> str:
    """Check if uv is installed and return its path."""
    uv_path = shutil.which("uv")
    if uv_path is None:
        print(
            "uv is not installed. Please, install it from: "
            "https://docs.astral.sh/uv/getting-started/installation/#installation-methods"
        )
        sys.exit(1)
    return uv_path


def initialise_precommit(uv_path: str) -> None:
    """Install pre-commit and initialise hooks using the provided uv executable path."""
    subprocess.run(
        [uv_path, "run", "pre-commit", "install"], check=True, capture_output=True
    )
    subprocess.run(
        [uv_path, "run", "pre-commit", "run", "--all-files"],
        check=True,
        capture_output=True,
    )


def get_user_input() -> tuple[str, str, str]:
    """Get project details from user."""
    project_name = input("Enter project name: ").strip()
    description = input("Enter description [empty]: ").strip() or ""

    default_python_version = "3.11"
    python_version = (
        input(f"Enter Python version (e.g., 3.12) [{default_python_version}]: ").strip()
        or default_python_version
    )
    return project_name, description, python_version


def replace_variables(project_name: str, description: str, python_version: str) -> None:
    """Replace variables placeholders in all files."""
    replacements = {
        "python-project-initialiser": project_name,
        "python-project-initialiser-description": description,
        "3.11": python_version,
        "py311": f"py{python_version.replace('.', '')}",
    }

    for root, dirs, files in os.walk("."):
        # Skip .venv and other directories if needed.
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

        for file in files:
            # Skip the script itself.
            if file == Path(__file__).name:
                continue

            path = Path(root) / file
            try:
                content = path.read_text(encoding="utf-8")
                new_content = content
                for var, val in replacements.items():
                    new_content = new_content.replace(var, val)
                if new_content != content:
                    path.write_text(new_content, encoding="utf-8")
                    print(f"Updated {path}")
            except (UnicodeDecodeError, OSError):
                # Skip binary files or files that cannot be read.
                pass


def initialise_github_repository(project_name: str) -> None:
    """Initialize a local Git repository and create a remote GitHub repository."""
    git_path = shutil.which("git")
    if git_path is None:
        print("Git is not installed. Please install Git to initialize the repository.")
        return

    gh_path = shutil.which("gh")
    if gh_path is None:
        print(
            "GitHub CLI (gh) is not installed. Please install it from: https://cli.github.com/"
        )
        return

    # Check if authenticated with GitHub.
    try:
        subprocess.run([gh_path, "auth", "status"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print(
            "You are not authenticated with GitHub. Please run 'gh auth login' first."
        )
        return

    # Check if already a git repository.
    if Path(".git").exists():
        print("Git repository already initialized.")
        # Check if there are commits.
        try:
            subprocess.run(
                [git_path, "log", "--oneline", "-1"], check=True, capture_output=True
            )
            has_commits = True
        except subprocess.CalledProcessError:
            has_commits = False
    else:
        # Initialize git repository.
        subprocess.run([git_path, "init"], check=True)
        print("Initialized empty Git repository.")
        has_commits = False

    if not has_commits:
        # Add all files.
        subprocess.run([git_path, "add", "."], check=True)
        print("Added files to staging area.")

        # Commit.
        try:
            subprocess.run(
                [git_path, "commit", "-m", "feat: Initial commit"], check=True
            )
            print("Created initial commit.")
        except subprocess.CalledProcessError:
            print("No changes to commit or commit failed.")
            return  # Don't create repo if no commit.

    # Ask for repository visibility
    visibility = input("Make the repository public? (y/n) [y]: ").strip().lower()
    visibility_flag = "--public" if visibility in ("", "y", "yes") else "--private"

    # Create a GitHub repository.
    try:
        subprocess.run(
            [
                gh_path,
                "repo",
                "create",
                project_name,
                visibility_flag,
                "--source=.",
                "--push",
            ],
            check=True,
        )
        print(f"Created and pushed to GitHub repository: {project_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to create GitHub repository: {e}")
        return


def main() -> None:
    """Run the main setup process."""
    uv_path = check_uv_installed()
    project_name, description, python_version = get_user_input()
    replace_variables(project_name, description, python_version)
    initialise_precommit(uv_path)

    # Remove the script itself.
    Path(__file__).unlink()

    initialise_github_repository(project_name)

    print("The setup script has been removed.")
    print("Project initialized successfully.")


if __name__ == "__main__":
    main()
