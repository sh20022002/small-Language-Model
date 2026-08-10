#!/usr/bin/env python3
"""
Check notebook paths for correctness.

Validates that paths referenced in notebooks point to existing files
or will exist in the Kaggle environment.
"""

import json
import re
import sys
from pathlib import Path


# Common Kaggle paths that will exist
KAGGLE_PATHS = {
    "/kaggle/input",
    "/kaggle/working",
    "/kaggle/models",
}

# Relative paths that should exist in repo
REPO_PATHS = {
    "./src",
    "./tests",
    "./notebooks",
    "./models",
    "./data",
}


def extract_paths_from_code(code: str) -> list[str]:
    """
    Extract file paths from Python code.

    Args:
        code: Python code to search

    Returns:
        List of file paths found
    """
    paths = []

    # Match string literals with path-like content
    patterns = [
        r"['\"]([^'\"]*\/[^'\"]*)['\"]",  # Paths with /
        r"['\"](\.\.[^'\"]*)['\"]",       # Relative paths starting with ..
        r"Path\(['\"]([^'\"]+)['\"]\)",   # Path() constructor
        r"open\(['\"]([^'\"]+)['\"]\)",   # open() calls
    ]

    for pattern in patterns:
        matches = re.findall(pattern, code)
        paths.extend(matches)

    # Filter for actual path-like strings
    valid_paths = []
    for path in paths:
        if "/" in path or "\\" in path:
            valid_paths.append(path)

    return valid_paths


def check_notebook_paths(notebook_path: Path, repo_root: Path = Path(".")) -> tuple[bool, list[str]]:
    """
    Check paths in a notebook.

    Args:
        notebook_path: Path to notebook
        repo_root: Root of repository for relative path checking

    Returns:
        Tuple of (is_valid, warning_messages)
    """
    warnings = []

    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    for cell_idx, cell in enumerate(notebook.get("cells", []), 1):
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])
        if isinstance(source, list):
            code = "".join(source)
        else:
            code = source

        paths = extract_paths_from_code(code)

        for path_str in paths:
            # Skip magic commands and shell commands
            if path_str.startswith("!") or path_str.startswith("%"):
                continue

            # Check Kaggle paths
            is_kaggle_path = any(path_str.startswith(kp) for kp in KAGGLE_PATHS)
            if is_kaggle_path:
                # These are expected in Kaggle notebooks
                continue

            # Check repo paths
            if path_str.startswith("./") or path_str.startswith("../"):
                abs_path = repo_root / path_str
                if not abs_path.exists():
                    warnings.append(
                        f"Cell {cell_idx}: Path may not exist: {path_str}"
                    )
            elif path_str.startswith("/"):
                # Absolute paths outside Kaggle are problematic
                warnings.append(
                    f"Cell {cell_idx}: Absolute path not in Kaggle location: {path_str}"
                )

    return len(warnings) == 0, warnings


def main():
    """Check paths in all notebooks."""
    root_dir = Path(".")
    notebooks = sorted(root_dir.glob("**/*.ipynb"))

    if not notebooks:
        print("No notebooks found.")
        return 0

    print(f"Checking paths in {len(notebooks)} notebook(s)...\n")

    issues = []
    for notebook_path in notebooks:
        is_valid, warnings = check_notebook_paths(notebook_path, root_dir)

        if is_valid:
            print(f"✓ {notebook_path}")
        else:
            print(f"⚠ {notebook_path}")
            for warning in warnings:
                print(f"  {warning}")
            issues.append(notebook_path)

    print(f"\n{'='*60}")
    print(f"Notebooks with path warnings: {len(issues)}/{len(notebooks)}")

    if issues:
        print(f"\nNotebooks with issues:")
        for nb in issues:
            print(f"  - {nb}")
        print(f"\nNote: Warnings are not fatal; verify Kaggle compatibility manually")
        return 0  # Don't fail, just warn

    print(f"✓ All paths validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
