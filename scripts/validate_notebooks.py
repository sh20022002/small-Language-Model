#!/usr/bin/env python3
"""
Validate Jupyter notebook JSON structure and format.

Ensures all notebooks in the repository are valid Jupyter notebook files
with correct JSON structure, metadata, and kernel specifications.
"""

import json
import sys
from pathlib import Path

import nbformat


def validate_notebook(notebook_path: Path) -> tuple[bool, list[str]]:
    """
    Validate a single notebook file.

    Args:
        notebook_path: Path to the notebook file

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check file exists
    if not notebook_path.exists():
        return False, [f"File not found: {notebook_path}"]

    # Check file is readable JSON
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON in {notebook_path}: {e}"]
    except Exception as e:
        return False, [f"Error reading {notebook_path}: {e}"]

    # Validate with nbformat
    try:
        nb = nbformat.reads(json.dumps(data), as_version=4)
        nbformat.validate(nb)
    except nbformat.ValidationError as e:
        errors.append(f"Notebook validation failed: {e}")
        return False, errors
    except Exception as e:
        errors.append(f"Unexpected error validating notebook: {e}")
        return False, errors

    # Check structure
    required_keys = {"cells", "metadata", "nbformat", "nbformat_minor"}
    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        errors.append(f"Missing required keys: {missing_keys}")

    # Check cells is a list
    if not isinstance(data.get("cells"), list):
        errors.append("'cells' must be a list")

    # Check metadata is dict
    if not isinstance(data.get("metadata"), dict):
        errors.append("'metadata' must be a dict")

    # Check kernel specification exists
    if "kernelspec" not in data.get("metadata", {}):
        errors.append("Missing 'kernelspec' in metadata")

    return len(errors) == 0, errors


def find_notebooks(root_dir: Path = Path(".")) -> list[Path]:
    """Find all notebook files in the repository."""
    notebooks = []
    patterns = ["**/*.ipynb"]

    for pattern in patterns:
        notebooks.extend(root_dir.glob(pattern))

    return sorted(set(notebooks))


def main():
    """Validate all notebooks in the repository."""
    root_dir = Path(".")
    notebooks = find_notebooks(root_dir)

    if not notebooks:
        print("No notebooks found.")
        return 0

    print(f"Validating {len(notebooks)} notebook(s)...\n")

    failed = []
    passed = []

    for notebook_path in notebooks:
        is_valid, errors = validate_notebook(notebook_path)

        if is_valid:
            print(f"✓ {notebook_path}")
            passed.append(notebook_path)
        else:
            print(f"✗ {notebook_path}")
            for error in errors:
                print(f"  - {error}")
            failed.append(notebook_path)

    print(f"\n{'='*60}")
    print(f"Passed: {len(passed)}/{len(notebooks)}")
    print(f"Failed: {len(failed)}/{len(notebooks)}")

    if failed:
        print(f"\nFailed notebooks:")
        for nb in failed:
            print(f"  - {nb}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
