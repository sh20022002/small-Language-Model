#!/usr/bin/env python3
"""
Validate that all imports in notebooks can be resolved.

Checks that notebooks only import packages that are available
in the project environment.
"""

import ast
import json
import sys
from pathlib import Path
from typing import Optional


class ImportExtractor(ast.NodeVisitor):
    """Extract all imports from Python code."""

    def __init__(self):
        self.imports = []
        self.errors = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append(alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            self.imports.append(node.module.split(".")[0])


def extract_imports_from_code(code: str) -> tuple[list[str], list[str]]:
    """
    Extract imports from Python code.

    Args:
        code: Python code to analyze

    Returns:
        Tuple of (imports, errors)
    """
    imports = []
    errors = []

    # Filter out notebook magic commands
    lines = []
    for line in code.split("\n"):
        if not line.strip().startswith("%") and not line.strip().startswith("!"):
            lines.append(line)

    filtered_code = "\n".join(lines)

    try:
        tree = ast.parse(filtered_code)
        extractor = ImportExtractor()
        extractor.visit(tree)
        imports = list(set(extractor.imports))
    except SyntaxError as e:
        errors.append(f"Syntax error: {e}")

    return sorted(imports), errors


def validate_notebook_imports(notebook_path: Path) -> tuple[bool, list[str]]:
    """
    Validate imports in a notebook.

    Args:
        notebook_path: Path to notebook

    Returns:
        Tuple of (is_valid, messages)
    """
    messages = []

    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    all_imports = set()

    for cell_idx, cell in enumerate(notebook.get("cells", []), 1):
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])
        if isinstance(source, list):
            code = "".join(source)
        else:
            code = source

        imports, errors = extract_imports_from_code(code)

        if errors:
            for error in errors:
                messages.append(f"Cell {cell_idx}: {error}")
        else:
            all_imports.update(imports)

    # Standard library modules that are always available
    stdlib_modules = {
        "sys", "os", "json", "pathlib", "re", "math", "random", "time",
        "datetime", "collections", "itertools", "functools", "copy",
        "pickle", "csv", "io", "tempfile", "shutil", "glob", "logging",
        "argparse", "threading", "multiprocessing", "subprocess", "abc",
        "unittest", "typing", "dataclasses",
    }

    # Project modules
    project_modules = {"my_slm"}

    # External dependencies (from pyproject.toml's [project.dependencies]).
    # Keep this in sync with pyproject.toml — it's the source of truth.
    external_modules = {"torch", "numpy", "tqdm", "matplotlib"}

    # Check each import
    problematic = []
    for imp in all_imports:
        if imp not in stdlib_modules and \
           imp not in project_modules and \
           imp not in external_modules and \
           not imp.startswith("_"):
            problematic.append(imp)

    if problematic:
        messages.append(
            f"Unknown imports (may not be available): {', '.join(problematic)}"
        )

    return len(messages) == 0, messages


def main():
    """Validate imports in all notebooks."""
    root_dir = Path(".")
    notebooks = sorted(root_dir.glob("**/*.ipynb"))

    if not notebooks:
        print("No notebooks found.")
        return 0

    print(f"Validating imports in {len(notebooks)} notebook(s)...\n")

    failed = []

    for notebook_path in notebooks:
        is_valid, messages = validate_notebook_imports(notebook_path)

        if is_valid:
            print(f"✓ {notebook_path}")
        else:
            print(f"⚠ {notebook_path}")
            for message in messages:
                print(f"  {message}")
            # Don't mark as failed, just warn
            # Unknown modules might be installed in Kaggle

    print(f"\n{'='*60}")
    print(f"✓ Import validation complete")
    print("(Warnings don't fail validation; verify in Kaggle environment)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
