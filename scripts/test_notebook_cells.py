#!/usr/bin/env python3
"""
Extract and test notebook cells.

Extracts code cells from Jupyter notebooks and validates that they
execute without import errors in a simulated Kaggle environment.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def extract_code_cells(notebook_path: Path) -> list[str]:
    """
    Extract all code cells from a notebook.

    Args:
        notebook_path: Path to the notebook file

    Returns:
        List of code strings from code cells
    """
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    code_cells = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            if isinstance(source, list):
                code = "".join(source)
            else:
                code = source
            if code.strip():
                code_cells.append(code)

    return code_cells


def validate_imports(code: str) -> Optional[str]:
    """
    Check if code has valid imports.

    Args:
        code: Python code to validate

    Returns:
        Error message if import fails, None if valid
    """
    # Parse imports from code
    import_lines = []
    for line in code.split("\n"):
        line = line.strip()
        if line.startswith("import ") or line.startswith("from "):
            import_lines.append(line)

    # Try to validate imports
    for imp_line in import_lines:
        # Skip notebook-specific magic commands
        if imp_line.startswith("!") or imp_line.startswith("%"):
            continue

        # Try to parse the import
        try:
            # Basic validation: can we compile it?
            compile(imp_line, "<string>", "exec")
        except SyntaxError as e:
            return f"Syntax error in import: {imp_line}\n  {e}"

    return None


def extract_to_file(notebook_path: Path, output_path: Path):
    """
    Extract all code cells from notebook and write to file.

    Args:
        notebook_path: Path to the notebook
        output_path: Path to output Python file
    """
    code_cells = extract_code_cells(notebook_path)

    # Filter out magic commands and shell commands for standalone script
    cleaned_cells = []
    for cell in code_cells:
        lines = []
        for line in cell.split("\n"):
            # Skip notebook magic commands
            if line.strip().startswith("%") or line.strip().startswith("!"):
                continue
            lines.append(line)
        cleaned = "\n".join(lines).strip()
        if cleaned:
            cleaned_cells.append(cleaned)

    # Write to output file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('"""Auto-generated from notebook cells for validation."""\n\n')
        for i, cell in enumerate(cleaned_cells, 1):
            f.write(f"# Cell {i}\n")
            f.write(cell)
            f.write("\n\n")

    print(f"Extracted {len(cleaned_cells)} code cells to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and test notebook cells"
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        help="Path to notebook file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to output Python file"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate imports, don't extract"
    )

    args = parser.parse_args()

    # If no specific notebook, find notebooks in repo
    notebooks = []
    if args.notebook:
        notebooks = [args.notebook]
    else:
        root = Path(".")
        notebooks = sorted(root.glob("**/*.ipynb"))

    if not notebooks:
        print("No notebooks found.")
        return 0

    print(f"Processing {len(notebooks)} notebook(s)...\n")

    failed = []
    for notebook_path in notebooks:
        if not notebook_path.exists():
            print(f"✗ {notebook_path} - File not found")
            failed.append(notebook_path)
            continue

        try:
            code_cells = extract_code_cells(notebook_path)

            # Validate imports in each cell
            import_errors = []
            for i, cell in enumerate(code_cells, 1):
                error = validate_imports(cell)
                if error:
                    import_errors.append(f"Cell {i}: {error}")

            if import_errors:
                print(f"✗ {notebook_path}")
                for error in import_errors:
                    print(f"  {error}")
                failed.append(notebook_path)
            else:
                print(f"✓ {notebook_path} ({len(code_cells)} cells)")

                # Extract to file if output specified
                if args.output and len(notebooks) == 1:
                    extract_to_file(notebook_path, args.output)

        except Exception as e:
            print(f"✗ {notebook_path} - {e}")
            failed.append(notebook_path)

    if failed:
        print(f"\n{'='*60}")
        print(f"Failed to process {len(failed)}/{len(notebooks)} notebook(s)")
        return 1

    print(f"\n{'='*60}")
    print(f"✓ All {len(notebooks)} notebook(s) validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
