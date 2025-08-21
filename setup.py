from pathlib import Path
from setuptools import setup, find_packages

# ----- Package metadata -----
PACKAGE_NAME = "hybrid-tokenizer"              # Distribution name on PyPI (can differ from import name)
VERSION = "0.1.0"
DESCRIPTION = "Hybrid tokenizer + tiny Transformer training utilities."
URL = "https://github.com/sh20022002/small-Language-Model"          # Replace with your repo URL
AUTHOR = "shmuel toren"
AUTHOR_EMAIL = "shmuel.tor@gmail.com"
LICENSE = "MIT"

# ----- Long description from README if available -----
readme = Path("README.md")
if readme.exists():
    long_description = readme.read_text(encoding="utf-8")
    long_description_content_type = "text/markdown"
else:
    long_description = DESCRIPTION
    long_description_content_type = "text/plain"

# ----- Runtime dependencies -----
install_requires = [
    "torch>=2.0",
    "numpy>=1.21",
    "tqdm>=4.60",
]

# ----- Optional extras (pip install .[dev]) -----
extras_require = {
    "dev": ["pytest", "black", "isort", "ruff"],
}

# Layout:
# your-repo/
# ├─ setup.py     ← this file
# ├─ README.md
# ├─ notebooks/   ← NOT packaged
# ├─ tests/       ← NOT packaged
# └─ src/
#    └─ my_slm/   ← your import package (must contain __init__.py)

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,

      package_dir={"": "src"},
    packages=find_packages(where="src", include=["my_slm", "my_slm.*"]),
    include_package_data=True,
    package_data={
        "my_slm": ["data/*.pkl.gz", "data/*.json", "data/*.txt", "data/*.yaml", "data/*.yml"],
    },

    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=["tokenizer", "nlp", "transformer", "pytorch"],
    
    # Example CLI (uncomment if you add a main() at my_slm/cli.py):
    # entry_points={
    #     "console_scripts": [
    #         "my-slm-train = my_slm.cli:main",
    #     ]
    # },
)
