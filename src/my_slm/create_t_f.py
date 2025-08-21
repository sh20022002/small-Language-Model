"""
wikidump2txt.py  —  Download + extract full Wikipedia text in any language.

• Finds the latest dump at https://dumps.wikimedia.org/{lang}wiki/latest/
• Downloads `{lang}wiki-latest-pages-articles.xml.bz2`
• Runs WikiExtractor to strip markup → plain text
• Writes a single UTF-8 text file: 1 article per line.

Dependencies: requests, tqdm, wikiextractor  (pip install wikiextractor tqdm)
"""

import os, sys, re, argparse, subprocess, shutil, tempfile, bz2, gzip, html
from pathlib import Path
from urllib.parse import urljoin

import requests
from tqdm import tqdm


DUMPS_BASE = "https://dumps.wikimedia.org/"

def latest_dump_url(lang: str) -> str:
    """Return direct URL to the latest {lang}wiki pages-articles dump (.bz2)."""
    index = requests.get(urljoin(DUMPS_BASE, f"{lang}wiki/latest/")).text
    m = re.search(rf'href="({lang}wiki-latest-pages-articles\.xml\.bz2)"', index)
    if not m:
        raise RuntimeError("Could not locate dump name on index page.")
    return urljoin(DUMPS_BASE, f"{lang}wiki/latest/{m.group(1)}")

def download(url: str, dest: Path):
    """Streaming download with progress bar."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        size = int(r.headers.get("content-length", 0))
        bar  = tqdm(total=size, unit="B", unit_scale=True, desc="Downloading")
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=8192):
                fh.write(chunk)
                bar.update(len(chunk))
        bar.close()

def ensure_wikiextractor() -> str:
    """Return the WikiExtractor CLI; install if missing."""
    exe = shutil.which("wikiextractor")
    if exe:
        return exe
    # fallback: try python -m wikiextractor.__main__
    try:
        import wikiextractor  # noqa: F401
        return f"{sys.executable} -m wikiextractor"
    except ImportError:
        sys.exit("WikiExtractor not installed.  Run:  pip install wikiextractor")

def run_wikiextractor(bz2_xml: Path, out_dir: Path, min_text_len=50):
    """Call WikiExtractor → output plain text files under out_dir."""
    cmd = [
        *ensure_wikiextractor().split(),
        "--json",           # structured JSON; easier to post-process
        "-o", str(out_dir),
        "--bytes", "200K",  # chunk size
        "--min_text_length", str(min_text_len),
        str(bz2_xml),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def collect_json_to_txt(json_root: Path, out_txt: Path):
    """Concatenate WikiExtractor JSON output into one UTF-8 text file."""
    with open(out_txt, "w", encoding="utf-8") as outfile:
        for json_file in sorted(json_root.rglob("*.json")):
            for line in json_file.open(encoding="utf-8"):
                try:
                    title, text = _json_to_title_text(line)
                    if text:
                        outfile.write(text.replace("\n", " ").strip() + "\n")
                except Exception:
                    continue  # skip malformed lines

def _json_to_title_text(json_line: str):
    """Parse tiny WikiExtractor JSON line → (title, text)."""
    import json as _json
    d = _json.loads(json_line)
    return d["title"], html.unescape(d["text"])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True, help="Wikipedia language code, e.g. en, he, de")
    p.add_argument("--out", required=True, help="output .txt path (one article per line)")
    p.add_argument("--tmp", default=None, help="optional temp dir")
    args = p.parse_args()

    tmpdir = Path(args.tmp or tempfile.mkdtemp(prefix="wikidump_"))
    tmpdir.mkdir(parents=True, exist_ok=True)

    bz2_path = tmpdir / f"{args.lang}wiki-latest-pages-articles.xml.bz2"
    if not bz2_path.exists():
        url = latest_dump_url(args.lang)
        print("Latest dump:", url)
        download(url, bz2_path)

    extract_dir = tmpdir / "extract"
    extract_dir.mkdir(exist_ok=True)

    run_wikiextractor(bz2_path, extract_dir)
    collect_json_to_txt(extract_dir, Path(args.out))

    print("✔ Done. Output saved to", args.out)
    # Uncomment next line to remove temp files automatically:
    # shutil.rmtree(tmpdir)

if __name__ == "__main__":
    main()