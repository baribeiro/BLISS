"""Build the anonymized copy of the project page in ``review/``.

The copy is served through anonymous.4open.science, which serves the page but
not the ``fetch()`` of its JSON files. Every file of ``static/data/`` is
therefore written as a small script that stores its content in
``window.BLISS_DATA``, and ``app.js`` loads those scripts when
``window.BLISS_EMBED`` is set.

Run from the root of the ``gh-pages`` branch::

    python scripts/build_review.py

The script fails if an identifying term is left in the copy.
"""

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "review"

# Data files the page does not read; left out of the copy.
SKIP_DATA = {"explorer.json"}

# Terms that must not appear anywhere in the anonymized copy.
IDENTIFYING = [
    "bribeiro", "baribeiro", "bruno", "ribeiro", "bessa", "brown.edu",
    "brown university", "reis", "joao97", "oscar", "mbessa", "github.com/",
    "huggingface.co/",
]
TEXT_SUFFIXES = {".html", ".css", ".js", ".json", ".md", ".txt"}


def write_data(src, dst):
    """Write every JSON file of ``src`` as an embedding script under ``dst``.

    Parameters
    ----------
    src : pathlib.Path
        The ``static/data`` folder of the site.
    dst : pathlib.Path
        The ``static/data`` folder of the anonymized copy.

    Returns
    -------
    int
        Number of files written.
    """
    n = 0
    for f in sorted(src.rglob("*.json")):
        key = f.relative_to(src).as_posix()
        if key in SKIP_DATA:
            continue
        data = json.loads(f.read_text())
        out = dst / (key[:-len(".json")] + ".js")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "(window.BLISS_DATA = window.BLISS_DATA || {})[%s] = %s;\n"
            % (json.dumps(key), json.dumps(data, separators=(",", ":")))
        )
        n += 1
    return n


def check_anonymous(folder):
    """Fail if an identifying term is found in a text file of ``folder``.

    Parameters
    ----------
    folder : pathlib.Path
        The anonymized copy.
    """
    pattern = re.compile("|".join(re.escape(t) for t in IDENTIFYING), re.I)
    hits = []
    for f in folder.rglob("*"):
        if f.is_file() and f.suffix in TEXT_SUFFIXES:
            for m in pattern.finditer(f.read_text(errors="ignore")):
                hits.append("%s: %r" % (f.relative_to(ROOT), m.group(0)))
    if hits:
        print("identifying terms left in review/:", *hits[:50], sep="\n  ")
        sys.exit(1)


def main():
    """Rebuild ``review/`` from ``index.html`` and ``static/``."""
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "static").mkdir(parents=True)
    for sub in ("css", "js", "images"):
        shutil.copytree(ROOT / "static" / sub, OUT / "static" / sub)
    n = write_data(ROOT / "static" / "data", OUT / "static" / "data")

    html = (ROOT / "index.html").read_text()
    if "<head>" not in html or 'src="static/js/app.js' not in html:
        sys.exit("index.html has no <head> or does not load static/js/app.js")
    html = html.replace("<head>", "<head>\n  <script>window.BLISS_EMBED = true;</script>", 1)
    (OUT / "index.html").write_text(html)

    check_anonymous(OUT)
    print("review/ written: %d data scripts, no identifying terms" % n)


if __name__ == "__main__":
    main()
