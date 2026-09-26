"""Download the BLISS review sample into ``BLISS-1.0/`` and link it for the scripts.

Fetches the sample archive from its anonymized link, checks its SHA-256 and unpacks it to ``BLISS-1.0/`` at the
root of the repository, in the layout of the full release: ``shared.npz``, ``canonical/``, ``sweeps/``,
``metadata/``, ``splits/``. Then creates, once, the links the benchmark scripts read: ``splits/v2/`` (B1, B2 and B3
under their script names), ``metadata/``, ``release`` and ``BLISS-1.0/core``, and copies the frozen configurations
of ``configs/`` to ``outputs/``. For the review sample it also writes to ``metadata/`` one table of the defects of every shell
and the id lists of its four families. Standard library only.

Run from the root of the repository::

    python benchmark/data/download.py
    python benchmark/data/download.py --links-only      # full release already in BLISS-1.0/
"""

import argparse
import csv
import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile

B = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Anonymized download link of the review sample archive (290 MB) and its SHA-256
URL = "https://osf.io/download/6ab75f9e046567248c3c3a54/?view_only=4c5fd2f7aae1440e8de3e0bebece4648"
SHA256 = "2db67ad51c6235b37c96e4d3e354d002e00fed15c5971c1b94bfd90644aac4c2"

# Script names of the three protocols
SPLITS = {"B1": "random", "B2": "delta", "B3": "regime"}
# Parametric tables of the review sample, under the names the scripts read
SAMPLE_TABLES = {"shells.csv": "shells.csv", "single_defect_kd.csv": "single_defect_kd.csv",
                 "truth_dominance.csv": "truth_dominance.csv"}
# The sample lists the defects of its core shells and of its single- and two-defect shells in two files;
# the scripts read them as one table, defects.csv
SAMPLE_DEFECTS = ("defect_parameters.csv", "reference_defect_parameters.csv")


def download(url, path):
    """Stream ``url`` to ``path``.

    Parameters
    ----------
    url : str
        Source URL.
    path : str
        Destination file.

    Returns
    -------
    str
        SHA-256 hex digest of the downloaded bytes.
    """
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as resp, open(path, "wb") as out:
        while chunk := resp.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    return digest.hexdigest()


def fetch(url, sha256, data_dir):
    """Download, verify and unpack the review sample into ``data_dir``.

    Parameters
    ----------
    url : str
        Download link of the archive.
    sha256 : str
        Expected SHA-256 of the archive.
    data_dir : str
        Destination folder, ``BLISS-1.0/`` by default.
    """
    with tempfile.TemporaryDirectory(dir=B) as tmp:
        archive = os.path.join(tmp, "BLISS.zip")
        print("downloading %s" % url.split("?")[0])
        got = download(url, archive)
        if got != sha256:
            raise SystemExit("SHA-256 mismatch: expected %s, got %s" % (sha256, got))
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp)
        shutil.move(os.path.join(tmp, "BLISS"), data_dir)
    print("unpacked into %s" % data_dir)


def link(src, dst):
    """Create the symbolic link ``dst`` -> ``src`` unless ``dst`` already exists.

    Parameters
    ----------
    src : str
        Target of the link, relative to the folder of ``dst``.
    dst : str
        Path of the link.
    """
    if not os.path.lexists(dst):
        os.symlink(src, dst)


def write_defects(data_dir):
    """Write ``metadata/defects.csv``, the defects of every shell of the review sample in one table.

    Parameters
    ----------
    data_dir : str
        The review sample.
    """
    path = os.path.join(B, "metadata", "defects.csv")
    if os.path.lexists(path):
        return
    with open(path, "w") as out:
        for i, name in enumerate(SAMPLE_DEFECTS):
            with open(os.path.join(data_dir, "metadata", name)) as f:
                lines = f.readlines()
            out.writelines(lines if i == 0 else lines[1:])


def write_family_lists(data_dir):
    """Write the id lists of the four families of the review sample, as the full release holds them.

    The lists (one id per line, no header) are read from ``sample_ids.csv`` of the sample, whose ``block`` column
    names the family and whose ``reason`` starts with "dense" for a dense core shell (defects at least 10 deg apart);
    the other core shells, sparse ones and the sources of P3, have their defects at least 25 deg apart.

    Parameters
    ----------
    data_dir : str
        The review sample.
    """
    fam = {"multi_defects_phi_min_25deg": [], "multi_defects_phi_min_10deg": [], "single_defect": [], "two_defects": []}
    with open(os.path.join(data_dir, "sample_ids.csv")) as f:
        for r in csv.DictReader(f):
            if r["block"] == "core":
                fam["multi_defects_phi_min_10deg" if r["reason"].startswith("dense") else "multi_defects_phi_min_25deg"].append(r["id"])
            elif r["block"] == "single":
                fam["single_defect"].append(r["id"])
            elif r["block"] == "double":
                fam["two_defects"].append(r["id"])
    for k, ids in fam.items():
        path = os.path.join(B, "metadata", k + ".csv")
        if not os.path.exists(path):
            with open(path, "w") as out:
                out.write("".join(i + "\n" for i in ids))


def make_links(data_dir):
    """Create the links and folders the benchmark scripts read.

    Parameters
    ----------
    data_dir : str
        The release or the review sample.
    """
    name = os.path.basename(os.path.normpath(data_dir))
    link(name, os.path.join(B, "release"))
    link("canonical", os.path.join(data_dir, "core"))
    os.makedirs(os.path.join(B, "splits", "v2"), exist_ok=True)
    for prot, script_name in SPLITS.items():
        link(os.path.join("..", "..", name, "splits", prot + ".json"), os.path.join(B, "splits", "v2", script_name + ".json"))
    if os.path.exists(os.path.join(data_dir, "metadata", "defects.csv")):      # full release
        link(os.path.join(name, "metadata"), os.path.join(B, "metadata"))
    else:                                                                      # review sample
        os.makedirs(os.path.join(B, "metadata"), exist_ok=True)
        for dst, src in SAMPLE_TABLES.items():
            link(os.path.join("..", name, "metadata", src), os.path.join(B, "metadata", dst))
        write_defects(data_dir)
        write_family_lists(data_dir)
    os.makedirs(os.path.join(B, "outputs"), exist_ok=True)
    for f in sorted(os.listdir(os.path.join(B, "configs"))):
        if f.endswith(".json") and not os.path.exists(os.path.join(B, "outputs", f)):
            shutil.copy(os.path.join(B, "configs", f), os.path.join(B, "outputs", f))
    print("links ready: release, %s/core, splits/v2/, metadata/, outputs/" % name)


def main():
    """Download the review sample unless it is there, then create the links."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default=URL, help="download link of the archive")
    ap.add_argument("--sha256", default=SHA256, help="expected SHA-256 of the archive")
    ap.add_argument("--data-dir", default=os.path.join(B, "BLISS-1.0"), help="where the release is unpacked")
    ap.add_argument("--links-only", action="store_true", help="only create the links (release already in place)")
    a = ap.parse_args()
    if not a.links_only:
        if os.path.exists(os.path.join(a.data_dir, "shared.npz")):
            print("%s already holds the data" % a.data_dir)
        else:
            fetch(a.url, a.sha256, a.data_dir)
    make_links(a.data_dir)


if __name__ == "__main__":
    main()
