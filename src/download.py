"""Download the sirens dataset at a PINNED commit and verify file hashes.

The upstream repo gets new commits daily; pinning makes every result reproducible.
"""
import hashlib
import subprocess
import sys

from data import DATASET_DIR

REPO_URL = "https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset.git"
COMMIT = "bf0c323c78e3e4508721458797dcd38c95df13f3"   # 2026-10-11 04:50 UTC, "Add new data"
SHA256 = {
    "datasets/official_data_en.csv": "b396f87696243069eec0219880e87dcd642b20beb0a7bae0d0e1f61e1ed20ec4",
    "datasets/volunteer_data_en.csv": "d03747ee8bf056032b2eeef50cdcf964d65fa60a92243e91862db48b31377465",
}


def git(*args):
    subprocess.run(["git", "-C", str(DATASET_DIR), *args], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not (DATASET_DIR / ".git").exists():
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        git("init", "-q")
        git("remote", "add", "origin", REPO_URL)
    head = subprocess.run(["git", "-C", str(DATASET_DIR), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if head != COMMIT:
        print(f"fetching {REPO_URL} @ {COMMIT[:10]} ...")
        git("fetch", "--depth", "1", "origin", COMMIT)
        git("checkout", "-q", "--force", COMMIT)
    for rel, expected in SHA256.items():
        got = sha256(DATASET_DIR / rel)
        if got != expected:
            sys.exit(f"hash mismatch for {rel}: {got}")
    print(f"dataset OK: commit {COMMIT}, {len(SHA256)} files verified by SHA-256")


if __name__ == "__main__":
    main()
