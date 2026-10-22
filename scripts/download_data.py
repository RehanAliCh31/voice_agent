"""Download and verify the CLINC150 dataset.

CLINC150 is mirrored on Kaggle under several slugs, so the slug is passed in
rather than hard-coded. Find one with:

    kaggle datasets list -s clinc150
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REQUIRED_SPLITS = ("train", "val", "test", "oos_train", "oos_val", "oos_test")


def verify_clinc_json(path: Path) -> dict[str, int]:
    """Check a CLINC150 JSON file and return the row count per split.

    Raises ValueError naming the file and the problem, never a bare KeyError.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CLINC150 file not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object, got {type(payload).__name__}")

    missing = [key for key in REQUIRED_SPLITS if key not in payload]
    if missing:
        raise ValueError(
            f"{path} is missing required splits {missing}. "
            f"Expected keys: {list(REQUIRED_SPLITS)}"
        )

    counts: dict[str, int] = {}
    for split in REQUIRED_SPLITS:
        rows = payload[split]
        if not isinstance(rows, list):
            raise ValueError(f"{path} split {split!r} must be a list, got {type(rows).__name__}")
        for index, row in enumerate(rows):
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                raise ValueError(
                    f"{path} split {split!r} row {index} is not a [text, label] pair: {row!r}"
                )
        counts[split] = len(rows)
    return counts


def download(dataset: str, out_dir: Path) -> None:
    """Fetch and unzip a Kaggle dataset into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(out_dir), "--unzip"],
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="Kaggle dataset slug, e.g. owner/clinc150")
    parser.add_argument("--out", default="data/raw", help="Destination directory")
    parser.add_argument("--verify", help="Verify an existing CLINC150 JSON file and exit")
    args = parser.parse_args(argv)

    if args.verify:
        counts = verify_clinc_json(Path(args.verify))
        for split, count in counts.items():
            print(f"{split:12s} {count:6d}")
        print(f"{'total':12s} {sum(counts.values()):6d}")
        return 0

    if not args.dataset:
        parser.error("--dataset is required unless --verify is given")

    download(args.dataset, Path(args.out))
    print(f"downloaded {args.dataset} into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
