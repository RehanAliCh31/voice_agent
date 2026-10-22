"""Makes scripts/ importable from tests, so helper scripts are testable too."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
