"""The model file the live session runs locally, and where it comes from.

Silero VAD 6.2.3 (MIT), about 2 MB, kept in models/ (not committed), taken
from its own release on PyPI, whose hash PyPI publishes. Only the ONNX file
is extracted; none of the package's code is installed or run. It is checked
against a pinned SHA-256 before use, so a file that changed upstream or
arrived damaged is refused rather than run.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import io
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from voicebot.assistant.config import VadConfig
from voicebot.utils.logging import get_logger

log = get_logger("assistant.models")

SILERO_VERSION = "6.2.3"
SILERO_MEMBER = "silero_vad/data/silero_vad.onnx"

# Where the live session looks for it (live.vad_model_path).
SILERO_PATH = Path(VadConfig().model_path)

# What each file was when it was downloaded and verified for this project.
SHA256 = {
    "silero_vad.onnx": "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
}


@dataclass(frozen=True)
class ModelFile:
    path: Path
    present: bool
    verified: bool


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def status(path: Path = SILERO_PATH) -> list[ModelFile]:
    present = path.is_file()
    expected = SHA256.get(path.name)
    verified = present and (expected is None or _sha256(path.read_bytes()) == expected)
    return [ModelFile(path, present, verified)]


class ModelDownloadError(RuntimeError):
    """A model file could not be fetched, or was not the file it should be."""


# One more try after a pause: a release download that fails once on an
# office network usually works the second time, and a third rarely does.
RETRY_PAUSE_S = 2.0


def _fetch(url: str, timeout: float = 60.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return bytes(response.read())


def _worth_retrying(exc: Exception) -> bool:
    """A dropped or timed-out download is; a 404 or a 403 will only say so again."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code >= 500 or exc.code == 429
    return True


def _download(url: str, timeout: float = 60.0) -> bytes:
    """`_fetch`, tried once more after a pause if the network let it down."""
    try:
        return _fetch(url, timeout)
    except (OSError, http.client.HTTPException) as exc:  # URLError, timeouts, cut-off bodies
        if not _worth_retrying(exc):
            raise ModelDownloadError(f"could not download {url}: {exc}") from exc
        log.warning("downloading %s failed (%s); trying again", url, exc)
    time.sleep(RETRY_PAUSE_S)
    try:
        return _fetch(url, timeout)
    except (OSError, http.client.HTTPException) as exc:
        raise ModelDownloadError(f"could not download {url}: {exc}") from exc


def _save(path: Path, data: bytes) -> None:
    """Write `data` under `path` whole or not at all.

    Written to a temporary file beside it and moved into place, so a write
    that is interrupted never leaves half a model under the real name -- one
    that `status` would then report as merely damaged.
    """
    expected = SHA256.get(path.name)
    if expected is not None and _sha256(data) != expected:
        raise ModelDownloadError(
            f"{path.name} does not match its pinned SHA-256; refusing to use it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
    log.info("saved %s (%.1f MB)", path, len(data) / 1e6)


def _silero_onnx() -> bytes:
    """The Silero VAD model, out of the wheel PyPI publishes, checked against PyPI's hash."""
    try:
        meta = json.loads(_download(f"https://pypi.org/pypi/silero-vad/{SILERO_VERSION}/json", 15))
        wheel = next(f for f in meta["urls"] if f["filename"].endswith(".whl"))
        url, digest = wheel["url"], wheel["digests"]["sha256"]
    except (ValueError, TypeError, KeyError, StopIteration) as exc:
        raise ModelDownloadError(f"PyPI's description of silero-vad was unreadable: {exc}") from exc
    data = _download(url)
    if _sha256(data) != digest:
        raise ModelDownloadError("the Silero VAD release does not match the hash PyPI publishes")
    try:
        return zipfile.ZipFile(io.BytesIO(data)).read(SILERO_MEMBER)
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ModelDownloadError(f"the Silero VAD release has no {SILERO_MEMBER}") from exc


def download(path: Path = SILERO_PATH) -> list[ModelFile]:
    """Fetch it if it is missing or fails its check. Safe to run again.

    Raises ModelDownloadError if it could not be fetched or was not the one
    pinned, and OSError if it could not be written.
    """
    if not status(path)[0].verified:
        _save(path, _silero_onnx())
    return status(path)
