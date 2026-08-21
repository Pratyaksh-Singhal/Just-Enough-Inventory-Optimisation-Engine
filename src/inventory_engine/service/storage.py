"""Where uploaded CSVs live."""

from __future__ import annotations

import hashlib
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

#: Read size while streaming an upload. 1 MiB is large enough that the syscall overhead is
#: irrelevant and small enough that peak memory stays flat regardless of file size.
CHUNK_BYTES = 1024 * 1024


class UploadTooLarge(Exception):
    """Raised when an upload exceeds the configured ceiling, mid-stream."""

    def __init__(self, limit_bytes: int) -> None:
        """Record the ceiling that was breached, for the 413 response body."""
        super().__init__(f"upload exceeds the {limit_bytes / 1024 / 1024:.0f} MB limit")
        self.limit_bytes = limit_bytes


@dataclass(frozen=True)
class StoredBlob:
    """A stored upload."""

    uri: str
    sha256: str
    byte_size: int


class BlobStorage(Protocol):
    """The storage operations this service needs, and no more."""

    def put(self, source: BinaryIO, *, suffix: str = ".csv", limit_bytes: int) -> StoredBlob:
        """Stream ``source`` into storage, returning its URI, digest and size."""
        ...

    def open(self, uri: str) -> BinaryIO:
        """Open a stored blob for reading."""
        ...

    def delete(self, uri: str) -> None:
        """Remove a stored blob. Silent if it is already gone."""
        ...


class LocalDiskStorage:
    """Blob storage on the local filesystem."""

    def __init__(self, root: Path) -> None:
        """Create the store rooted at ``root``, making the directory if it is absent."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, source: BinaryIO, *, suffix: str = ".csv", limit_bytes: int) -> StoredBlob:
        """Stream ``source`` to disk under a fresh name, hashing as it goes."""
        name = f"{uuid.uuid4().hex}{suffix}"
        path = self.root / name
        digest = hashlib.sha256()
        written = 0

        try:
            with path.open("wb") as out:
                while chunk := source.read(CHUNK_BYTES):
                    written += len(chunk)
                    if written > limit_bytes:
                        raise UploadTooLarge(limit_bytes)
                    digest.update(chunk)
                    out.write(chunk)
        except BaseException:
            path.unlink(missing_ok=True)
            raise

        return StoredBlob(uri=path.as_uri(), sha256=digest.hexdigest(), byte_size=written)

    def open(self, uri: str) -> BinaryIO:
        """Open a ``file://`` blob for reading."""
        return self._path_for(uri).open("rb")

    def delete(self, uri: str) -> None:
        """Remove a ``file://`` blob, ignoring one that is already gone."""
        self._path_for(uri).unlink(missing_ok=True)

    @staticmethod
    def _path_for(uri: str) -> Path:
        """Resolve a ``file://`` URI back to a path, on Windows and POSIX alike."""
        if not uri.startswith("file://"):
            raise ValueError(f"LocalDiskStorage cannot read {uri!r}; expected a file:// URI")
        from urllib.parse import unquote, urlparse

        path = unquote(urlparse(uri).path)
        # A drive letter directly after the leading slash means the Windows form.
        if re.match(r"^/[A-Za-z]:", path):
            path = path[1:]
        return Path(path)

    def free_bytes(self) -> int:
        """Bytes available on the volume holding the store, for the health probe."""
        return shutil.disk_usage(self.root).free
