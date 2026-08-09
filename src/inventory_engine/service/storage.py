"""Where uploaded CSVs live.

Not in memory, and not in the database. An upload is streamed to a blob store in fixed
chunks, hashed as it goes, and the row that references it carries only a URI. Two reasons,
in order of how much they matter:

1. A 50 MB CSV read with ``await file.read()`` is 50 MB of resident memory per concurrent
   upload, plus whatever pandas allocates on top when it parses. Streaming makes the
   memory cost of an upload independent of its size.
2. The worker is a different process, and may be a different machine. A file the API held
   in memory is a file the worker cannot see.

``LocalDiskStorage`` is the implementation today. The interface is the small subset of S3
semantics that this service needs, so an ``S3Storage`` is a drop-in: put bytes, get a
readable handle, delete.
"""

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
    """Raised when an upload exceeds the configured ceiling, mid-stream.

    Raised while streaming rather than after, so the ceiling is a real bound on what
    touches the disk instead of a check that happens once the damage is done.
    """

    def __init__(self, limit_bytes: int) -> None:
        """Record the ceiling that was breached, for the 413 response body."""
        super().__init__(f"upload exceeds the {limit_bytes / 1024 / 1024:.0f} MB limit")
        self.limit_bytes = limit_bytes


@dataclass(frozen=True)
class StoredBlob:
    """A stored upload.

    Attributes:
        uri: How to fetch it back. ``file://`` today; ``s3://`` under another backend.
        sha256: Hex digest of the bytes, computed during the same pass that wrote them.
        byte_size: Bytes written.

    """

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
    """Blob storage on the local filesystem.

    Adequate for a single API and a worker sharing a volume, which is what Compose gives
    us. The moment those are on separate hosts this must become S3-compatible -- the
    interface is unchanged, only the class swaps.
    """

    def __init__(self, root: Path) -> None:
        """Create the store rooted at ``root``, making the directory if it is absent."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, source: BinaryIO, *, suffix: str = ".csv", limit_bytes: int) -> StoredBlob:
        """Stream ``source`` to disk under a fresh name, hashing as it goes.

        Raises:
            UploadTooLarge: If ``limit_bytes`` is exceeded. The partial file is removed
                before the exception propagates, so a rejected upload leaves nothing behind.

        """
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
        """Resolve a ``file://`` URI back to a path, on Windows and POSIX alike.

        The two platforms disagree about the leading slash, and getting it wrong is not a
        cosmetic difference -- it turns an absolute path into a relative one.

        * Windows: ``file:///D:/data/x.csv`` -> ``urlparse().path`` is ``/D:/data/x.csv``.
          The slash is an artefact of the URI form and must go, leaving ``D:/data/x.csv``.
        * POSIX: ``file:///data/uploads/x.csv`` -> ``urlparse().path`` is
          ``/data/uploads/x.csv``. That slash is the root and must **stay**.

        The first version stripped it unconditionally, which worked on the machine it was
        written on and broke the moment the service ran in a Linux container: the API wrote
        to ``/data/uploads/x.csv`` and then tried to read ``data/uploads/x.csv`` relative to
        the working directory, giving ``FileNotFoundError`` on every single upload. Found by
        building the image and running it, not by any test on Windows -- which is the point
        of running it.

        Raises:
            ValueError: For any other scheme, rather than silently treating the URI as a
                relative path and reading something unintended.

        """
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
