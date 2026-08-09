"""Blob storage, including the URI forms of both platforms.

The path tests here run identically on Windows and Linux **by construction** -- they assert
on the resolved string rather than touching the filesystem. That matters because the bug
they exist to prevent was invisible on the machine the code was written on.
"""

from __future__ import annotations

import io

import pytest

from inventory_engine.service.storage import (
    LocalDiskStorage,
    StoredBlob,
    UploadTooLarge,
)


@pytest.fixture
def storage(tmp_path) -> LocalDiskStorage:
    """A store rooted in a temp directory."""
    return LocalDiskStorage(tmp_path / "uploads")


# --------------------------------------------------------------------------- uri forms


def test_a_posix_uri_stays_absolute():
    """Regression: the leading slash is the root on POSIX and must not be stripped.

    Stripping it turned ``/data/uploads/x.csv`` into ``data/uploads/x.csv`` -- relative to
    whatever the working directory happened to be. Every upload inside the Linux container
    failed with FileNotFoundError while the identical code worked on Windows.
    """
    path = LocalDiskStorage._path_for("file:///data/uploads/abc.csv")
    assert path.as_posix() == "/data/uploads/abc.csv"


def test_a_windows_uri_drops_the_slash_before_the_drive_letter():
    """``/D:/x`` is not a path; the slash is an artefact of the URI form."""
    path = LocalDiskStorage._path_for("file:///D:/Project/data/abc.csv")
    assert path.as_posix() == "D:/Project/data/abc.csv"


def test_percent_escapes_are_decoded():
    """Real project paths contain spaces, and the URI form escapes them."""
    path = LocalDiskStorage._path_for("file:///D:/Forecasting%20project/x.csv")
    assert "Forecasting project" in path.as_posix()
    assert "%20" not in path.as_posix()


def test_a_non_file_scheme_is_refused_rather_than_guessed():
    """Better a loud error than silently reading some relative path."""
    with pytest.raises(ValueError, match="expected a file:// URI"):
        LocalDiskStorage._path_for("s3://bucket/key.csv")


# --------------------------------------------------------------------------- round trip


def test_a_blob_round_trips(storage):
    blob = storage.put(io.BytesIO(b"sku,date,units_sold\nA,2025-01-01,3\n"), limit_bytes=10_000)
    assert isinstance(blob, StoredBlob)
    with storage.open(blob.uri) as handle:
        assert handle.read().startswith(b"sku,date")


def test_the_digest_and_size_describe_the_bytes_written(storage):
    payload = b"x" * 5000
    blob = storage.put(io.BytesIO(payload), limit_bytes=10_000)
    import hashlib

    assert blob.byte_size == 5000
    assert blob.sha256 == hashlib.sha256(payload).hexdigest()


def test_two_uploads_of_the_same_bytes_get_distinct_uris_and_equal_digests(storage):
    a = storage.put(io.BytesIO(b"same"), limit_bytes=100)
    b = storage.put(io.BytesIO(b"same"), limit_bytes=100)
    assert a.uri != b.uri
    assert a.sha256 == b.sha256


def test_delete_is_silent_when_the_blob_is_already_gone(storage):
    blob = storage.put(io.BytesIO(b"bye"), limit_bytes=100)
    storage.delete(blob.uri)
    storage.delete(blob.uri)  # must not raise


# --------------------------------------------------------------------------- the ceiling


def test_an_oversized_upload_is_stopped_mid_stream(storage):
    with pytest.raises(UploadTooLarge):
        storage.put(io.BytesIO(b"x" * 5000), limit_bytes=1000)


def test_a_rejected_upload_leaves_no_partial_file_behind(storage):
    """The ceiling has to bound what touches the disk, not just what is accepted."""
    with pytest.raises(UploadTooLarge):
        storage.put(io.BytesIO(b"x" * 5_000_000), limit_bytes=1000)
    assert list(storage.root.glob("*")) == []
