"""Tests for the standard-Zarr-v3-array plumbing on :class:`Group`.

Covers :meth:`Group.write_array`, :meth:`Group.read_array`,
:meth:`Group.write_vlen_array`, :meth:`Group.read_vlen_array`, and the
matching ``read_array_attrs`` / ``standalone_array_exists`` helpers.

These methods are the foundation of the v0.7 layout migration (see plan
``can-you-do-an-compressed-wilkes.md``): every logical array becomes a
single standard Zarr v3 array at its logical path, readable by stock
``zarr`` tooling without library-specific conventions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from zarr_vectors.core.store import create_store
from zarr_vectors.exceptions import StoreError


# ---------------------------------------------------------------------------
# write_array / read_array — dense chunked arrays
# ---------------------------------------------------------------------------

def test_write_array_round_trip(tmp_store_path: Path):
    root = create_store(str(tmp_store_path))
    data = np.arange(120, dtype=np.float32).reshape(10, 12)

    root.write_array("object_attributes/intensity", data)

    out = root.read_array("object_attributes/intensity")
    np.testing.assert_array_equal(out, data)
    assert out.dtype == np.float32
    assert out.shape == (10, 12)


def test_write_array_creates_single_v3_array_no_child_groups(
    tmp_store_path: Path,
):
    """On disk, the path must be a standard Zarr v3 array — not a group
    containing ``data`` / ``present_mask`` child arrays.
    """
    root = create_store(str(tmp_store_path))
    data = np.arange(40, dtype=np.int64).reshape(4, 10)
    root.write_array("object_attributes/intensity", data)

    array_dir = tmp_store_path / "object_attributes" / "intensity"
    assert (array_dir / "zarr.json").is_file()
    # Standard v3 chunk key for the single (0, 0) chunk under default
    # encoding (separator='/').
    assert (array_dir / "c" / "0" / "0").is_file()
    # Absolutely no legacy children.
    for forbidden in ("data", "present_mask", "offsets"):
        assert not (array_dir / forbidden).exists(), (
            f"unexpected legacy child array {forbidden!r} at "
            f"{array_dir / forbidden}"
        )


def test_write_array_attributes_persist_on_the_array(tmp_store_path: Path):
    root = create_store(str(tmp_store_path))
    data = np.zeros(8, dtype=np.float32)
    root.write_array(
        "object_attributes/intensity",
        data,
        attributes={
            "zv_array": "object_attribute",
            "name": "intensity",
            "fill_sentinel_meaning": "absent",
        },
    )

    attrs = root.read_array_attrs("object_attributes/intensity")
    assert attrs["zv_array"] == "object_attribute"
    assert attrs["name"] == "intensity"
    assert attrs["fill_sentinel_meaning"] == "absent"


def test_write_array_is_readable_with_stock_zarr(tmp_store_path: Path):
    """The on-disk array must be loadable by plain ``zarr.open_array``
    with no knowledge of this library's conventions.
    """
    root = create_store(str(tmp_store_path))
    data = np.arange(24, dtype=np.float32).reshape(4, 6)
    root.write_array(
        "object_attributes/intensity",
        data,
        attributes={"zv_array": "object_attribute"},
    )

    from zarr.storage import LocalStore
    store = LocalStore(str(tmp_store_path))
    arr = zarr.open_array(store=store, path="object_attributes/intensity", mode="r")
    np.testing.assert_array_equal(np.asarray(arr[:]), data)
    assert dict(arr.attrs).get("zv_array") == "object_attribute"


def test_write_array_with_nan_fill_round_trips(tmp_store_path: Path):
    """``fill_value="NaN"`` survives a write/read cycle (spec-required
    JSON encoding for special floats).
    """
    root = create_store(str(tmp_store_path))
    data = np.array([1.0, np.nan, 3.0, np.nan], dtype=np.float32)
    root.write_array(
        "object_attributes/intensity",
        data,
        fill_value="NaN",
    )

    out = root.read_array("object_attributes/intensity")
    assert np.isnan(out[1]) and np.isnan(out[3])
    assert out[0] == 1.0 and out[2] == 3.0


def test_write_array_with_integer_sentinel_fill(tmp_store_path: Path):
    root = create_store(str(tmp_store_path))
    data = np.array([0, 5, -1, 7, -1], dtype=np.int32)
    sentinel = np.iinfo(np.int32).min
    # Replace the user-facing "-1 means absent" with the sentinel so the
    # written array stores the sentinel directly.
    written = np.where(data == -1, sentinel, data)
    root.write_array(
        "object_attributes/cluster_id",
        written,
        fill_value=int(sentinel),
    )

    out = root.read_array("object_attributes/cluster_id")
    np.testing.assert_array_equal(out, written)


def test_write_array_overwrite_replaces(tmp_store_path: Path):
    root = create_store(str(tmp_store_path))
    root.write_array("a/b", np.arange(5, dtype=np.int64))
    root.write_array("a/b", np.arange(8, dtype=np.float32))

    out = root.read_array("a/b")
    assert out.shape == (8,)
    assert out.dtype == np.float32


def test_write_array_with_explicit_chunks(tmp_store_path: Path):
    root = create_store(str(tmp_store_path))
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    root.write_array("x/y", data, chunks=(5, 5))

    # Stock zarr should report the chunk shape we asked for.
    from zarr.storage import LocalStore
    arr = zarr.open_array(LocalStore(str(tmp_store_path)), path="x/y", mode="r")
    assert tuple(arr.chunks) == (5, 5)
    np.testing.assert_array_equal(np.asarray(arr[:]), data)


def test_read_array_missing_raises(tmp_store_path: Path):
    root = create_store(str(tmp_store_path))
    with pytest.raises(StoreError, match="not found"):
        root.read_array("nope/nada")


def test_read_array_on_group_raises(tmp_store_path: Path):
    """An Option-G logical array (a group of chunk arrays) is not a
    standard array — ``read_array`` should reject it.
    """
    root = create_store(str(tmp_store_path))
    # Create the legacy Option-G layout: a group with a single-chunk
    # uint8 array inside.
    root.write_bytes("legacy/attr", "data", b"hello")
    with pytest.raises(StoreError, match="not an Array"):
        root.read_array("legacy/attr")


# ---------------------------------------------------------------------------
# write_vlen_array / read_vlen_array — ragged byte blobs
# ---------------------------------------------------------------------------

def test_write_vlen_array_round_trip(tmp_store_path: Path):
    root = create_store(str(tmp_store_path))
    blobs = [b"alpha", b"", b"beta gamma", b"\x00\x01\x02\x03"]
    root.write_vlen_array("groups", blobs)

    out = root.read_vlen_array("groups")
    assert out == blobs


def test_write_vlen_array_creates_single_array_no_child_groups(
    tmp_store_path: Path,
):
    root = create_store(str(tmp_store_path))
    root.write_vlen_array("groups", [b"x", b"yy", b"zzz"])

    array_dir = tmp_store_path / "groups"
    assert (array_dir / "zarr.json").is_file()
    # No legacy data + offsets siblings.
    for forbidden in ("data", "offsets", "present_mask"):
        assert not (array_dir / forbidden).exists()


def test_write_vlen_array_attributes_persist(tmp_store_path: Path):
    root = create_store(str(tmp_store_path))
    root.write_vlen_array(
        "groups",
        [b"a", b"b"],
        attributes={"zv_array": "groups", "num_groups": 2},
    )
    attrs = root.read_array_attrs("groups")
    assert attrs["zv_array"] == "groups"
    assert attrs["num_groups"] == 2


def test_write_vlen_array_empty_is_noop(tmp_store_path: Path):
    root = create_store(str(tmp_store_path))
    root.write_vlen_array("groups", [])
    # Nothing should have been created at "groups".
    assert "groups" not in list(iter(root._zarr))


def test_write_vlen_array_with_explicit_chunks(tmp_store_path: Path):
    root = create_store(str(tmp_store_path))
    blobs = [f"blob-{i}".encode() for i in range(7)]
    root.write_vlen_array("xs", blobs, chunks=3)

    from zarr.storage import LocalStore
    arr = zarr.open_array(LocalStore(str(tmp_store_path)), path="xs", mode="r")
    assert tuple(arr.chunks) == (3,)
    assert [bytes(b) for b in arr[:]] == blobs


# ---------------------------------------------------------------------------
# standalone_array_exists — discriminator for migration code paths
# ---------------------------------------------------------------------------

def test_standalone_array_exists_true_for_standard_array(
    tmp_store_path: Path,
):
    root = create_store(str(tmp_store_path))
    root.write_array("p/q", np.zeros(3, dtype=np.int32))
    assert root.standalone_array_exists("p/q")


def test_standalone_array_exists_false_for_legacy_group(
    tmp_store_path: Path,
):
    """A legacy Option-G logical array is a group, not an array —
    ``standalone_array_exists`` must say False so migration code can
    distinguish layouts.
    """
    root = create_store(str(tmp_store_path))
    root.write_bytes("legacy/attr", "data", b"hello")
    assert not root.standalone_array_exists("legacy/attr")
    # But the legacy ``array_exists`` (which checks for a group) does
    # report True — the two methods are complementary.
    assert root.array_exists("legacy/attr")


def test_standalone_array_exists_false_for_missing_path(
    tmp_store_path: Path,
):
    root = create_store(str(tmp_store_path))
    assert not root.standalone_array_exists("does/not/exist")
