"""Synchronous, stat-validated ``zarr.json`` resolution on a LocalStore."""

from __future__ import annotations

import json
import os
import time

import pytest
import zarr
from zarr.storage import LocalStore, MemoryStore

import zarr_vectors.core._local_meta as local_meta
from zarr_vectors.core._local_meta import resolve_local_node


def _age(fn, seconds=10):
    """Backdate ``fn`` so it is old enough to be cached."""
    old = time.time_ns() - seconds * 1_000_000_000
    os.utime(fn, ns=(old, old))


@pytest.fixture
def root(tmp_path):
    g = zarr.open_group(LocalStore(tmp_path / "s"), mode="w", zarr_format=3)
    g.create_array("a", shape=(4,), dtype="i4", attributes={"k": [1, 2]})
    g.create_group("g", attributes={"x": 1})
    for path in ("a", "g"):
        _age(tmp_path / "s" / path / "zarr.json")
    return g


def test_matches_zarr(root):
    for path in ("a", "g"):
        for _ in range(2):  # miss, then hit
            node = resolve_local_node(root, path)
            ref = root[path]
            assert type(node) is type(ref)
            assert node.metadata == ref.metadata
            assert node.path == ref.path
    assert resolve_local_node(root, "missing") is None


def test_hit_is_served_from_cache(root):
    resolve_local_node(root, "a")
    fn = os.path.join(str(root.store.root), "a", "zarr.json")
    assert fn in local_meta._CACHE


def test_fresh_document_is_not_cached(root, tmp_path):
    root["g"].attrs["x"] = 2  # just written: inside the racy window
    fn = str(tmp_path / "s" / "g" / "zarr.json")
    local_meta._CACHE.pop(fn, None)
    assert resolve_local_node(root, "g").attrs["x"] == 2
    assert fn not in local_meta._CACHE


def test_rewrite_by_rename_is_seen(root, tmp_path):
    resolve_local_node(root, "a")
    zarr.open_array(LocalStore(tmp_path / "s"), path="a", mode="r+").attrs["k"] = 3
    assert resolve_local_node(root, "a").attrs["k"] == 3


def test_same_size_rewrite_in_place_is_seen(root, tmp_path):
    fn = tmp_path / "s" / "a" / "zarr.json"
    assert resolve_local_node(root, "a").attrs["k"] == [1, 2]
    doc = fn.read_text()
    edited = doc.replace("1,", "3,")
    assert edited != doc and len(edited) == len(doc)
    with open(fn, "r+") as fh:  # same inode, same size
        fh.write(edited)
    assert resolve_local_node(root, "a").attrs["k"] == [3, 2]


def test_deleted_node_is_absent(root, tmp_path):
    resolve_local_node(root, "g")
    os.remove(tmp_path / "s" / "g" / "zarr.json")
    assert resolve_local_node(root, "g") is None


def test_attributes_are_private_to_each_node(root):
    first = resolve_local_node(root, "a")
    first.metadata.attributes["k"].append(99)
    first.metadata.attributes["new"] = 1
    again = resolve_local_node(root, "a")
    assert again.attrs.asdict() == {"k": [1, 2]}


def test_declines_other_stores():
    g = zarr.open_group(MemoryStore(), mode="w", zarr_format=3)
    g.create_group("g")
    with pytest.raises(NotImplementedError):
        resolve_local_node(g, "g")


def test_declines_consolidated_metadata(root, tmp_path):
    zarr.consolidate_metadata(LocalStore(tmp_path / "s"))
    consolidated = zarr.open_group(LocalStore(tmp_path / "s"), mode="r")
    assert consolidated.metadata.consolidated_metadata is not None
    with pytest.raises(NotImplementedError):
        resolve_local_node(consolidated, "a")
    doc = json.loads((tmp_path / "s" / "zarr.json").read_text())
    assert "consolidated_metadata" in doc
