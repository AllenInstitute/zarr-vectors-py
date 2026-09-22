"""``chunk_exists`` answers from a set, and never from a stale one.

It scanned the manifest list, which a writer asking per cell paid once
per cell against a handle held across the batch.  The set that replaces
the scan is keyed to the list object, so a stamp -- which always assigns
a new list -- is seen at once.
"""

from __future__ import annotations

from zarr_vectors.core.group import _NONEMPTY_CHUNKS_ATTR
from zarr_vectors.core.store import create_store


def _cell_array(root, name="members", grid_shape=(4, 4, 4)):
    root.create_sharded_chunk_array(name, grid_shape)
    return name


def test_a_stamp_is_seen_by_the_next_lookup(tmp_store_path):
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)
    root.write_bytes(name, "0.0.0", b"a")
    assert root.chunk_exists(name, "0.0.0")
    assert root.chunk_exists(name, "0.0.0")  # second sighting: the set
    assert not root.chunk_exists(name, "1.1.1")

    root.write_bytes(name, "1.1.1", b"b")
    assert root.chunk_exists(name, "1.1.1")
    root.write_bytes(name, "0.0.0", b"")
    assert not root.chunk_exists(name, "0.0.0")


def test_an_unsorted_manifest_still_answers_present(tmp_store_path):
    # Only the sharding spec says the list is sorted.  A bisection would
    # miss "0.0.0" here; a set cannot.
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)
    root.zarr_group[name].attrs[_NONEMPTY_CHUNKS_ATTR] = ["3.3.3", "0.0.0", "2.1.0"]
    with root.cached_nodes():
        for _ in range(3):
            assert root.chunk_exists(name, "0.0.0")
            assert root.chunk_exists(name, "2.1.0")
            assert not root.chunk_exists(name, "1.0.0")
