"""``defer_presence``: a build that writes no state shared between cells.

``nonempty_chunks`` is one attribute per array, so each stamp rewrites
the whole list: disjoint writers race on it, and a build of N cells costs
O(N²) metadata bytes.  A deferred level carries no manifest while it is
built, every stamp is a no-op, readers ask the store, and one rebuild at
the end writes each manifest once.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors.building import (
    defer_presence,
    get_resolution_level,
    rebuild_presence,
    write_chunk_vertices,
)
from zarr_vectors.core.arrays import (
    create_links_array,
    finalize_links,
    read_link_attributes,
    read_links,
    write_link_attribute_cells,
    write_link_cells,
)
from zarr_vectors.core.group import _NONEMPTY_CHUNKS_ATTR, _PRESENCE_DECL_ATTR
from zarr_vectors.core.store import create_store, open_store

CELLS = [(0, 0, 0), (1, 0, 0), (2, 3, 1), (4, 4, 4)]


def _store(path, **kw):
    create_store(
        str(path),
        bounds=([0.0, 0.0, 0.0], [500.0, 500.0, 500.0]),
        chunk_shape=(100.0, 100.0, 100.0),
        geometry_types=["point_cloud"],
        ndim=3,
        **kw,
    )
    return get_resolution_level(open_store(str(path), mode="r+"), 0)


def _write(level, cell):
    pts = np.full((3, 3), 50.0, dtype=np.float32) + np.asarray(cell) * 100.0
    write_chunk_vertices(level, cell, [pts])


def _key(cell):
    return ".".join(str(c) for c in cell)


@pytest.fixture
def metadata_writes(monkeypatch) -> list[str]:
    from zarr.storage import LocalStore

    keys: list[str] = []
    original = LocalStore.set

    async def counting(self, key, value, *a, **kw):
        if key.endswith("zarr.json"):
            keys.append(key)
        return await original(self, key, value, *a, **kw)

    monkeypatch.setattr(LocalStore, "set", counting)
    return keys


def test_declaring_drops_the_manifests_and_marks_the_level(tmp_path):
    level = _store(tmp_path / "s.zv")
    _write(level, CELLS[0])
    assert level.list_chunks("vertices") == ["0.0.0"]

    names = defer_presence(level)

    assert "vertices" in names
    assert level.presence_deferred()
    assert level.zarr_group.attrs[_PRESENCE_DECL_ATTR] == "deferred"
    assert _NONEMPTY_CHUNKS_ATTR not in level.read_array_meta("vertices")
    # ...and the cell written before the declaration is still found.
    assert level.list_chunks("vertices") == ["0.0.0"]


def test_an_array_allocated_after_declaring_is_born_without_one(tmp_path):
    level = _store(tmp_path / "s.zv")
    defer_presence(level)
    level.create_sharded_chunk_array("late", (5, 5, 5))
    assert _NONEMPTY_CHUNKS_ATTR not in level.read_array_meta("late")


def test_no_presence_is_written_during_the_build(tmp_path, metadata_writes):
    level = _store(tmp_path / "s.zv")
    _write(level, CELLS[0])  # allocate the arrays before counting
    defer_presence(level)
    metadata_writes.clear()

    for cell in CELLS[1:]:
        _write(level, cell)  # default record_presence=True, deliberately

    assert metadata_writes == [], metadata_writes


def test_cells_are_visible_before_the_rebuild(tmp_path):
    level = _store(tmp_path / "s.zv")
    defer_presence(level)
    for cell in CELLS:
        _write(level, cell)

    assert level.list_chunks("vertices") == sorted(_key(c) for c in CELLS)
    assert level.chunk_exists("vertices", "2.3.1")
    assert not level.chunk_exists("vertices", "3.3.3")


def test_disjoint_workers_both_keep_their_cells(tmp_path):
    path = tmp_path / "s.zv"
    coordinator = _store(path)
    defer_presence(coordinator)

    # Two workers, each on its own handle opened after the declaration.
    a = get_resolution_level(open_store(str(path), mode="r+"), 0)
    b = get_resolution_level(open_store(str(path), mode="r+"), 0)
    _write(a, CELLS[0])
    _write(b, CELLS[1])
    _write(a, CELLS[2])
    _write(b, CELLS[3])

    rebuild_presence(coordinator)
    reopened = get_resolution_level(open_store(str(path), mode="r"), 0)
    assert reopened.list_chunks("vertices") == sorted(_key(c) for c in CELLS)


def test_the_rebuild_leaves_what_an_undeferred_build_leaves(tmp_path):
    plain = _store(tmp_path / "plain.zv")
    for cell in CELLS:
        _write(plain, cell)

    deferred = _store(tmp_path / "deferred.zv")
    defer_presence(deferred)
    for cell in CELLS:
        _write(deferred, cell)
    rebuild_presence(deferred)

    assert not deferred.presence_deferred()
    assert _PRESENCE_DECL_ATTR not in deferred.zarr_group.attrs
    for name in ("vertices", "vertex_fragments"):
        assert (
            deferred.read_array_meta(name)[_NONEMPTY_CHUNKS_ATTR]
            == plain.read_array_meta(name)[_NONEMPTY_CHUNKS_ATTR]
        )
    # Stamping resumes once the declaration is gone.
    deferred.create_sharded_chunk_array("after", (5, 5, 5))
    deferred.write_bytes("after", "1.1.1", b"x")
    assert deferred.read_array_meta("after")[_NONEMPTY_CHUNKS_ATTR] == ["1.1.1"]


def test_collected_stamps_are_not_folded_into_a_missing_manifest(tmp_path):
    level = _store(tmp_path / "s.zv")
    defer_presence(level)
    with level.collect_presence() as pending:
        for cell in CELLS[:2]:
            _write(level, cell)
    assert pending  # collected as usual...
    assert level.apply_presence(pending) == 0  # ...but nothing to fold into
    assert _NONEMPTY_CHUNKS_ATTR not in level.read_array_meta("vertices")
    assert level.list_chunks("vertices") == ["0.0.0", "1.0.0"]


def test_links_written_in_several_batches_survive(tmp_path):
    level = _store(tmp_path / "s.zv")
    defer_presence(level)
    create_links_array(level, link_width=2, delta=0, sid_ndim=3)
    p1 = write_link_cells(level, [[((0, 0, 0), 1), ((1, 0, 0), 2)]], sid_ndim=3)
    write_link_attribute_cells(
        level, "w", np.array([1.0], dtype=np.float32), partition=p1,
    )
    p2 = write_link_cells(level, [[((0, 0, 0), 3), ((1, 0, 0), 4)]], sid_ndim=3)
    write_link_attribute_cells(
        level, "w", np.array([2.0], dtype=np.float32), partition=p2,
    )
    rebuild_presence(level)
    finalize_links(level, delta=0)

    links = read_links(level, delta=0)
    attrs = read_link_attributes(level, "w", delta=0)
    assert dict(zip((tuple(r) for r in links), attrs)) == {
        (((0, 0, 0), 1), ((1, 0, 0), 2)): 1.0,
        (((0, 0, 0), 3), ((1, 0, 0), 4)): 2.0,
    }


def test_a_sharded_array_is_derived_from_its_shards(tmp_path):
    level = _store(tmp_path / "s.zv", shard_shape=2)
    defer_presence(level)
    for cell in CELLS:
        _write(level, cell)
    assert level._sharded_chunk_array("vertices").shards is not None
    assert level.list_chunks("vertices") == sorted(_key(c) for c in CELLS)
    rebuild_presence(level)
    assert level.read_array_meta("vertices")[_NONEMPTY_CHUNKS_ATTR] == sorted(
        _key(c) for c in CELLS
    )


def test_a_deferred_store_can_be_repacked_before_its_rebuild(tmp_path):
    # shard_store selects cells through list_chunks.  Under the per-call
    # flag alone that found nothing until the rebuild; a deferred level's
    # arrays are listed from the store, so repacking first loses nothing.
    from zarr_vectors.sharding import shard_store

    path = tmp_path / "s.zv"
    level = _store(path)
    defer_presence(level)
    for cell in CELLS:
        _write(level, cell)

    shard_store(str(path), shard_shape=2)

    level = get_resolution_level(open_store(str(path), mode="r+"), 0)
    assert level._sharded_chunk_array("vertices").shards is not None
    assert level.list_chunks("vertices") == sorted(_key(c) for c in CELLS)
    rebuild_presence(level)
    assert level.read_array_meta("vertices")[_NONEMPTY_CHUNKS_ATTR] == sorted(
        _key(c) for c in CELLS
    )


def test_the_validator_warns_about_a_level_left_deferred(tmp_path):
    from zarr_vectors.validate.metadata import validate_metadata

    path = tmp_path / "s.zv"
    level = _store(path)
    _write(level, CELLS[0])
    defer_presence(level)
    deferred = validate_metadata(str(path))
    assert deferred.ok
    assert any("presence is deferred" in w for w in deferred.warnings)

    rebuild_presence(level)
    assert not any("deferred" in w for w in validate_metadata(str(path)).warnings)
