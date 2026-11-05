"""Byte-range reads within a cell (``Group.read_cell_range``).

An uncompressed vlen-bytes cell is its payload behind an 8-byte header,
so a reader that knows which rows it wants -- from a fragment index --
can fetch just those bytes.  These tests pin that the range read returns
exactly what slicing the whole cell would, on every layout, that it
falls back to the whole cell wherever it cannot do better, and that a
one-object mesh read on an uncompressed store reads only the object's
rows.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

import zarr_vectors as zv
import zarr_vectors.core._batch_reader as batch_reader
from zarr_vectors.core.arrays import (
    list_chunk_keys,
    read_fragment,
    read_vertex_fragment_index,
)
from zarr_vectors.core.group import Group
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.exceptions import StoreError
from zarr_vectors.types.meshes import read_mesh, write_mesh

from .test_meshes import BOUNDS, CHUNK, _face_set, _patch_mesh

LAYOUTS = {
    "none": dict(compressor=None),
    "none-sharded": dict(compressor=None, shard_shape=(2, 2, 2)),
    "zstd": dict(compressor="zstd"),
    "zstd-sharded": dict(compressor="zstd", shard_shape=(2, 2, 2)),
}


@pytest.fixture(params=sorted(LAYOUTS))
def layout_store(request, tmp_path):
    v, f, o = _patch_mesh()
    path = tmp_path / f"{request.param}.zv"
    write_mesh(str(path), v, f, chunk_shape=CHUNK, bounds=BOUNDS, object_ids=o,
               **LAYOUTS[request.param])
    return request.param, path, v, f, o


def _level(path):
    return get_resolution_level(open_store(str(path)), 0)


# ---------------------------------------------------------------------------
# The primitive
# ---------------------------------------------------------------------------


def test_range_equals_slice_of_whole_cell(layout_store):
    _, path, *_ = layout_store
    lg = _level(path)
    for cc in list_chunk_keys(lg):
        key = ".".join(map(str, cc))
        whole = lg.read_bytes("vertices", key)
        for a, b in [(0, 12), (12, 36), (len(whole) - 12, len(whole)), (0, len(whole))]:
            assert lg.read_cell_range("vertices", key, a, b) == whole[a:b]


def test_range_past_the_payload_matches_slicing(layout_store):
    """A short read is not served as a range; the fallback slices."""
    _, path, *_ = layout_store
    lg = _level(path)
    key = ".".join(map(str, list_chunk_keys(lg)[0]))
    whole = lg.read_bytes("vertices", key)
    assert lg.read_cell_range("vertices", key, len(whole) - 4, len(whole) + 40) == whole[-4:]
    assert lg.read_cell_range("vertices", key, 5, 5) == b""


def test_range_readable_follows_the_codec(layout_store):
    name, path, *_ = layout_store
    lg = _level(path)
    assert lg.range_readable("vertices") is name.startswith("none")


def test_compressed_store_is_never_range_read(layout_store, monkeypatch):
    name, path, *_ = layout_store
    if name.startswith("none"):
        pytest.skip("uncompressed")
    served = []
    real = batch_reader.fetch_ranges
    monkeypatch.setattr(
        batch_reader, "fetch_ranges",
        lambda *a, **k: served.append(out := real(*a, **k)) or out,
    )
    lg = _level(path)
    key = ".".join(map(str, list_chunk_keys(lg)[0]))
    lg.read_cell_range("vertices", key, 0, 12)
    assert all(not s for s in served)


def test_batched_range_reads_fetch_once_and_serve_from_cache(layout_store, monkeypatch):
    name, path, *_ = layout_store
    if not name.startswith("none"):
        pytest.skip("only uncompressed cells are range-read")
    lg = _level(path)
    keys = [".".join(map(str, cc)) for cc in list_chunk_keys(lg)[:3]]
    reqs = [("vertices", k, 0, 24) for k in keys]
    calls = []
    real = batch_reader.fetch_ranges
    monkeypatch.setattr(
        batch_reader, "fetch_ranges",
        lambda zg, rs, *a, **k: calls.append(len(rs)) or real(zg, rs, *a, **k),
    )
    with lg.batched_range_reads(reqs):
        got = [lg.read_cell_range(*r) for r in reqs]
    assert calls == [3]
    assert got == [lg.read_bytes("vertices", k)[:24] for k in keys]


def test_batched_range_reads_do_not_nest(layout_store):
    _, path, *_ = layout_store
    lg = _level(path)
    with lg.batched_range_reads([]):
        with pytest.raises(StoreError, match="nesting"):
            with lg.batched_range_reads([]):
                pass


def test_read_fragment_identical_on_every_layout(layout_store):
    _, path, v, _, o = layout_store
    lg = _level(path)
    for cc in list_chunk_keys(lg):
        fi = read_vertex_fragment_index(lg, cc)
        full = np.frombuffer(lg.read_bytes("vertices", ".".join(map(str, cc))),
                             dtype=np.float32).reshape(-1, 3)
        for f in range(fi.num_fragments):
            s, c = fi.range(f)
            assert np.array_equal(read_fragment(lg, cc, f), full[s:s + c])


# ---------------------------------------------------------------------------
# The mesh object read
# ---------------------------------------------------------------------------


def test_object_read_identical_on_every_layout(layout_store):
    _, path, v, f, o = layout_store
    for k in np.unique(o):
        got = read_mesh(str(path), object_ids=[int(k)])
        mask = o == k
        assert got["vertex_count"] == int(mask.sum())
        want = f[mask[f].all(axis=1)]
        assert _face_set(got["vertices"], got["faces"]) == _face_set(v, want)


def test_uncompressed_object_read_fetches_only_the_objects_rows(layout_store, monkeypatch):
    """The point of the layout: no whole vertex or intra-chunk face cell."""
    name, path, v, f, o = layout_store
    if not name.startswith("none"):
        pytest.skip("compressed cells are read whole by design")
    whole: list[str] = []
    real_flush = batch_reader.flush_prefetch
    real_read = Group.read_bytes

    def flush(zg, plan, *a, **k):
        whole.extend(n for n, keys in plan if keys)
        return real_flush(zg, plan, *a, **k)

    def read(self, array_name, chunk_key):
        cached = self._prefetch_cache is not None and (
            (array_name, chunk_key) in self._prefetch_cache
        )
        if not cached:
            whole.append(array_name)
        return real_read(self, array_name, chunk_key)

    monkeypatch.setattr(batch_reader, "flush_prefetch", flush)
    monkeypatch.setattr(Group, "read_bytes", read)
    got = read_mesh(str(path), object_ids=[3])
    assert got["vertex_count"] == int((o == 3).sum())
    assert "vertices" not in whole
    assert not any(n.startswith("links/0/0.0.0_0.0.0") for n in whole)


def test_async_object_read_matches_sync(layout_store):
    """The offline replay path is cell-granular; it must still be right."""
    _, path, *_ = layout_store
    level = zv.open(str(path)).level(0)
    sync = level.select(objects=[3]).read()
    out = asyncio.run(level.select(objects=[3]).aread())
    assert np.array_equal(out.positions, sync.positions)
    assert np.array_equal(out.faces, sync.faces)
