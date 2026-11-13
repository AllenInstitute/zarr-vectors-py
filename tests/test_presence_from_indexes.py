"""Presence from shard indexes, checked presence, stored objects, batched links.

- A sharded array's presence is derived from each shard's INDEX (one
  ranged read of its last bytes), not by decoding every cell: the answer
  must equal the whole-shard read's, over edge shards, a grid origin,
  empty payloads stored as such, small and large cells, an index that
  cannot be read, on a local store and a MemoryStore.
- ``set_presence(verify="objects"|"index")`` checks claims against the
  store before writing: missing, extra and partly claimed cells.
- ``stored_objects`` lists what the store holds, as ``cell_objects``
  names it.
- ``read_links`` reads its cells in one batched prefetch.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from zarr_vectors import building as zb
from zarr_vectors.core import _cells_on_disk as cod
from zarr_vectors.core import arrays as A
from zarr_vectors.exceptions import PresenceMismatchError

BOUNDS = ([0.0] * 3, [100.0] * 3)


def _level(where, *, shard=2, chunk=20.0):
    root = zb.create_store(
        where if not isinstance(where, str) else where,
        bounds=BOUNDS, chunk_shape=(chunk,) * 3, geometry_types=["polyline"],
        shard_shape=shard,
    )
    return zb.get_resolution_level(root, 0)


def _stores(tmp_path):
    return {
        "local": lambda: str(tmp_path / "s.zv"),
        "memory": lambda: zarr.storage.MemoryStore(),
    }


def _fill(lg, rng, n_cells, *, empties=0, name="vertices"):
    """Random cells of random sizes; ``empties`` written as b"" too."""
    grid = lg._sharded_chunk_array(name).shape
    picks = rng.choice(int(np.prod(grid)), size=n_cells + empties, replace=False)
    coords = [tuple(int(c) for c in np.unravel_index(p, grid)) for p in picks]
    written = []
    for i, c in enumerate(coords):
        key = ".".join(map(str, c))
        if i < n_cells:
            size = int(rng.choice([1, 4, 12, 40, 600]))
            lg.write_bytes(name, key, rng.integers(0, 255, size, dtype=np.uint8).tobytes())
            written.append(key)
        else:
            lg.write_bytes(name, key, b"")
    return sorted(written)


def _whole_shard_presence(lg, name, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(cod, "shard_cells", lambda node, shards: None)
        return sorted(lg._presence_from_store(name, lg._sharded_chunk_array(name), verify=True))


@pytest.mark.parametrize("store", ["local", "memory"])
@pytest.mark.parametrize("write_empty", [False, True])
def test_index_presence_equals_the_whole_shard_read(tmp_path, monkeypatch, store, write_empty):
    rng = np.random.default_rng(0)
    indexes = []
    real = cod.read_indexes

    def spy(src, keys):
        out = real(src, keys)
        indexes.extend(out)
        return out

    with zarr.config.set({"array.write_empty_chunks": write_empty}):
        lg = _level(_stores(tmp_path)[store](), shard=2)   # 5^3 grid: edge shards
        written = _fill(lg, rng, 40, empties=15)
        name = "vertices"
        with monkeypatch.context() as m:
            m.setattr(cod, "read_indexes", spy)
            got = lg.derive_nonempty_chunks(name)
    assert got == written
    # Every shard answered from its index, none read whole.
    assert indexes and all(isinstance(i, np.ndarray) for i in indexes)
    assert got == _whole_shard_presence(lg, name, monkeypatch)


def test_index_presence_honours_the_grid_origin(tmp_path, monkeypatch):
    root = zb.create_store(str(tmp_path / "s.zv"))
    root.create_sharded_chunk_array("sh", (5, 4, 3), shard_shape=(2, 2, 2), origin=(10, -3, 7))
    for key in ("10.-3.7", "14.0.9", "11.-2.8"):
        root.write_bytes("sh", key, b"payload")
    root.write_bytes("sh", "12.-1.7", b"")
    got = root.derive_nonempty_chunks("sh")
    assert got == sorted(["10.-3.7", "14.0.9", "11.-2.8"])
    assert got == _whole_shard_presence(root, "sh", monkeypatch)


def test_index_presence_reads_indexes_not_cells(tmp_path):
    rng = np.random.default_rng(1)
    lg = _level(str(tmp_path / "s.zv"), shard=5)   # one shard holds the grid
    written = _fill(lg, rng, 60)
    read = []
    real = cod._read_ranges

    def spy(items):
        read.extend(size for _p, _o, size in items)
        return real(items)

    cod._read_ranges = spy
    try:
        got = lg.derive_nonempty_chunks("vertices")
    finally:
        cod._read_ranges = real
    assert got == written
    # Only the small cells are read, to rule out a stored empty payload.
    assert all(0 < size <= cod.SMALL_CELL_BYTES for size in read)


def test_a_shard_whose_index_cannot_be_read_is_read_whole(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    lg = _level(str(tmp_path / "s.zv"), shard=2)
    written = _fill(lg, rng, 30)
    real = cod.read_indexes

    def flaky(src, keys):
        out = real(src, keys)
        out[0] = ValueError("shard index fails its crc32c check")
        return out

    monkeypatch.setattr(cod, "read_indexes", flaky)
    assert lg.derive_nonempty_chunks("vertices") == written


# --- stored_objects ----------------------------------------------------


@pytest.mark.parametrize("store", ["local", "memory"])
@pytest.mark.parametrize("shard", [None, 2])
def test_stored_objects_are_what_cell_objects_names(tmp_path, store, shard):
    rng = np.random.default_rng(3)
    lg = _level(_stores(tmp_path)[store](), shard=shard)
    written = _fill(lg, rng, 25, empties=5)
    coords = [tuple(int(c) for c in k.split(".")) for k in written]
    assert zb.stored_objects(lg, "vertices") == sorted(set(zb.cell_objects(lg, "vertices", coords)))


def test_stored_objects_skip_what_is_not_a_chunk(tmp_path):
    lg = _level(str(tmp_path / "s.zv"), shard=2)
    lg.write_bytes("vertices", "0.0.0", b"x")
    (key,) = zb.stored_objects(lg, "vertices")
    shard_dir = tmp_path / "s.zv" / key.rsplit("/", 1)[0]
    (shard_dir / "0.partial").write_bytes(b"left by a dead transaction")
    assert zb.stored_objects(lg, "vertices") == [key]
    with pytest.raises(zb.StoreError):
        zb.stored_objects(lg, "object_index")


# --- set_presence(verify=) ---------------------------------------------


def _manifest(lg, name="vertices"):
    return sorted(lg._sharded_chunk_array(name).attrs.get("nonempty_chunks") or [])


@pytest.mark.parametrize("store", ["local", "memory"])
@pytest.mark.parametrize("shard", [None, 2])
@pytest.mark.parametrize("verify", ["objects", "index"])
def test_exact_claims_pass_and_are_written(tmp_path, store, shard, verify):
    rng = np.random.default_rng(4)
    lg = _level(_stores(tmp_path)[store](), shard=shard)
    written = _fill(lg, rng, 30, empties=4)
    lg._sharded_chunk_array("vertices").attrs["nonempty_chunks"] = []
    assert zb.set_presence(lg, {"vertices": written}, verify=verify) == ["vertices"]
    assert _manifest(lg) == written


@pytest.mark.parametrize("shard", [None, 2])
@pytest.mark.parametrize("verify", ["objects", "index"])
def test_a_claimed_cell_with_no_object_is_missing(tmp_path, shard, verify):
    lg = _level(str(tmp_path / "s.zv"), shard=shard)
    written = _fill(lg, np.random.default_rng(5), 10)
    before = _manifest(lg)
    with pytest.raises(PresenceMismatchError) as err:
        zb.set_presence(lg, {"vertices": [*written, "4.4.4"]}, verify=verify)
    assert err.value.mismatches["vertices"]["missing"] == ["4.4.4"]
    assert _manifest(lg) == before  # nothing written


@pytest.mark.parametrize("verify", ["objects", "index"])
def test_an_unclaimed_cell_is_extra(tmp_path, verify):
    lg = _level(str(tmp_path / "s.zv"), shard=None)
    written = _fill(lg, np.random.default_rng(6), 10)
    with pytest.raises(PresenceMismatchError) as err:
        zb.set_presence(lg, {"vertices": written[1:]}, verify=verify)
    assert err.value.mismatches["vertices"] == {
        "missing": [], "extra": [written[0]], "extra_objects": [],
    }


@pytest.mark.parametrize("verify", ["objects", "index"])
def test_a_partly_claimed_shard_with_an_unclaimed_mate_is_caught(tmp_path, verify):
    lg = _level(str(tmp_path / "s.zv"), shard=2)
    for key in ("0.0.0", "0.0.1", "1.1.1"):
        lg.write_bytes("vertices", key, b"payload")
    with pytest.raises(PresenceMismatchError) as err:
        zb.set_presence(lg, {"vertices": ["0.0.0", "1.1.1"]}, verify=verify)
    assert err.value.mismatches["vertices"]["extra"] == ["0.0.1"]


@pytest.mark.parametrize("verify", ["objects", "index"])
def test_a_stored_shard_nobody_claimed_is_an_extra_object(tmp_path, verify):
    lg = _level(str(tmp_path / "s.zv"), shard=2)
    lg.write_bytes("vertices", "0.0.0", b"payload")
    lg.write_bytes("vertices", "4.4.4", b"payload")
    with pytest.raises(PresenceMismatchError) as err:
        zb.set_presence(lg, {"vertices": ["0.0.0"]}, verify=verify)
    found = err.value.mismatches["vertices"]
    assert found["extra_objects"] == zb.cell_objects(lg, "vertices", [(4, 4, 4)])


def test_index_mode_sees_a_claimed_cell_its_shard_does_not_hold(tmp_path):
    """The difference between the modes: a claim inside a stored shard."""
    lg = _level(str(tmp_path / "s.zv"), shard=2)
    lg.write_bytes("vertices", "0.0.0", b"payload")
    claims = {"vertices": ["0.0.0", "0.0.1"]}
    zb.set_presence(lg, claims, verify="objects")       # the shard is there
    with pytest.raises(PresenceMismatchError) as err:
        zb.set_presence(lg, claims, verify="index")       # the cell is not
    assert err.value.mismatches["vertices"]["missing"] == ["0.0.1"]


@pytest.mark.parametrize("store", ["local", "memory"])
def test_on_mismatch_derive_writes_what_the_store_holds(tmp_path, store):
    lg = _level(_stores(tmp_path)[store](), shard=2)
    written = _fill(lg, np.random.default_rng(7), 20)
    frag_written = _fill(lg, np.random.default_rng(8), 5, name="vertex_fragments")
    names = zb.set_presence(
        lg, {"vertices": written[:-3], "vertex_fragments": frag_written},
        verify="index", on_mismatch="derive",
    )
    assert names == ["vertex_fragments", "vertices"]
    assert _manifest(lg) == written
    assert _manifest(lg, "vertex_fragments") == frag_written


def test_verify_arguments_are_checked(tmp_path):
    lg = _level(str(tmp_path / "s.zv"), shard=None)
    with pytest.raises(zb.ArrayError):
        zb.set_presence(lg, {"vertices": []}, verify="cells")
    with pytest.raises(zb.ArrayError):
        zb.set_presence(lg, {"vertices": []}, verify="index", on_mismatch="ignore")


# --- read_links batched -----------------------------------------------


def test_read_links_reads_every_cell_in_one_prefetch(tmp_path, monkeypatch):
    lg = _level(str(tmp_path / "s.zv"), shard=None, chunk=50.0)
    rng = np.random.default_rng(9)
    links = [
        ((tuple(int(c) for c in rng.integers(0, 2, 3)), int(rng.integers(0, 50))),
         (tuple(int(c) for c in rng.integers(0, 2, 3)), int(rng.integers(0, 50))))
        for _ in range(200)
    ]
    A.write_links(lg, links, 3)
    want = A.read_links(lg)
    plans = []
    real = A._maybe_batched_reads

    def spy(level_group, plan):
        plans.append(plan)
        return real(level_group, plan)

    monkeypatch.setattr(A, "_maybe_batched_reads", spy)
    got = A.read_links(lg)
    assert got == want and len(got) == 200
    assert len(plans) == 1 and sum(len(keys) for _n, keys in plans[0]) > 1
    chunks, vi = A.read_link_arrays(lg)
    assert got == [tuple(zip(map(tuple, c.tolist()), v.tolist())) for c, v in zip(chunks, vi)]
    sel = [0, 7, 199]
    assert A.read_links(lg, select=sel) == [want[i] for i in sel]
