"""Octree levels refuse every operation that does not understand them.

Pinned here:

* the marker codec: ``to_octree`` puts ``zarr_vectors_octree_chunks`` in
  the cell codecs of every per-chunk array it writes, so a zarr reader
  that does not register it -- an older zarr-vectors among them -- fails
  on the level instead of reading leaves as uniform cells; ``to_uniform``
  removes it (every ``zarr.json`` is again what it was), and the shard
  repackers keep it;
* writers, edits, pyramid builders, rechunkers and the uniform-cell level
  APIs refuse an octree level, before they change anything, with an
  error naming the way out; ``Group`` refuses any write to the level's
  handle unless the conversion marked it;
* ``validate`` checks an octree level on its own terms: a correct one
  passes, a broken one does not;
* ``select(limit=)`` works on an octree level, and ``rechunk`` /
  ``rechunk_spatial`` carry the store's ``octree_sharding`` marker.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import zarr_vectors as zv
from zarr_vectors.core.arrays import _ensure_array_dir, _ensure_chunk_arrays, open_write_session
from zarr_vectors.core.metadata import LevelMetadata
from zarr_vectors.core.store import (
    create_resolution_level,
    get_resolution_level,
    open_store,
    read_root_metadata,
)
from zarr_vectors.encoding.octree_marker import OCTREE_MARKER_CODEC
from zarr_vectors.exceptions import StoreError
from zarr_vectors.octree_chunks import (
    OctreeError,
    converting,
    is_octree_level,
    octree_levels,
    refuse_octree_write,
    to_octree,
    to_uniform,
)
from zarr_vectors.types.graphs import read_graph, write_graph
from zarr_vectors.types.points import read_points, write_points

BOX = 400.0
CELL = 25.0
REPO = Path(__file__).resolve().parents[1]
OLD_COMMIT = "467e80e"  # the last zarr-vectors before octree chunks


def _uneven(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.concatenate([rng.normal(120, 22, (n, 3)), rng.uniform(0, BOX, (n // 3, 3))])
    return np.clip(pos, 0.5, BOX - 0.5).astype(np.float32)


def _short_edges(pos, reach=5.0):
    """Pairs closer than ``reach`` along a sort -- short, so within adjacent leaves."""
    order = np.lexsort(pos.T[::-1])
    a, b = order[:-1], order[1:]
    keep = np.linalg.norm(pos[a] - pos[b], axis=1) < reach
    return np.column_stack([a[keep], b[keep]]).astype(np.int64)


def _graph(path, *, octree=True, shard=None, n=3000, seed=0):
    pos = _uneven(n, seed)
    write_graph(str(path), pos, _short_edges(pos), chunk_shape=(CELL,) * 3,
                bounds=([0] * 3, [BOX] * 3), object_ids=np.arange(len(pos)) // 40,
                shard_shape=shard)
    if octree:
        to_octree(str(path), target=200)
    return str(path)


def _points(path, *, octree=True, shard=None, n=3000, seed=1, **kw):
    pos = _uneven(n, seed)
    write_points(str(path), pos, chunk_shape=(CELL,) * 3, bounds=([0] * 3, [BOX] * 3),
                 object_ids=np.arange(len(pos)) // 40, shard_shape=shard, **kw)
    if octree:
        to_octree(str(path), target=200)
    return str(path)


def _files(path):
    out = {}
    for d, _, fs in os.walk(path):
        for f in fs:
            q = os.path.join(d, f)
            out[os.path.relpath(q, path)] = Path(q).read_bytes()
    return out


def _digest(path):
    h = hashlib.sha1()
    for k, v in sorted(_files(path).items()):
        h.update(k.encode())
        h.update(v)
    return h.hexdigest()


def _cell_codecs(path, array):
    codecs = json.loads(Path(path, "0", array, "zarr.json").read_text())["codecs"]
    if "sharding" in codecs[0]["name"]:
        codecs = codecs[0]["configuration"]["codecs"]
    return [c["name"] for c in codecs]


# -- the marker codec ---------------------------------------------------------


@pytest.mark.parametrize("shard", [None, 4])
def test_to_octree_marks_every_per_chunk_array_and_to_uniform_removes_it(tmp_path, shard):
    src = _graph(tmp_path / "u.zv", octree=False, shard=shard)
    before = _files(src)
    to_octree(src, target=200)
    level = Path(src, "0")
    arrays = [p.parent for p in level.rglob("zarr.json")
              if json.loads(p.read_text()).get("node_type") == "array"
              and "octree_chunks" not in p.parts and "object_index" not in p.parts]
    names = {str(a.relative_to(level)) for a in arrays}
    assert {"vertices", "vertex_fragments", "links/0/0.0.0"} <= names
    for a in arrays:
        assert OCTREE_MARKER_CODEC in _cell_codecs(src, a.relative_to(level)), a
    # The reads are the uniform level's (the marker changes no bytes).
    assert read_graph(src)["edge_count"] == read_graph(tmp_path / "u.zv")["edge_count"]

    to_uniform(src)
    after = _files(src)
    meta = {k for k in before if k.endswith("zarr.json")}
    assert meta == {k for k in after if k.endswith("zarr.json")}
    assert all(before[k] == after[k] for k in meta), "a zarr.json changed across the round trip"
    assert not any(OCTREE_MARKER_CODEC.encode() in after[k] for k in meta)


def test_points_round_trip_is_byte_identical(tmp_path):
    src = _points(tmp_path / "p.zv", octree=False)
    before = _files(src)
    to_octree(src, target=200)
    assert OCTREE_MARKER_CODEC in _cell_codecs(src, "vertices")
    to_uniform(src)
    assert _files(src) == before


def test_a_zarr_reader_without_the_marker_refuses_the_level(tmp_path):
    """Plain zarr, no zarr_vectors import: the octree level's arrays do not
    open, the uniform level's do."""
    octree = _points(tmp_path / "o.zv")
    uniform = _points(tmp_path / "u.zv", octree=False)
    code = (
        "import sys, zarr\n"
        "for p in sys.argv[1:]:\n"
        "    try:\n"
        "        zarr.open_array(p + '/0/vertices', mode='r')\n"
        "        print('opened')\n"
        "    except Exception as e:\n"
        "        print(type(e).__name__, e)\n"
    )
    out = subprocess.run([sys.executable, "-c", code, octree, uniform],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    assert out[0].startswith("UnknownCodecError") and OCTREE_MARKER_CODEC in out[0]
    assert out[1] == "opened"


def _old_tree(tmp_path_factory):
    try:
        subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{OLD_COMMIT}^{{commit}}"],
                       check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"commit {OLD_COMMIT} is not in this checkout")
    tree = tmp_path_factory.mktemp("old")
    archive = subprocess.run(["git", "-C", str(REPO), "archive", OLD_COMMIT, "zarr_vectors"],
                             check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tree)], input=archive, check=True)
    return tree


_OLD_READS = r"""
import sys, warnings
warnings.simplefilter("ignore")
import zarr_vectors as zv
from zarr_vectors.types.graphs import read_graph
from zarr_vectors.types.points import read_points
box = ([40, 40, 40], [160, 160, 160])
for kind, p in zip(sys.argv[1::2], sys.argv[2::2]):
    calls = {
        "all": lambda: (read_points if kind == "p" else read_graph)(p),
        "bbox": lambda: (read_points if kind == "p" else read_graph)(p, bbox=box),
        "select": lambda: zv.open(p).level(0).select(bbox=box).read(),
    }
    if kind == "p":
        calls["objects"] = lambda: read_points(p, object_ids=[3, 7])
        calls["chunks"] = lambda: read_points(p, chunks=[(4, 4, 4)])
    for name, fn in calls.items():
        try:
            r = fn()
            n = r["vertex_count"] if isinstance(r, dict) and "vertex_count" in r else (
                r["node_count"] if isinstance(r, dict) else r.vertex_count)
            print(kind, name, "read", n)
        except Exception as e:
            print(kind, name, "raised", type(e).__name__, str(e)[:120])
"""


def test_an_older_zarr_vectors_refuses_octree_levels(tmp_path, tmp_path_factory):
    """The reader of commit 467e80e (before octree chunks) raises on an
    octree level -- points and graphs, sharded or not, every read path --
    and reads a converted-back store as the original.  Before the marker
    it returned wrong rows and edges without an error."""
    old = _old_tree(tmp_path_factory)
    stores = {
        "po": _points(tmp_path / "po.zv"),
        "go": _graph(tmp_path / "go.zv"),
        "gs": _graph(tmp_path / "gs.zv", shard=4),
        "pu": _points(tmp_path / "pu.zv", octree=False),
        "gu": _graph(tmp_path / "gu.zv", octree=False),
    }
    rt = tmp_path / "rt.zv"
    shutil.copytree(stores["gu"], rt)
    to_octree(str(rt), target=200)
    to_uniform(str(rt))

    def run(*pairs):
        env = {**os.environ, "PYTHONPATH": str(old)}
        res = subprocess.run([sys.executable, "-c", _OLD_READS, *pairs], cwd=str(tmp_path),
                             capture_output=True, text=True, env=env, check=True)
        return [line.split(" ", 3) for line in res.stdout.splitlines()]

    lines = run("p", stores["po"], "g", stores["go"], "g", stores["gs"])
    assert len(lines) == 11
    for kind, name, what, rest in lines:
        assert what == "raised" and rest.startswith("UnknownCodecError"), (kind, name, rest)
        assert OCTREE_MARKER_CODEC in rest

    lines = run("p", stores["pu"], "g", stores["gu"], "g", str(rt))
    assert all(what == "read" for _, _, what, _ in lines), lines
    gu = [rest for kind, name, what, rest in lines[5:8]]
    assert gu == [rest for kind, name, what, rest in lines[8:11]]  # rt reads as the original


@pytest.mark.parametrize("unshard", [False, True])
def test_shard_repackers_keep_the_marker(tmp_path, unshard):
    from zarr_vectors.sharding.io import shard_store, unshard_store

    src = _graph(tmp_path / "g.zv")
    want = read_graph(src)
    shard_store(src, shard_shape=4)
    if unshard:
        unshard_store(src)
    for array in ("vertices", "vertex_fragments", "links/0/0.0.0"):
        assert OCTREE_MARKER_CODEC in _cell_codecs(src, array), array
    got = read_graph(src)
    assert got["edge_count"] == want["edge_count"] and got["node_count"] == want["node_count"]


def test_set_octree_keeps_the_marker(tmp_path):
    pytest.importorskip("zarr_octree")
    from zarr_vectors.sharding.io import set_octree

    src = _graph(tmp_path / "g.zv", shard=4)
    want = read_graph(src)["edge_count"]
    set_octree(src, True)
    codecs = json.loads(Path(src, "0", "vertices", "zarr.json").read_text())["codecs"]
    assert codecs[0]["name"] == "zarr_octree_sharding"
    assert OCTREE_MARKER_CODEC in [c["name"] for c in codecs[0]["configuration"]["codecs"]]
    assert read_graph(src)["edge_count"] == want


def test_cell_source_sees_through_the_marker(tmp_path):
    """The raw-cell paths (host prefetch, device fetch) still recognise
    an octree level's arrays: the marker is the identity on bytes."""
    from zarr_vectors.core._cells_on_disk import cell_source

    for shard in (None, 4):
        src = _points(tmp_path / f"p{shard}.zv", shard=shard)
        node = get_resolution_level(open_store(src), 0).zarr_group["vertices"]
        assert cell_source("vertices", node) is not None
        assert cell_source("vertices", node, any_codecs=True) is not None


# -- writers refuse -------------------------------------------------------------


def _refused(path, fn, match="has octree chunks", exc=OctreeError):
    before = _digest(path)
    with pytest.raises(exc, match=match):
        fn()
    assert _digest(path) == before, "the store changed before the refusal"


def test_writers_refuse_an_octree_level_and_change_nothing(tmp_path):
    g, p = _graph(tmp_path / "g.zv"), _points(tmp_path / "p.zv")
    rng = np.random.default_rng(9)
    pos2 = rng.uniform(10, 390, (50, 3)).astype(np.float32)
    # Dataset.add_* writes a store once, octree or not.
    held = dict(match="already holds", exc=StoreError)
    _refused(g, lambda: zv.open(g, mode="r+").add_graph(
        pos2, np.array([[0, 1]]), object_ids=np.full(50, 999)), **held)
    _refused(g, lambda: zv.open(g, mode="r+").add_points(pos2), **held)
    _refused(p, lambda: zv.open(p, mode="r+").add_points(pos2), **held)
    _refused(p, lambda: write_points(p, pos2, chunk_shape=(CELL,) * 3))
    _refused(g, lambda: write_graph(g, pos2, np.array([[0, 1]]), chunk_shape=(CELL,) * 3))
    msg = str(pytest.raises(
        OctreeError, write_points, p, pos2, chunk_shape=(CELL,) * 3).value)
    assert "to_uniform(store, level=0)" in msg


def test_the_same_writes_succeed_on_a_uniform_level(tmp_path):
    g, p = _graph(tmp_path / "g.zv", octree=False), _points(tmp_path / "p.zv", octree=False)
    pos2 = np.random.default_rng(9).uniform(10, 390, (50, 3)).astype(np.float32)
    # The refusal is the octree level's: on a uniform level the writers run.
    write_points(p, pos2, chunk_shape=(CELL,) * 3)
    write_graph(g, pos2, np.array([[0, 1]]), chunk_shape=(CELL,) * 3)


def test_converting_back_lets_the_write_through(tmp_path):
    p = _points(tmp_path / "p.zv")
    pos2 = np.random.default_rng(9).uniform(10, 390, (50, 3)).astype(np.float32)
    to_uniform(p)
    write_points(p, pos2, chunk_shape=(CELL,) * 3)


def test_edits_refuse_an_octree_level_and_change_nothing(tmp_path):
    g, p = _graph(tmp_path / "g.zv"), _points(tmp_path / "p.zv")

    def move():
        with zv.open(g, mode="r+").editing(in_place=True) as e:
            e.move_vertex(object=3, index=0, to=[100.0, 100.0, 100.0])

    def remove():
        with zv.open(p, mode="r+").editing(in_place=True) as e:
            e.remove_object(3)

    # Refused at the first edit staged, or before it, when the edit's own
    # read of the cell (read_chunk_vertices) is refused.
    _refused(g, move, match="an edit|read_chunk_vertices")
    _refused(p, remove, match="an edit|read_chunk_vertices")


def test_the_per_cell_readers_refuse_an_octree_level(tmp_path):
    # Tools that read cell by cell (e.g. a viewer's chunk server) would take
    # each leaf for one cell; read_points / read_graph read octree levels.
    from zarr_vectors.core.arrays import read_chunk_links, read_chunk_vertices
    g = _graph(tmp_path / "g.zv")
    level = get_resolution_level(open_store(g), 0)
    with pytest.raises(OctreeError, match="read_chunk_vertices"):
        read_chunk_vertices(level, (0, 0, 0))
    with pytest.raises(OctreeError, match="read_chunk_links"):
        read_chunk_links(level, (0, 0, 0))
    from zarr_vectors.core.arrays import list_chunk_keys
    from zarr_vectors.constants import VERTICES
    u = get_resolution_level(open_store(_graph(tmp_path / "u.zv", octree=False)), 0)
    assert read_chunk_vertices(u, list_chunk_keys(u, VERTICES)[0])


def test_lazy_writer_and_level_metadata_refuse(tmp_path):
    import warnings

    from zarr_vectors.core.store import update_level_metadata
    from zarr_vectors.exceptions import MetadataError
    from zarr_vectors.lazy.store import open_zv

    p = _points(tmp_path / "p.zv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        _refused(p, lambda: open_zv(p)[0].writer(), match="ZVWriter")
    lg = get_resolution_level(open_store(p, mode="r+"), 0)
    before = _digest(p)
    with pytest.raises(MetadataError, match="octree_chunks"):
        update_level_metadata(lg, octree_chunks=None)
    assert _digest(p) == before


def test_vacuum_and_rebin_refuse(tmp_path):
    from zarr_vectors.ops.vacuum import vacuum
    from zarr_vectors.rechunk.rebin import rebin_level

    p = _points(tmp_path / "p.zv")
    _refused(p, lambda: vacuum(open_store(p, mode="r+")), match="vacuum")
    _refused(p, lambda: rebin_level(p, (CELL / 2,) * 3), match="rebin_level")


def test_group_writes_to_an_octree_level_are_refused_unless_converting(tmp_path):
    p = _points(tmp_path / "p.zv")
    root = open_store(p, mode="r+")
    lg = get_resolution_level(root, 0)
    key = lg.list_chunks("vertices")[0]
    raw = lg.read_bytes("vertices", key)
    meta = read_root_metadata(root)
    for fn in (
        lambda: lg.write_bytes("vertices", key, raw),
        lambda: lg.write_cells("vertices", [(key, raw)]),
        lambda: lg.write_array_meta("vertices", {"x": 1}),
        lambda: lg.write_array("extra", np.zeros(3)),
        lambda: lg.delete_subtree("vertices"),
        lambda: _ensure_array_dir(lg, "vertex_attributes/extra"),
        lambda: _ensure_chunk_arrays(lg, [("vertex_attributes/extra", None)]),
        lambda: open_write_session(lg, bounds=meta.bounds, chunk_shape=(CELL,) * 3).__enter__(),
        lambda: create_resolution_level(root, 0, LevelMetadata(
            level=0, vertex_count=0, arrays_present=[])),
    ):
        _refused(p, fn)
    # The conversion's own handles pass (here a same-bytes rewrite).
    with converting(lg):
        lg.write_bytes("vertices", key, raw)
    assert read_points(p)["vertex_count"] == read_points(tmp_path / "p.zv")["vertex_count"]


def test_a_level_mid_conversion_is_refused_with_how_to_finish(tmp_path):
    p = _points(tmp_path / "p.zv", octree=False)
    lg = get_resolution_level(open_store(p, mode="r+"), 0)
    level = dict(lg.attrs["zarr_vectors_level"])
    lg.attrs.update({"zarr_vectors_level": {**level, "octree_chunks": {"converting": "to_octree"}}})
    lg = get_resolution_level(open_store(p, mode="r+"), 0)
    assert is_octree_level(lg)
    with pytest.raises(OctreeError, match="part way through to_octree"):
        refuse_octree_write(lg, "this write")
    with pytest.raises(OctreeError, match="part way through"):
        write_points(p, np.ones((3, 3), np.float32) * 50, chunk_shape=(CELL,) * 3)


# -- pyramids and rechunking ------------------------------------------------------


def test_build_pyramid_and_coarsen_refuse_an_octree_level(tmp_path):
    from zarr_vectors.multiresolution.coarsen import build_pyramid, coarsen_level

    g = _graph(tmp_path / "g.zv")
    _refused(g, lambda: zv.open(g, mode="r+").build_pyramid(factors=[(2.0, 1.0)]),
             match="building a pyramid")
    _refused(g, lambda: coarsen_level(g, 0, 1, coarsen_factor=2.0), match="coarsening")
    assert not Path(g, "1").exists()

    # An octree level 1 above a uniform level 0: coarsening from it, or any
    # pyramid build on the store, is refused too.
    p = _points(tmp_path / "p.zv", octree=False)
    build_pyramid(p, factors=[(2.0, 1.0)], cross_level_depth=0)
    to_octree(p, target=100, level=1)
    assert octree_levels(open_store(p)) == [1]
    _refused(p, lambda: coarsen_level(p, 1, 2, coarsen_factor=2.0), match="level 1")
    _refused(p, lambda: build_pyramid(p, factors=[(2.0, 1.0)], cross_level_depth=0))


def test_rechunk_spatial_refuses_an_octree_level(tmp_path):
    from zarr_vectors.rechunk import rechunk_spatial

    g = _graph(tmp_path / "g.zv")
    out = tmp_path / "out.zv"
    _refused(g, lambda: rechunk_spatial(g, str(out), (50.0,) * 3), match="rechunk_spatial")
    assert not out.exists()


def test_rechunk_refuses_an_octree_level(tmp_path):
    from zarr_vectors.rechunk import rechunk_by_attribute

    rng = np.random.default_rng(0)
    p = _points(tmp_path / "p.zv", object_attributes={"cluster": rng.integers(0, 5, 100)})
    out = tmp_path / "out.zv"
    _refused(p, lambda: rechunk_by_attribute(p, "cluster", output=str(out)), match="rechunk")
    assert not out.exists()


def _octree_sharded(path, **kw):
    pytest.importorskip("zarr_octree")
    from zarr_vectors.sharding.io import set_octree

    src = kw.pop("make")(path, octree=False, shard=4, **kw)
    set_octree(src, True)
    assert read_root_metadata(open_store(src)).octree_sharding is not None
    return src


def _octree_arrays(path):
    names = set()
    for p in Path(path).rglob("zarr.json"):
        d = json.loads(p.read_text())
        if d.get("node_type") == "array" and d.get("codecs"):
            names.add(d["codecs"][0]["name"])
    return names


def test_rechunk_spatial_carries_the_octree_sharding_marker(tmp_path):
    from zarr_vectors.rechunk import rechunk_spatial

    src = _octree_sharded(tmp_path / "g.zv", make=_graph)
    out = tmp_path / "out.zv"
    rechunk_spatial(src, str(out), (50.0,) * 3)
    meta = read_root_metadata(open_store(str(out)))
    assert meta.octree_sharding == read_root_metadata(open_store(src)).octree_sharding
    assert "zarr_octree_sharding" in _octree_arrays(out)
    assert "sharding_indexed" not in _octree_arrays(out)
    assert read_graph(str(out))["node_count"] == read_graph(src)["node_count"]


def test_rechunk_carries_the_octree_sharding_marker(tmp_path):
    from zarr_vectors.rechunk import rechunk_by_attribute

    rng = np.random.default_rng(0)
    src = _octree_sharded(tmp_path / "p.zv", make=_points,
                          object_attributes={"cluster": rng.integers(0, 5, 100)})
    out = tmp_path / "out.zv"
    rechunk_by_attribute(src, "cluster", output=str(out))
    meta = read_root_metadata(open_store(str(out)))
    assert meta.octree_sharding == read_root_metadata(open_store(src)).octree_sharding
    assert "zarr_octree_sharding" in _octree_arrays(Path(out, "0"))


# -- validate and the level api ---------------------------------------------------------


@pytest.mark.parametrize("make,shard", [(_graph, None), (_graph, 4), (_points, None)])
def test_validate_passes_a_correct_octree_level(tmp_path, make, shard):
    src = make(tmp_path / "s.zv", shard=shard)
    result = zv.open(src).validate(level=5)
    assert result.ok, result.errors
    assert any("octree chunks" in p for p in result.passed)


def test_validate_catches_a_broken_octree_level(tmp_path):
    from zarr_vectors.validate import validate

    import zarr

    src = _graph(tmp_path / "g.zv")
    leaves = zarr.open_array(f"{src}/0/octree_chunks/leaves", mode="r+")
    leaves[0, 4] = int(leaves[0, 4]) + 1  # a leaf weight the leaf does not hold
    errors = validate(src, level=3).errors
    assert any("than the leaf table weight" in e for e in errors), errors

    src = _graph(tmp_path / "h.zv")
    vz = Path(src, "0", "vertices", "zarr.json")
    d = json.loads(vz.read_text())
    d["codecs"] = [c for c in d["codecs"] if c["name"] != OCTREE_MARKER_CODEC]
    vz.write_text(json.dumps(d))
    result = validate(src, level=3)
    assert any(OCTREE_MARKER_CODEC in w for w in result.warnings), result.warnings

    src = _points(tmp_path / "m.zv")
    lg = get_resolution_level(open_store(src, mode="r+"), 0)
    level = dict(lg.attrs["zarr_vectors_level"])
    lg.attrs.update({"zarr_vectors_level": {**level, "octree_chunks": {"converting": "to_uniform"}}})
    assert any("part way through to_uniform" in e for e in validate(src, level=3).errors)


def test_select_limit_reads_an_octree_level(tmp_path):
    for make in (_graph, _points):
        src = make(tmp_path / f"{make.__name__}.zv")
        level = zv.open(src).level(0)
        got = level.select(limit=10).read()
        assert got.vertex_count == 10
        every = {tuple(r) for r in np.round(level.select().read().positions, 4).tolist()}
        assert {tuple(r) for r in np.round(got.positions, 4).tolist()} <= every


def test_cell_apis_refuse_an_octree_level(tmp_path):
    from zarr_vectors.core.cells import read_neighbourhood

    src = _graph(tmp_path / "g.zv")
    level = zv.open(src).level(0)
    with pytest.raises(OctreeError, match="Level.cells"):
        level.cells()
    with pytest.raises(OctreeError, match="Query.cells"):
        level.select(bbox=([40] * 3, [160] * 3)).cells()
    lg = get_resolution_level(open_store(src), 0)
    with pytest.raises(OctreeError, match="read_neighbourhood"):
        read_neighbourhood(lg, (4, 4, 4))
    # Uniform levels are untouched.
    u = _graph(tmp_path / "u.zv", octree=False)
    assert len(list(zv.open(u).level(0).cells())) > 0
