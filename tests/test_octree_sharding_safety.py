"""The store-wide octree marker and its conversion fail safe.

:func:`zarr_vectors.sharding.set_octree` builds each converted array in a
scratch copy and swaps it in only once complete, so an interrupted run
loses nothing and a re-run finishes it; the root's marker and capability
follow the arrays (``arrays=``, ``unshard_store``); a failed root read
is an error, not "unmarked"; a store sharded after the fact can be marked;
``Layout(octree={})`` asks for the octree as ``True`` does; and a bad
configuration, or no zarr-octree, fails before the store exists.

Every failure here is injected into the store a conversion writes
through, at each step of the conversion, on a local store (the swap is a
rename) and on a store that is not one (the swap is a copy).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from zarr.storage import LocalStore, WrapperStore

import zarr_vectors as zv
from zarr_vectors.exceptions import MetadataError

try:
    import zarr_octree  # noqa: F401

    HAVE_OCTREE = True
except ImportError:
    HAVE_OCTREE = False

needs_octree = pytest.mark.skipif(not HAVE_OCTREE, reason="zarr-octree is not installed")

BOX = 100.0
SCRATCH = "__set_octree"
TARGET = {"target": 2_000}


def _schema(octree=False, kind="point_cloud", shard_cells=4):
    return zv.Schema(bounds=([0, 0, 0], [BOX, BOX, BOX]), kind=kind,
                     layout=zv.Layout(cells=8, shard_cells=shard_cells, compression="zstd", octree=octree))


def _points(n=1_500, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(5, 95, (n, 3)).astype(np.float32), np.arange(n) // 50


def _make(path, octree=False, kind="point_cloud", shard_cells=4):
    pts, ids = _points()
    ds = zv.create(str(path), schema=_schema(octree, kind, shard_cells))
    if kind == "point_cloud":
        ds.add_points(pts, object_ids=ids)
    else:
        faces = np.stack([np.arange(0, len(pts) - 2, 3), np.arange(1, len(pts) - 1, 3),
                          np.arange(2, len(pts), 3)], 1)
        ds.add_mesh(pts, faces, object_ids=np.arange(len(pts)) // 150)
    return path


def _codecs(path):
    """{array path: sharding codec name} for every sharded array of the store."""
    out = {}
    for z in path.rglob("zarr.json"):
        m = json.loads(z.read_text())
        if m.get("node_type") == "array" and m["codecs"][0]["name"] in ("sharding_indexed", "zarr_octree_sharding"):
            out[str(z.parent.relative_to(path))] = m["codecs"][0]["name"]
    return out


def _root(path):
    return json.loads((path / "zarr.json").read_text())["attributes"]["zarr_vectors"]


def _positions(path):
    p = np.asarray(zv.open(str(path)).read().positions)
    return p[np.lexsort(p.T[::-1])]


def _no_scratch(path):
    return not any(SCRATCH in str(p) for p in path.rglob("*"))


class _Crash(Exception):
    """The injected failure: a preemption, an OOM kill, a failed request."""


class _Faults:
    """Raise :class:`_Crash` at the ``nth`` store call ``op`` whose key
    ``match`` accepts; ``after`` does the call first (a delete that went
    through before the process died)."""

    def __init__(self, op, match, nth=1, after=False):
        self.op, self.match, self.nth, self.after, self.seen = op, match, nth, after, 0

    async def __call__(self, op, key, do):
        hit = op == self.op and self.match(key)
        if hit:
            self.seen += 1
        if hit and self.seen == self.nth:
            if self.after:
                await do()
            raise _Crash(f"{op} {key}")
        return await do()


class _FaultyLocal(LocalStore):
    faults: _Faults | None = None

    async def set(self, key, value):
        return await _go(self.faults, "set", key, lambda: LocalStore.set(self, key, value))

    async def delete_dir(self, prefix):
        return await _go(self.faults, "delete_dir", prefix, lambda: LocalStore.delete_dir(self, prefix))


class _FaultyObject(WrapperStore):
    """Not a LocalStore: the code path of an object store."""

    faults: _Faults | None = None

    async def set(self, key, value):
        return await _go(self.faults, "set", key, lambda: self._store.set(key, value))

    async def delete_dir(self, prefix):
        return await _go(self.faults, "delete_dir", prefix, lambda: self._store.delete_dir(prefix))


async def _go(faults, op, key, do):
    if faults is None:
        return await do()
    return await faults(op, key, do)


def _faulty(path, where, faults):
    store = _FaultyLocal(str(path)) if where == "local" else _FaultyObject(LocalStore(str(path)))
    store.faults = faults
    return store


def _in_scratch_cells(key):
    return f"/{SCRATCH}/" in key and "/c/" in key


def _scratch_state(key):
    return key.endswith(f"{SCRATCH}/zarr.json")


def _a_real_array(key):
    return key.startswith("0/") and SCRATCH not in key


def _real_cells(key):
    return _a_real_array(key) and "/c/" in key


def _crash_once_moved_aside(monkeypatch, match=lambda path: True):
    """A local swap renames the source aside (into the scratch group) and
    then the copy into its place: crash between the two renames."""
    real = os.replace

    def replace(src, dst):
        moving_aside = os.path.isdir(src) and SCRATCH not in str(src) and match(str(src))
        real(src, dst)
        if moving_aside:
            raise _Crash(f"after moving {src} aside")

    monkeypatch.setattr(os, "replace", replace)


# step: (op, match, nth, after, stores it applies to)
STEPS = {
    # the copy half written: the source untouched
    "copy_cells": ("set", _in_scratch_cells, 2, False, ("local", "object")),
    # the copy written, not yet stamped complete
    "stamp_complete": ("set", _scratch_state, 2, False, ("local", "object")),
    # the source deleted, the copy not yet in its place (object stores) ...
    "source_deleted": ("delete_dir", _a_real_array, 1, True, ("object",)),
    # ... or moved aside, the copy not yet renamed in (local stores; the
    # fault is a rename, see _crash_once_moved_aside)
    "source_aside": (None, None, None, None, ("local",)),
    # the copy half copied over the source (object stores only)
    "swap_copy": ("set", _real_cells, 2, False, ("object",)),
    # the swap done, the scratch group not yet dropped (the 1st drop is the
    # check for an earlier run's leftovers, at the start of the level)
    "drop_scratch": ("delete_dir", lambda k: k.endswith(SCRATCH), 2, False, ("local", "object")),
}
CASES = [(step, where) for step, spec in STEPS.items() for where in spec[4]]


@needs_octree
@pytest.mark.parametrize(("step", "where"), CASES)
@pytest.mark.parametrize("direction", ["mark", "unmark"])
def test_an_interrupted_set_octree_loses_nothing_and_a_rerun_finishes_it(
        tmp_path, monkeypatch, step, where, direction):
    """The review's blocker: an array deleted before its cells were written
    back read 0 vertices, and a re-run skipped it and reported success."""
    from zarr_vectors.sharding import set_octree

    p = _make(tmp_path / "s.zv", octree=False if direction == "mark" else TARGET)
    want = _positions(p)
    n_arrays = len(_codecs(p))
    config, codec = (TARGET, "zarr_octree_sharding") if direction == "mark" else (False, "sharding_indexed")
    op, match, nth, after, _ = STEPS[step]

    with pytest.raises(_Crash), monkeypatch.context() as m:
        if step == "source_aside":
            _crash_once_moved_aside(m)
            set_octree(str(p), config)
        else:
            set_octree(_faulty(p, where, _Faults(op, match, nth, after)), config)

    if step not in ("source_deleted", "source_aside", "swap_copy"):
        # Between the runs every cell is still in the store: the source
        # was not touched, or the swap is done.
        assert np.array_equal(_positions(p), want)
    if direction == "unmark":
        assert _root(p)["octree_sharding"] == TARGET  # not removed before the arrays are converted

    out = set_octree(str(p), config)
    assert np.array_equal(_positions(p), want)
    assert set(_codecs(p).values()) == {codec} and len(_codecs(p)) == n_arrays
    assert _no_scratch(p)
    if direction == "mark":
        assert _root(p)["octree_sharding"] == TARGET and "octree_sharding" in _root(p)["format_capabilities"]
    else:
        assert "octree_sharding" not in _root(p) and "octree_sharding" not in _root(p)["format_capabilities"]
    # The array the first run had finished copying is swapped in, not redone.
    assert out["arrays_repacked"] == n_arrays - (step not in ("copy_cells", "stamp_complete"))
    assert set_octree(str(p), config)["arrays_repacked"] == 0


@needs_octree
@pytest.mark.parametrize("where", ["local", "object"])
@pytest.mark.parametrize("nth", [1, 3, 7, 12])
def test_a_crash_at_any_cell_write_is_recovered_by_a_rerun(tmp_path, where, nth):
    """Whatever the implementation: the process dies at the ``nth`` write of
    a shard object anywhere in the store, and a re-run gives every point back."""
    from zarr_vectors.sharding import set_octree

    p = _make(tmp_path / "s.zv")
    want = _positions(p)
    with pytest.raises(_Crash):
        set_octree(_faulty(p, where, _Faults("set", lambda k: "/c/" in k, nth)), TARGET)
    set_octree(str(p), TARGET)
    assert np.array_equal(_positions(p), want)
    assert set(_codecs(p).values()) == {"zarr_octree_sharding"}


@needs_octree
@pytest.mark.parametrize("where", ["local", "object"])
def test_an_interrupted_swap_of_a_links_array_is_finished(tmp_path, monkeypatch, where):
    """A nested array (``links/<delta>/<offsets>``), its source deleted mid-swap."""
    from zarr_vectors.sharding import set_octree

    p = _make(tmp_path / "s.zv", kind="mesh")
    want = _positions(p)
    links = sorted(a for a in _codecs(p) if "/links/" in a)
    target = links[len(links) // 2]
    with pytest.raises(_Crash), monkeypatch.context() as m:
        if where == "local":
            _crash_once_moved_aside(m, lambda path: path.endswith(target))
            set_octree(str(p), TARGET)
        else:
            set_octree(_faulty(p, where, _Faults("delete_dir", lambda k: k == target, 1, after=True)), TARGET)
    assert not (p / target / "zarr.json").exists()  # the source is gone, the copy is in scratch
    set_octree(str(p), TARGET)
    assert (p / target / "zarr.json").exists() and _codecs(p)[target] == "zarr_octree_sharding"
    assert set(_codecs(p).values()) == {"zarr_octree_sharding"} and _no_scratch(p)
    assert np.array_equal(_positions(p), want)
    ds = zv.open(str(p))
    assert len(ds.read().positions) == len(want)


@needs_octree
def test_a_rerun_with_another_configuration_redoes_the_array_it_finishes(tmp_path, monkeypatch):
    from zarr_vectors.sharding import set_octree

    p = _make(tmp_path / "s.zv")
    want = _positions(p)
    with pytest.raises(_Crash), monkeypatch.context() as m:
        _crash_once_moved_aside(m)
        set_octree(str(p), TARGET)
    set_octree(str(p), {"target": 5_000})
    configs = {json.loads((p / a / "zarr.json").read_text())["codecs"][0]["configuration"]["target"]
               for a in _codecs(p)}
    assert configs == {5_000.0} and np.array_equal(_positions(p), want) and _no_scratch(p)


# ---------------------------------------------------------------------------
# The marker follows the arrays
# ---------------------------------------------------------------------------


@needs_octree
def test_set_octree_with_arrays_leaves_the_marker_until_every_array_is_converted(tmp_path):
    from zarr_vectors.sharding import set_octree

    p = _make(tmp_path / "s.zv", octree=TARGET)
    names = sorted(_codecs(p))
    assert len(names) > 1
    set_octree(str(p), False, arrays=[names[0].split("/", 1)[1]])
    # One array is sharding_indexed, the rest still octree: the store is
    # still marked, and still needs zarr-octree.
    assert _root(p)["octree_sharding"] == TARGET and "octree_sharding" in _root(p)["format_capabilities"]
    set_octree(str(p), False, arrays=[n.split("/", 1)[1] for n in names[1:]])
    assert set(_codecs(p).values()) == {"sharding_indexed"}
    assert "octree_sharding" not in _root(p) and "octree_sharding" not in _root(p)["format_capabilities"]

    # And the other way: one array octree in an unmarked store says
    # zarr-octree is needed, before the marker is there.
    set_octree(str(p), TARGET, arrays=[names[0].split("/", 1)[1]])
    assert "octree_sharding" not in _root(p) and "octree_sharding" in _root(p)["format_capabilities"]
    set_octree(str(p), TARGET, arrays=[n.split("/", 1)[1] for n in names[1:]])
    assert _root(p)["octree_sharding"] == TARGET and "octree_sharding" in _root(p)["format_capabilities"]
    assert len(_positions(p)) == 1_500


@needs_octree
def test_unshard_store_unmarks_a_marked_store(tmp_path):
    from zarr_vectors.sharding import unshard_store

    p = _make(tmp_path / "s.zv", octree=TARGET)
    want = _positions(p)
    unshard_store(str(p))
    assert not _codecs(p)
    assert "octree_sharding" not in _root(p) and "octree_sharding" not in _root(p)["format_capabilities"]
    assert np.array_equal(_positions(p), want)


@needs_octree
def test_unshard_store_with_arrays_keeps_the_marker_while_octree_arrays_remain(tmp_path):
    from zarr_vectors.sharding import unshard_store

    p = _make(tmp_path / "s.zv", octree=TARGET)
    unshard_store(str(p), arrays=["vertices"])
    assert "zarr_octree_sharding" in _codecs(p).values()
    assert _root(p)["octree_sharding"] == TARGET and "octree_sharding" in _root(p)["format_capabilities"]


def test_unshard_store_leaves_an_unmarked_root_alone(tmp_path):
    from zarr_vectors.sharding import unshard_store

    p = _make(tmp_path / "s.zv")
    before = json.loads((p / "zarr.json").read_text())
    unshard_store(str(p))
    after = json.loads((p / "zarr.json").read_text())
    # A whole-store unshard clears the root shard_shape; the octree marker
    # and its capability are not added, and nothing else changes.
    before["attributes"]["zarr_vectors"].pop("shard_shape")
    assert after == before


def test_set_octree_false_leaves_an_unmarked_store_alone(tmp_path):
    from zarr_vectors.sharding import set_octree

    p = _make(tmp_path / "s.zv")
    before = {f: f.read_bytes() for f in p.rglob("*") if f.is_file()}
    assert set_octree(str(p), False)["arrays_repacked"] == 0
    assert {f: f.read_bytes() for f in p.rglob("*") if f.is_file()} == before


# ---------------------------------------------------------------------------
# A store sharded after the fact
# ---------------------------------------------------------------------------


@needs_octree
def test_set_octree_accepts_a_store_sharded_by_shard_store(tmp_path):
    from zarr_vectors.sharding import set_octree, shard_store

    p = _make(tmp_path / "s.zv", shard_cells=None)
    want = _positions(p)
    shard_store(str(p), shard_shape=4)
    assert _root(p).get("shard_shape") is None
    out = set_octree(str(p), TARGET)
    assert out["arrays_repacked"] == len(_codecs(p)) > 0
    assert set(_codecs(p).values()) == {"zarr_octree_sharding"}
    assert _root(p)["shard_shape"] == [4, 4, 4] and _root(p)["octree_sharding"] == TARGET
    assert np.array_equal(_positions(p), want)
    # A later write allocates octree arrays, as in a store born marked.
    n = len(_codecs(p))
    zv.open(str(p), mode="r+").build_pyramid(factors=[(2.0, 1.0)])
    assert len(_codecs(p)) > n
    assert set(_codecs(p).values()) == {"zarr_octree_sharding"}


@needs_octree
def test_set_octree_refuses_a_store_sharded_with_two_shapes(tmp_path):
    from zarr_vectors.sharding import set_octree, shard_store

    p = _make(tmp_path / "s.zv", shard_cells=None)
    shard_store(str(p), shard_shape=4, arrays=["vertices"])
    shard_store(str(p), shard_shape=2, arrays=["vertex_fragments"])
    before = (p / "zarr.json").read_bytes()
    with pytest.raises(ValueError, match="reshard"):
        set_octree(str(p), TARGET)
    assert (p / "zarr.json").read_bytes() == before


# ---------------------------------------------------------------------------
# The marker is read strictly
# ---------------------------------------------------------------------------


class _FlakyRoot(WrapperStore):
    fail_next_root = False

    async def get(self, key, prototype, byte_range=None):
        if key == "zarr.json" and _FlakyRoot.fail_next_root:
            _FlakyRoot.fail_next_root = False
            raise OSError("503 SlowDown (simulated)")
        return await self._store.get(key, prototype, byte_range)


@needs_octree
def test_a_failed_marker_read_raises_and_is_not_cached(tmp_path):
    """It used to answer "unmarked" and allocate sharding_indexed arrays in
    an octree store for the handle's lifetime."""
    from zarr_vectors import building as zb

    p = tmp_path / "s.zv"
    bounds, chunk = ([0.0, 0, 0], [400.0, 400, 400]), (100.0, 100, 100)
    zb.create_store(str(p), bounds=bounds, chunk_shape=chunk, shard_shape=2, octree_sharding=TARGET)
    level = zb.get_resolution_level(zb.open_store(_FlakyRoot(LocalStore(str(p))), mode="r+"), 0)
    with zb.open_write_session(level, bounds=bounds, chunk_shape=chunk):
        _FlakyRoot.fail_next_root = True
        with pytest.raises(OSError, match="503"):
            zb.create_vertices_array(level, dtype="float32")
        zb.create_vertices_array(level, dtype="float32")  # the same handle, retried
        zb.create_attribute_array(level, "w", dtype="float32")
    codecs = _codecs(p)
    assert {"0/vertices", "0/vertex_attributes/w"} <= set(codecs), codecs
    assert set(codecs.values()) == {"zarr_octree_sharding"}, codecs


def test_a_bare_group_is_unmarked(tmp_path):
    """No store root to ask (GroupNotFoundError) is the one case that means unmarked."""
    import zarr

    from zarr_vectors.core.group import Group

    store = LocalStore(str(tmp_path / "bare"))
    zg = zarr.open_group(store, path="lvl", mode="w")
    (tmp_path / "bare" / "zarr.json").unlink(missing_ok=True)
    g = Group.__new__(Group)
    g._zarr = zg
    assert g._store_octree() is None


# ---------------------------------------------------------------------------
# Layout(octree={}) and the configuration checked up front
# ---------------------------------------------------------------------------


@needs_octree
def test_layout_octree_as_an_empty_mapping_shards_an_auto_store(tmp_path):
    p = tmp_path / "s.zv"
    pts, ids = _points()
    zv.create(str(p), schema=_schema({}, shard_cells="auto")).add_points(pts, object_ids=ids)
    assert _root(p)["octree_sharding"] == {} and _root(p)["shard_shape"] is not None
    assert set(_codecs(p).values()) == {"zarr_octree_sharding"}
    # Round-tripped through Schema.from_store and resolved again.
    sch = zv.Schema.from_store(zv.open(str(p))._root_meta)
    assert sch.layout.octree == {}
    again = zv.Schema(bounds=sch.bounds, kind=sch.kind, layout=zv.Layout(cells=8, octree=sch.layout.octree))
    assert again.layout.resolve(again).shard_shape is not None


@needs_octree
@pytest.mark.parametrize("config", [{"bogus": 1}, {"target": -5}, {"layout": "nonsense"}])
def test_a_bad_octree_configuration_fails_before_the_store_exists(tmp_path, config):
    """It used to pass zv.create and fail in the first write, leaving a marked, empty store."""
    p = tmp_path / "s.zv"
    with pytest.raises(MetadataError, match="octree_sharding"):
        zv.create(str(p), schema=_schema(config))
    assert not p.exists() or not any(p.iterdir())


@needs_octree
@pytest.mark.parametrize("config", [{"target": object()}, "yes", ["target", 3]])
def test_create_store_refuses_a_marker_that_is_not_a_json_mapping(tmp_path, config):
    p = tmp_path / "s.zv"
    with pytest.raises(MetadataError, match="octree_sharding"):
        _create_store(p, config)
    assert not p.exists() or not any(p.iterdir())


def _create_store(p, config):
    from zarr_vectors.core.store import create_store

    return create_store(str(p), bounds=([0.0, 0, 0], [BOX, BOX, BOX]), chunk_shape=(25.0, 25, 25),
                        shard_shape=2, octree_sharding=config)


def test_create_store_octree_without_shard_shape_fails_before_the_store_exists(tmp_path):
    from zarr_vectors.core.store import create_store

    p = tmp_path / "s.zv"
    with pytest.raises(MetadataError, match="shard_shape"):
        create_store(str(p), bounds=([0.0, 0, 0], [BOX, BOX, BOX]), chunk_shape=(25.0, 25, 25),
                     octree_sharding={})
    assert not p.exists() or not any(p.iterdir())


@needs_octree
def test_set_octree_checks_the_configuration_before_touching_the_store(tmp_path):
    from zarr_vectors.sharding import set_octree

    p = _make(tmp_path / "s.zv")
    before = {f: f.read_bytes() for f in p.rglob("*") if f.is_file()}
    with pytest.raises(MetadataError):
        set_octree(str(p), {"target": -1})
    assert {f: f.read_bytes() for f in p.rglob("*") if f.is_file()} == before


_WITHOUT_OCTREE = textwrap.dedent("""
    import importlib.abc, json, os, sys

    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "zarr_octree" or name.startswith("zarr_octree."):
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    sys.meta_path.insert(0, Block())
    import zarr_vectors as zv

    root = sys.argv[1]
    marked = zv.Schema(bounds=([0, 0, 0], [100, 100, 100]), kind="point_cloud",
                       layout=zv.Layout(cells=8, shard_cells=4, octree=True))
    try:
        zv.create(root + "/b.zv", schema=marked)
        asked = "created"
    except Exception as exc:
        asked = f"{type(exc).__name__}: {exc}"
    left = os.path.exists(root + "/b.zv") and bool(os.listdir(root + "/b.zv"))
    print(json.dumps({"asked": asked, "left": left}))
""")


def test_without_zarr_octree_a_marked_create_fails_before_the_store_exists(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("ZARR_VECTORS_SHARD_")}
    r = subprocess.run([sys.executable, "-c", _WITHOUT_OCTREE, str(tmp_path)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["asked"].startswith("ValueError") and "zarr-octree" in out["asked"], out["asked"]
    assert not out["left"]
