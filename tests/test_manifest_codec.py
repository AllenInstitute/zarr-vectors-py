"""New object indexes are created at zstd level 9; an index that exists keeps its codecs.

The dense index's integer arrays -- ``manifest_spans``, ``manifest_blocks``
and the id table -- are created by many paths (a full write, the first
append, a reservation, the streaming writers, the appender, a vacuum), and
each used to take whatever codec the write session carried: zarr's level 0
outside one, the session's chunk-cell level inside. Every path now asks
:func:`zarr_vectors.core.dense_manifests.manifest_codecs`, which keeps a
session's choice of codec but raises a zstd level below 9.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import zarr_vectors.building as zb
from zarr_vectors.constants import OBJECT_INDEX
from zarr_vectors.core import dense_manifests as dense
from zarr_vectors.core.arrays import write_object_index
from zarr_vectors.core.streaming import ObjectIndexAppender

pytestmark = pytest.mark.vlen_only  # these pick their layouts themselves

INDEX_ARRAYS = ("manifest_spans", "manifest_blocks", "object_ids")


def _level(tmp_path, layout="dense"):
    root = zb.create_store(
        tmp_path / "s.zarrvectors", bounds=([0.0] * 3, [100.0] * 3),
        chunk_shape=(50.0,) * 3, geometry_types=["polyline"], manifest_layout=layout,
    )
    return zb.get_resolution_level(root, 0)


def _codecs(lg, name):
    """The (inner, when sharded) codec pipeline in ``name``'s zarr.json."""
    meta = json.loads((lg.path / OBJECT_INDEX / name / "zarr.json").read_text())
    codecs = meta["codecs"]
    if codecs and codecs[0]["name"] == "sharding_indexed":
        codecs = codecs[0]["configuration"]["codecs"]
    return codecs


def _zstd_level(lg, name):
    zstd = [c for c in _codecs(lg, name) if c["name"] == "zstd"]
    assert zstd, f"{name} is not zstd-compressed: {_codecs(lg, name)}"
    return int(zstd[0]["configuration"]["level"])


def _arrays(n, k=2):
    off = np.arange(0, (n + 1) * k, k, dtype=np.int64)
    cc = np.zeros((n * k, 3), np.int64)
    fi = np.arange(n * k, dtype=np.int64)
    return off, cc, fi


def _write_index(lg, n=50):
    write_object_index(lg, {o: [((0, 0, 0), o), ((1, 0, 0), o)] for o in range(n)}, 3)


def _append(lg, n=50, at=None):
    off, cc, fi = _arrays(n)
    zb.write_object_manifests(
        lg, chunk_coords=cc, fragment_idx=fi, manifest_offsets=off, mode="append", at=at,
    )


def _reserve(lg, n=50):
    zb.reserve_object_rows(lg, n, sid_ndim=3, n_blocks=2 * n)


def _stream(lg, n=50, at=0):
    off, cc, fi = _arrays(n)
    with zb.object_manifest_writer(lg, at=at) as w:
        w.write(chunk_coords=cc, fragment_idx=fi, manifest_offsets=off)


def _appender(lg, n=50):
    with ObjectIndexAppender(lg, 0, 3, []) as app:
        app.append([[((0, 0, 0), o)] for o in range(n)], [1] * n)


WRITERS = {
    "write_object_index": _write_index,
    "append": _append,
    "reserve": _reserve,
    "object_manifest_writer": _stream,
    "appender": _appender,
}


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_every_writer_creates_the_index_at_level_9(tmp_path, writer):
    lg = _level(tmp_path)
    WRITERS[writer](lg)
    for name in INDEX_ARRAYS:
        assert _zstd_level(lg, name) == dense.MANIFEST_ZSTD_LEVEL == 9, name


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_a_zstd_session_is_raised_to_level_9(tmp_path, writer):
    """BRIDGE writes inside a session compressing its cells at zstd 5."""
    lg = _level(tmp_path)
    zstd5 = [{"name": "zstd", "configuration": {"level": 5, "checksum": False}}]
    with lg.chunk_array_codecs(zstd5):
        WRITERS[writer](lg)
    for name in INDEX_ARRAYS:
        assert _zstd_level(lg, name) == 9, name


def test_inside_batched_writes_too(tmp_path):
    lg = _level(tmp_path)
    with lg.batched_writes(compressor="zstd"):  # zarr's level 0
        _append(lg)
    for name in INDEX_ARRAYS:
        assert _zstd_level(lg, name) == 9, name


def test_a_session_level_above_9_is_kept(tmp_path):
    lg = _level(tmp_path)
    zstd15 = [{"name": "zstd", "configuration": {"level": 15, "checksum": True}}]
    with lg.chunk_array_codecs(zstd15):
        _append(lg)
    zstd = [c for c in _codecs(lg, "manifest_blocks") if c["name"] == "zstd"][0]
    assert zstd["configuration"] == {"level": 15, "checksum": True}


def test_another_codec_is_the_sessions_choice(tmp_path):
    lg = _level(tmp_path)
    with lg.chunk_array_codecs("blosc"):
        _append(lg)
    for name in INDEX_ARRAYS:
        assert [c["name"] for c in _codecs(lg, name)] == ["bytes", "blosc"], name


@pytest.mark.parametrize("compressor", [None, "none"])
def test_a_session_that_does_not_compress_gets_the_default(tmp_path, compressor):
    lg = _level(tmp_path)
    with lg.chunk_array_codecs(compressor):
        _append(lg)
    for name in INDEX_ARRAYS:
        assert _zstd_level(lg, name) == 9, name


def test_a_vlen_indexs_id_table_too(tmp_path):
    """The id table is the same array in the V2 vlen layout."""
    lg = _level(tmp_path, layout="vlen")
    _write_index(lg)
    assert _zstd_level(lg, "object_ids") == 9


def _old_index(lg, monkeypatch, writer=_append):
    """An index written before this change: zarr's level 0 on every array."""
    monkeypatch.setattr(dense, "MANIFEST_ZSTD_LEVEL", 0)
    writer(lg)
    monkeypatch.undo()
    for name in INDEX_ARRAYS:
        assert _zstd_level(lg, name) == 0


def test_an_existing_index_keeps_its_codecs(tmp_path, monkeypatch):
    lg = _level(tmp_path)
    _old_index(lg, monkeypatch)
    _append(lg)                       # appends
    _append(lg, at=30)                # a torn flush's truncation, the id table rewritten
    _stream(lg, at=80)
    zb.reserve_object_rows(lg, 200, n_blocks=400)
    zb.patch_object_manifests(lg, {3: [((0, 0, 0), 7)], 500: [((1, 1, 1), 2)]}, 3)
    for name in INDEX_ARRAYS:
        assert _zstd_level(lg, name) == 0, name
    got = zb.read_all_object_manifests(lg)
    assert got[3] == [((0, 0, 0), 7)] and got[-1] == [((1, 1, 1), 2)]


def test_a_full_rewrite_takes_the_default(tmp_path, monkeypatch):
    """``write_object_index`` (and vacuum through it) recreates every array."""
    lg = _level(tmp_path)
    _old_index(lg, monkeypatch)
    _write_index(lg)
    for name in INDEX_ARRAYS:
        assert _zstd_level(lg, name) == 9, name


def test_manifest_codecs_shapes():
    class _Lg:
        _active_codecs = None

    lg = _Lg()
    assert dense.manifest_codecs(lg) == [
        {"name": "bytes"}, {"name": "zstd", "configuration": {"level": 9, "checksum": False}},
    ]
    lg._active_codecs = [{"name": "bytes"}, {"name": "zstd", "configuration": {"level": 3}}]
    assert dense.manifest_codecs(lg)[1] == {
        "name": "zstd", "configuration": {"level": 9, "checksum": False},
    }
    # The session's own list is not modified.
    assert lg._active_codecs[1]["configuration"] == {"level": 3}


def test_the_feature_is_advertised():
    import zarr_vectors as zv

    assert "manifest-codec" in zv.FEATURES
