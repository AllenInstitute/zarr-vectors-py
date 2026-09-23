"""A per-chunk pipeline written through the array-form calls leaves the store
the per-object calls leave, and that store validates.

Shaped like BRIDGE's graph stage on top of a polyline store: each "worker
flush" appends fragments to a cell, manifests for its new objects at the
committed count, a batch of object attribute columns, and seam links with
their attributes; a coordinator then finalizes the links and commits the
object count. One copy is built with the per-object calls, one with the
array calls, and a third from device arrays when a GPU is available.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._store_compare import assert_stores_identical
from zarr_vectors.building import (
    OBJECT_INDEX,
    encode_object_manifest_blocks,
    finalize_links,
    get_resolution_level,
    open_store,
    write_chunk_fragments,
    write_link_attribute_cells,
    write_link_cells,
    write_object_attribute_columns,
    write_object_attributes,
    write_object_manifests,
    write_polylines,
)
from zarr_vectors.validate import validate


def _base(tmp_path, name):
    rng = np.random.default_rng(0)
    polys = [rng.uniform(5, 95, (6, 3)).astype("float32") for _ in range(12)]
    (tmp_path / name).mkdir()
    path = tmp_path / name / "s.zarrvectors"
    write_polylines(str(path), polys, chunk_shape=(50.0,) * 3)
    return path, get_resolution_level(open_store(str(path), mode="r+"), 0)


def _flushes(seed=7, n=3):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        k = int(rng.integers(3, 9))
        counts = rng.integers(1, 4, k)
        frag_off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        frag_idx = rng.integers(0, 6, int(frag_off[-1])).astype(np.int64)
        seams = int(rng.integers(2, 10))
        base = rng.integers(0, 2, (seams, 1, 3))
        link_cc = np.clip(base + rng.integers(-1, 2, (seams, 2, 3)), 0, 1).astype(np.int64)
        yield {
            "cell": tuple(int(c) for c in rng.integers(0, 2, 3)),
            "frag_idx": frag_idx, "frag_off": frag_off,
            "cols": {
                "length": rng.uniform(0, 50, k).astype(np.float32),
                "n_nodes": rng.integers(1, 9, k).astype(np.int32),
            },
            "link_cc": link_cc,
            "link_vi": rng.integers(0, 6, (seams, 2)).astype(np.int64),
            "link_w": rng.uniform(0, 1, seams).astype(np.float32),
        }


def _committed(lg):
    return int(lg.read_array_meta(OBJECT_INDEX)["num_objects"])


def _commit(lg, n):
    meta = lg.read_array_meta(OBJECT_INDEX)
    lg.write_array_meta(OBJECT_INDEX, {**meta, "num_objects": n, "num_present": n})


def _per_object(lg):
    for f in _flushes():
        n0 = _committed(lg)
        frags = [
            f["frag_idx"][a:b] for a, b in zip(f["frag_off"][:-1], f["frag_off"][1:])
        ]
        got = write_chunk_fragments(lg, f["cell"], frags, mode="append")
        k = len(frags)
        blobs = [
            encode_object_manifest_blocks([(f["cell"], got[i])], sid_ndim=3)
            for i in range(k)
        ]
        write_object_manifests(lg, blobs, mode="append", at=n0)
        for name, col in f["cols"].items():
            write_object_attributes(lg, name, col, mode="append", at=n0)
        records = [
            [(tuple(f["link_cc"][r, j].tolist()), int(f["link_vi"][r, j])) for j in range(2)]
            for r in range(len(f["link_vi"]))
        ]
        part = write_link_cells(lg, records, 3)
        write_link_attribute_cells(lg, "w", f["link_w"], partition=part)
        _commit(lg, n0 + k)


def _arrays(lg, to=lambda a: a):
    for f in _flushes():
        n0 = _committed(lg)
        start, k = write_chunk_fragments(
            lg, f["cell"], csr=(to(f["frag_idx"]), to(f["frag_off"])), mode="append",
        )
        write_object_manifests(
            lg, chunk_coords=to(np.tile(f["cell"], (k, 1))),
            fragment_idx=to(np.arange(start, start + k)), mode="append", at=n0,
        )
        write_object_attribute_columns(lg, {n: to(c) for n, c in f["cols"].items()}, at=n0)
        write_link_cells(
            lg, chunks=to(f["link_cc"]), vids=to(f["link_vi"]),
            attributes={"w": to(f["link_w"])},
        )
        _commit(lg, n0 + k)


def _finish(lg, path):
    finalize_links(lg, delta=0)
    result = validate(str(path), level=5)
    assert result.ok, result.summary()


def test_array_calls_build_the_per_object_store(tmp_path):
    path_a, lg_a = _base(tmp_path, "per_object")
    path_b, lg_b = _base(tmp_path, "arrays")
    _per_object(lg_a)
    _arrays(lg_b)
    _finish(lg_a, path_a)
    _finish(lg_b, path_b)
    assert_stores_identical(path_a, path_b)


def test_device_arrays_build_it_too(tmp_path):
    cupy = pytest.importorskip("cupy")
    from zarr_vectors.gpu import device_count

    if device_count() == 0:
        pytest.skip("no CUDA device")
    path_a, lg_a = _base(tmp_path, "host")
    path_b, lg_b = _base(tmp_path, "device")
    _arrays(lg_a)
    _arrays(lg_b, to=cupy.asarray)
    _finish(lg_a, path_a)
    _finish(lg_b, path_b)
    assert_stores_identical(path_a, path_b)
