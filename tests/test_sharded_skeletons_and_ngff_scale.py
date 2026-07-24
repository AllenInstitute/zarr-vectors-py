"""Two features ported from PR #30 (raw-cell-sharded-arrays).

Both were missing from this branch: skeleton stores could not be sharded
at all, and the NGFF scale transform silently ignored ``bin_shape``.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import zarr

from zarr_vectors.core.metadata import LevelMetadata
from zarr_vectors.core.multiscale import read_multiscale_metadata
from zarr_vectors.core.store import create_resolution_level, open_store
from zarr_vectors.sharding.io import _is_native_sharded
from zarr_vectors.types.points import write_points
from zarr_vectors.types.skeletons import (
    finalize_skeleton_store,
    init_skeleton_store,
    write_skeleton_chunk,
)


def _skeleton_store(shard_shape) -> str:
    path = os.path.join(tempfile.mkdtemp(), "sk.zv")
    root, lg = init_skeleton_store(
        path,
        chunk_shape=(50.0, 50.0, 50.0),
        bounds=([0.0, 0.0, 0.0], [200.0, 200.0, 200.0]),
        ndim=3,
        attribute_dtypes={"radius": "float32"},
        shard_shape=shard_shape,
    )
    pieces = [{
        "positions": np.array(
            [[10.0, 10, 10], [20.0, 10, 10], [30.0, 10, 10]], dtype=np.float32,
        ),
        "edges": np.array([[1, 0], [2, 1]], dtype=np.int64),
        "segment_id": 1,
        "attributes": {"radius": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
    }]
    write_skeleton_chunk(lg, (0, 0, 0), pieces)
    finalize_skeleton_store(root)
    return path


class TestShardedSkeletons:
    """``init_skeleton_store`` must honour ``shard_shape``.

    Skeletons are written by streaming, so the arrays are allocated in
    init rather than by a whole-store writer — and without a write
    session there, shard_shape/compressor never reached array creation.
    """

    def test_shard_shape_shards_the_per_chunk_arrays(self) -> None:
        z = zarr.open_group(_skeleton_store((2, 2, 2)), mode="r")
        for name in ("vertices", "vertex_fragments", "vertex_attributes/radius"):
            assert _is_native_sharded(z[f"0/{name}"]), name

    def test_default_is_unsharded(self) -> None:
        z = zarr.open_group(_skeleton_store(None), mode="r")
        for name in ("vertices", "vertex_fragments", "vertex_attributes/radius"):
            assert not _is_native_sharded(z[f"0/{name}"]), name

    def test_sharded_store_round_trips(self) -> None:
        # Sharding is a storage-layer detail; the data must be unchanged.
        from zarr_vectors.core.arrays import read_chunk_vertices
        from zarr_vectors.core.store import get_resolution_level

        lg = get_resolution_level(open_store(_skeleton_store((2, 2, 2))), 0)
        groups = read_chunk_vertices(lg, (0, 0, 0))
        pos = np.concatenate([np.asarray(g) for g in groups], axis=0)
        assert pos.shape == (3, 3)
        np.testing.assert_allclose(pos[:, 0], [10.0, 20.0, 30.0])


class TestNGFFScaleFromBinShape:
    """The NGFF scale must be derived from ``bin_shape``, as a float.

    ``create_resolution_level`` entered its transform branch whenever
    ``bin_shape`` was set, but only computed a scale from ``bin_ratio`` —
    falling through to ``scale = [1.0] * ndim`` otherwise.  A caller that
    sets only ``bin_shape`` (letting the ratio be implied) got a wrong,
    non-cumulative 1.0 baked into the transform, silently.

    The ratio is a plain float: NGFF scale has no integer requirement,
    unlike the separately-typed ``bin_ratio: tuple[int, ...]`` field, so a
    fractional coarsen factor must work.
    """

    def _store_with_level1(self, bin_shape) -> str:
        path = os.path.join(tempfile.mkdtemp(), "ngff.zv")
        write_points(
            path,
            np.random.default_rng(0).uniform(0, 400, (200, 3)).astype("f4"),
            chunk_shape=(100.0, 100.0, 100.0),
            bin_shape=(50.0, 50.0, 50.0),
            bounds=([0.0, 0.0, 0.0], [400.0, 400.0, 400.0]),
        )
        root = open_store(path, mode="r+")
        create_resolution_level(root, 1, LevelMetadata(
            level=1, vertex_count=10, arrays_present=[], parent_level=0,
            bin_shape=bin_shape,   # bin_ratio deliberately unset
        ))
        return path

    def _scale_for(self, path: str, level: str) -> list[float]:
        ms = read_multiscale_metadata(open_store(path))
        for d in ms[0]["datasets"]:
            if d["path"] == level:
                for t in d["coordinateTransformations"]:
                    if t["type"] == "scale":
                        return t["scale"]
        raise AssertionError(f"no scale for level {level}")

    def test_integer_ratio_from_bin_shape(self) -> None:
        path = self._store_with_level1((100.0, 100.0, 100.0))  # 2x
        np.testing.assert_allclose(self._scale_for(path, "1"), [2.0, 2.0, 2.0])

    def test_fractional_ratio_from_bin_shape(self) -> None:
        # 75/50 = 1.5 — an integer-only ratio helper cannot express this,
        # and the old code silently produced 1.0.
        path = self._store_with_level1((75.0, 75.0, 75.0))
        np.testing.assert_allclose(self._scale_for(path, "1"), [1.5, 1.5, 1.5])

    def test_level_zero_is_identity(self) -> None:
        path = self._store_with_level1((75.0, 75.0, 75.0))
        np.testing.assert_allclose(self._scale_for(path, "0"), [1.0, 1.0, 1.0])
