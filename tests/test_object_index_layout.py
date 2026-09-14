"""Tests for the ``object_index`` vlen-bytes ragged-array layout.

Covers:

* Round-trip correctness — write a polyline store and verify single-OID
  manifest reads match ``read_all_object_manifests``.
* Layout discriminator — fresh writes emit ``layout == "vlen_manifests_v1"``
  and a single ``manifests`` zarr array (no legacy ``data``/``offsets``).
* Multi-chunk random access — when ``num_objects`` exceeds the per-chunk
  bucket, reads at OIDs straddling the chunk boundary still return the
  correct manifest.
* Per-object read scalability — reading one object's manifest does not
  scale with the total ``num_objects`` (the bug the layout was introduced
  to fix).
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.constants import OBJECT_INDEX
from zarr_vectors.core.arrays import (
    OBJECT_INDEX_LAYOUT_V2,
    OBJECT_INDEX_MANIFEST_BUCKET,
    read_all_object_manifests,
    read_object_manifest,
)
from zarr_vectors.core.store import open_store
from zarr_vectors.types.polylines import write_polylines


CHUNK = (200.0, 200.0, 200.0)
BIN = (50.0, 50.0, 50.0)


def _new_store(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"oi_{prefix}_")) / "store.zarrvectors"


def _make_polylines(n: int, rng: np.random.Generator) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for _ in range(n):
        steps = rng.normal(0, 5, (int(rng.integers(2, 6)), 3))
        start = rng.uniform(0, 1000, 3)
        out.append((start + steps.cumsum(axis=0)).astype(np.float32))
    return out


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_fresh_store_uses_vlen_layout(rng: np.random.Generator) -> None:
    store = _new_store("layout_check")
    write_polylines(
        store, _make_polylines(200, rng),
        chunk_shape=CHUNK, bin_shape=BIN,
    )
    level = open_store(str(store), mode="r")["0"]
    meta = level.read_array_meta(OBJECT_INDEX)
    assert meta.get("layout") == OBJECT_INDEX_LAYOUT_V2

    oi_grp = level.zarr_group[OBJECT_INDEX]
    children = set(oi_grp.array_keys())
    assert "manifests" in children
    # The ids are stored, not implied by row position.
    assert "object_ids" in children
    assert "data" not in children
    assert "offsets" not in children


def test_single_read_matches_read_all(rng: np.random.Generator) -> None:
    n = 500
    store = _new_store("parity")
    write_polylines(
        store, _make_polylines(n, rng), chunk_shape=CHUNK, bin_shape=BIN,
    )
    level = open_store(str(store), mode="r")["0"]
    all_manifests = read_all_object_manifests(level)
    assert len(all_manifests) == n
    for oid in (0, 1, n // 2, n - 2, n - 1):
        assert read_object_manifest(level, oid) == all_manifests[oid]


def test_multichunk_random_access_works(rng: np.random.Generator) -> None:
    # Force more than one zarr chunk in ``manifests`` by exceeding the
    # bucket size.  Use the smallest object that still produces a non-
    # empty manifest blob — keeps the test fast.
    n = OBJECT_INDEX_MANIFEST_BUCKET + 100
    store = _new_store("multichunk")
    write_polylines(
        store, _make_polylines(n, rng), chunk_shape=CHUNK, bin_shape=BIN,
    )
    level = open_store(str(store), mode="r")["0"]
    manifests = level.zarr_group[OBJECT_INDEX]["manifests"]
    assert manifests.shape == (n,)
    # At least two chunks — confirms ``manifests`` is actually multichunk.
    assert manifests.chunks[0] < n

    # OIDs spanning the chunk boundary must all decode correctly.
    boundary = OBJECT_INDEX_MANIFEST_BUCKET
    all_manifests = read_all_object_manifests(level)
    for oid in (0, boundary - 1, boundary, boundary + 1, n - 1):
        got = read_object_manifest(level, oid)
        assert got == all_manifests[oid], f"mismatch at oid={oid}"


def test_single_read_scales_constant_in_num_objects(
    rng: np.random.Generator,
) -> None:
    """Time a single-OID manifest read at two very different N.

    The legacy ``data``/``offsets`` layout had read amplification
    proportional to ``num_objects``: at 10K objects each
    ``read_object_manifest`` call loaded ~80 KB of offsets plus the
    entire concatenated manifest blob; at 100 objects it loaded ~0.8 KB.
    The new layout fetches one zarr chunk regardless of N — per-call
    latency should be similar at both scales.

    Asserts the 10K-N latency is no more than 10× the 100-N latency.
    A regression to the legacy layout would push that ratio toward
    ``num_objects / chunk_size`` (which is `>=` the ~100× of the
    underlying N ratio for small N).
    """
    def _time_single_read(n_objects: int) -> float:
        store = _new_store(f"scale_{n_objects}")
        write_polylines(
            store, _make_polylines(n_objects, rng),
            chunk_shape=CHUNK, bin_shape=BIN,
        )
        level = open_store(str(store), mode="r")["0"]
        # Warm the open + metadata read so we time the manifest fetch only.
        read_object_manifest(level, 0)

        samples: list[float] = []
        target_oids = rng.integers(0, n_objects, size=20)
        for oid in target_oids:
            t0 = time.perf_counter()
            read_object_manifest(level, int(oid))
            samples.append(time.perf_counter() - t0)
        return float(np.median(samples))

    small = _time_single_read(100)
    large = _time_single_read(10_000)
    # 10× slack: cold-cache effects and chunk-fill differences shouldn't
    # exceed this if the access pattern is genuinely chunk-local.  A
    # regression to full-array reads pushes the ratio to ~100×.
    assert large < small * 10 + 0.05, (
        f"single-read latency scales with num_objects: "
        f"small={small * 1e3:.2f}ms vs large={large * 1e3:.2f}ms"
    )
