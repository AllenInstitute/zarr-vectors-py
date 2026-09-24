"""The encoders run on the device: the numpy encoders' output, exactly.

Each device encoder is compared with its numpy original on the same
inputs -- fragments of every kind, manifests of 0, 1 and k blocks, link
records of every width and placement rule -- and must return the same
host values and raise the same errors. What makes it worth doing is what
crosses back: a run of consecutive indices comes home as one range row.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from tests.gpu._cuda import CUDA, cupy
from zarr_vectors import _xp
from zarr_vectors.encoding.fragments import (
    classify_fragments_csr,
    encode_fragments_csr,
    encode_object_manifests_csr,
)
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.spatial.boundary import partition_link_arrays

pytestmark = CUDA


def _fragments(rng, kind, num):
    """CSR fragments: ranges, explicit, empties, reversed twins, or a mix."""
    out = []
    for f in range(num):
        choice = kind if kind != "mix" else ("range", "explicit", "empty", "reversed")[f % 4]
        n = int(rng.integers(1, 30))
        start = int(rng.integers(0, 1000))
        if choice == "range":
            out.append(np.arange(start, start + n))
        elif choice == "reversed":
            out.append(np.arange(start, start + n)[::-1])
        elif choice == "empty":
            out.append(np.empty(0, dtype=np.int64))
        else:
            out.append(rng.integers(0, 1000, n))
    counts = [len(f) for f in out]
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    indices = np.concatenate(out).astype(np.int64) if out else np.empty(0, np.int64)
    return indices, offsets


def _same_sections(a, b):
    for field in ("is_range", "range_table", "explicit_offsets", "explicit_indices"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field), err_msg=field)
        assert getattr(a, field).dtype == getattr(b, field).dtype, field


@pytest.mark.parametrize("kind", ["range", "explicit", "empty", "reversed", "mix"])
@pytest.mark.parametrize("num", [0, 1, 7, 8, 9, 64, 1000])
@pytest.mark.parametrize("force_explicit", [False, True])
def test_fragments_classify_and_pack_like_numpy(kind, num, force_explicit):
    indices, offsets = _fragments(np.random.default_rng(num), kind, num)
    host = classify_fragments_csr(indices, offsets, force_explicit=force_explicit)
    dev = classify_fragments_csr(
        cupy.asarray(indices), cupy.asarray(offsets), force_explicit=force_explicit,
    )
    _same_sections(host, dev)
    assert encode_fragments_csr(
        cupy.asarray(indices), cupy.asarray(offsets), force_explicit=force_explicit,
    ) == encode_fragments_csr(indices, offsets, force_explicit=force_explicit)


@pytest.mark.parametrize("indices,offsets,match", [
    ([0, 1], [1, 2], "start at 0"),
    ([0, 1], [0, 1], "end at len"),
    ([0, 1, 2], [0, 2, 1, 3], "non-decreasing"),
    ([0, -1], [0, 2], "non-negative"),
    ([0.5], [0, 1], "integers"),
])
def test_fragment_errors_match(indices, offsets, match):
    for wrap in (np.asarray, cupy.asarray):
        with pytest.raises(ArrayError, match=match):
            classify_fragments_csr(wrap(indices), wrap(offsets))


def test_range_heavy_fragments_cross_back_small():
    indices, offsets = _fragments(np.random.default_rng(0), "range", 5000)
    with _xp.count_transfers() as stats:
        classify_fragments_csr(cupy.asarray(indices), cupy.asarray(offsets))
    assert stats.d2h_bytes < indices.nbytes / 4


@pytest.mark.parametrize("n", [0, 1, 5, 300])
@pytest.mark.parametrize("sid", [3, 4])
@pytest.mark.parametrize("with_offsets", [True, False])
def test_manifests_encode_like_numpy(n, sid, with_offsets):
    rng = np.random.default_rng(n + sid)
    counts = rng.integers(0, 5, n) if with_offsets else np.ones(n, np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    total = int(offsets[-1])
    coords = rng.integers(-3, 9, (total, sid)).astype(np.int64)
    frags = rng.integers(0, 2**40, total).astype(np.int64)
    off = offsets if with_offsets else None
    host = encode_object_manifests_csr(coords, frags, off, sid_ndim=sid)
    dev = encode_object_manifests_csr(
        cupy.asarray(coords), cupy.asarray(frags),
        None if off is None else cupy.asarray(off), sid_ndim=sid,
    )
    assert dev.dtype == object and list(dev) == list(host)


def test_manifest_errors_match():
    cases = [
        (([[0, 0, 0]], [-1], None), ">= 0"),
        (([[0, 0]], [1], None), "rank"),
        (([[0, 0, 0]], [1], [1, 1]), "start at 0"),
        (([[0, 0, 0], [0, 0, 0]], [1, 2], [0, 2, 1, 2]), "non-decreasing"),
    ]
    for (cc, fi, off), match in cases:
        for wrap in (np.asarray, cupy.asarray):
            with pytest.raises(ArrayError, match=match):
                encode_object_manifests_csr(
                    wrap(cc), wrap(fi), None if off is None else wrap(off), sid_ndim=3,
                )


def _records(rng, n, L, sid):
    centres = rng.integers(-3, 4, (4, sid))
    base = centres[rng.integers(0, 4, n)][:, None, :]
    chunks = base + rng.integers(-1, 2, (n, L, sid)) * (rng.random((n, 1, 1)) > 0.3)
    vi = rng.integers(0, 40, (n, L)).astype(np.int64)
    if n and L > 1:
        chunks[0, 1] = chunks[0, 0]
        vi[0, 1] = vi[0, 0]
    return chunks.astype(np.int64), vi


@pytest.mark.parametrize("L,sid,directed,cross_level", [
    (L, sid, directed, cross)
    for L, sid, directed, cross in itertools.product(
        [1, 2, 3, 4], [3, 4], [False, True], [False, True],
    )
    if not (cross and L == 1)
])
def test_link_partition_like_numpy(L, sid, directed, cross_level):
    rng = np.random.default_rng(L * 10 + sid)
    chunks, vi = _records(rng, 400, L, sid)
    scale_src = [2] * sid if cross_level else [1] * sid
    kw = dict(
        link_width=L, sid_ndim=sid, scale_src=scale_src, scale_trg=[1] * sid,
        directed=directed, cross_level=cross_level,
    )
    host = partition_link_arrays(chunks, vi, **kw)
    dev = partition_link_arrays(cupy.asarray(chunks), cupy.asarray(vi), **kw)
    assert list(host) == list(dev)
    for key in host:
        np.testing.assert_array_equal(dev[key][0], host[key][0], err_msg=str(key))
        np.testing.assert_array_equal(dev[key][1], host[key][1], err_msg=str(key))
        assert isinstance(dev[key][0], np.ndarray)


def test_encode_can_be_forced_onto_the_host(monkeypatch):
    indices, offsets = _fragments(np.random.default_rng(1), "range", 50)
    monkeypatch.setenv("ZARR_VECTORS_GPU_ENCODE", "0")
    with _xp.count_transfers() as stats:
        classify_fragments_csr(cupy.asarray(indices), cupy.asarray(offsets))
    assert stats.d2h_calls == 2  # the two arguments, copied off as they are
