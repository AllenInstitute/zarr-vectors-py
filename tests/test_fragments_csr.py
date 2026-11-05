"""The CSR fragment encoder writes the bytes the per-fragment one does.

``encode_fragments`` (a list, one Python object per fragment) is the
definition; ``encode_fragments_csr`` is the array form BRIDGE and the GPU
path hand arrays to. Every kind of fragment is covered: ranges at any
start, singletons, empties, BRIDGE's forward path with its reversed twin,
non-contiguous lists, all-explicit and all-range cells, and none at all.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from zarr_vectors.encoding.fragments import (
    classify_fragments_csr,
    concat_fragment_sections,
    decode_fragments,
    encode_fragments,
    encode_fragments_csr,
    fragment_sections_from_index,
    pack_fragment_sections,
    range_fragment_sections,
)
from zarr_vectors.exceptions import ArrayError


def _kind(rng, kind):
    n = int(rng.integers(1, 12))
    if kind == "range":
        start = int(rng.integers(0, 50))
        return np.arange(start, start + n)
    if kind == "single":
        return np.array([int(rng.integers(0, 50))])
    if kind == "empty":
        return np.array([], dtype=np.int64)
    if kind == "path":
        return rng.permutation(60)[:n] if n > 1 else np.array([3, 1])
    if kind == "reversed_range":
        start = int(rng.integers(0, 50))
        return np.arange(start + n, start, -1)
    if kind == "gappy":
        return np.cumsum(rng.integers(1, 4, n)) + int(rng.integers(0, 20))
    raise AssertionError(kind)


_KINDS = ("range", "single", "empty", "path", "reversed_range", "gappy")


def _random_fragments(rng, num, kinds=_KINDS):
    frags = []
    for _ in range(num):
        f = _kind(rng, kinds[int(rng.integers(len(kinds)))])
        frags.append(f.astype(np.int64))
        if kinds is _KINDS and rng.random() < 0.2:
            frags.append(f[::-1].copy())  # BRIDGE's reversed twin
    return frags


def _csr(frags):
    offsets = np.concatenate([[0], np.cumsum([len(f) for f in frags])]).astype(np.int64)
    indices = np.concatenate(frags) if frags else np.empty(0, dtype=np.int64)
    return indices.astype(np.int64), offsets


@pytest.mark.parametrize("num", [0, 1, 7, 8, 9, 64, 1000])
@pytest.mark.parametrize("force_explicit", [False, True])
def test_csr_bytes_match_the_list_encoder(num, force_explicit):
    rng = np.random.default_rng(num + 17 * force_explicit)
    frags = _random_fragments(rng, num)
    want = encode_fragments(frags, force_explicit=force_explicit)
    got = encode_fragments_csr(*_csr(frags), force_explicit=force_explicit)
    assert got == want


@pytest.mark.parametrize("kinds", [("range",), ("path", "gappy"), ("empty",), ("single",)])
def test_uniform_cells_match(kinds):
    frags = _random_fragments(np.random.default_rng(5), 37, kinds)
    assert encode_fragments_csr(*_csr(frags)) == encode_fragments(frags)


def test_range_sections_match_the_tuple_fast_path():
    rng = np.random.default_rng(9)
    counts = rng.integers(0, 9, 70)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    want = encode_fragments([(int(s), int(c)) for s, c in zip(starts, counts)])
    assert pack_fragment_sections(range_fragment_sections(starts, counts)) == want


def test_sections_round_trip_through_a_decoded_index():
    frags = _random_fragments(np.random.default_rng(3), 200)
    blob = encode_fragments(frags)
    sections = fragment_sections_from_index(decode_fragments(blob))
    assert pack_fragment_sections(sections) == blob


@pytest.mark.parametrize("seed", range(5))
def test_concatenated_sections_equal_encoding_the_whole(seed):
    rng = np.random.default_rng(seed)
    a = _random_fragments(rng, int(rng.integers(0, 40)))
    b = _random_fragments(rng, int(rng.integers(0, 40)))
    joined = concat_fragment_sections(
        classify_fragments_csr(*_csr(a)), classify_fragments_csr(*_csr(b)),
    )
    assert pack_fragment_sections(joined) == encode_fragments(a + b)


def test_bad_input_is_refused():
    with pytest.raises(ArrayError, match="non-negative"):
        encode_fragments_csr([2, -1], [0, 2])
    with pytest.raises(ArrayError, match="start at 0"):
        encode_fragments_csr([1, 2], [0, 1])
    with pytest.raises(ArrayError, match="non-decreasing"):
        encode_fragments_csr([1, 2, 3], [0, 2, 1, 3])
    with pytest.raises(ArrayError, match="integers"):
        encode_fragments_csr([1.5], [0, 1])


def test_a_million_fragments_encode_without_a_python_loop():
    rng = np.random.default_rng(0)
    num = 10**6
    counts = rng.integers(0, 20, num)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    indices = rng.integers(0, 10**6, int(offsets[-1]))
    t0 = time.perf_counter()
    encode_fragments_csr(indices, offsets)
    assert time.perf_counter() - t0 < 1.5


# --- the chunk writer --------------------------------------------------


def _level(tmp_path, name):
    from zarr_vectors.core.store import get_resolution_level, open_store
    from zarr_vectors.types.points import write_points

    # Same basename in separate directories: the store's name is written
    # into its root metadata, and two stores being compared must share it.
    (tmp_path / name).mkdir()
    path = str(tmp_path / name / "s.zarrvectors")
    rng = np.random.default_rng(1)
    write_points(
        path, rng.uniform(0, 100, (300, 3)).astype("float32"),
        chunk_shape=(50.0, 50.0, 50.0), bounds=[[0, 0, 0], [100, 100, 100]],
    )
    return path, get_resolution_level(open_store(path, mode="r+"), 0)


@pytest.mark.parametrize("seed", range(4))
def test_csr_appends_leave_the_store_the_list_appends_do(tmp_path, seed):
    from tests._reference import write_chunk_fragments_append_by_list
    from tests._store_compare import assert_stores_identical
    from zarr_vectors.core.arrays import write_chunk_fragments

    path_a, lg_a = _level(tmp_path, "list.zarrvectors")
    path_b, lg_b = _level(tmp_path, "csr.zarrvectors")
    rng = np.random.default_rng(seed)
    for step in range(6):
        cc = (int(rng.integers(0, 2)), 0, int(rng.integers(0, 2)))
        # No force_explicit history, so the two definitions must agree.
        batch = _random_fragments(rng, int(rng.integers(0, 15)))
        want = write_chunk_fragments_append_by_list(lg_a, cc, batch)
        got = write_chunk_fragments(lg_b, cc, csr=_csr(batch), mode="append")
        assert got == ((want[0], len(want)) if want else got)
    assert_stores_identical(path_a, path_b)


def test_the_list_form_is_unchanged(tmp_path):
    from zarr_vectors.core.arrays import read_vertex_fragment_index, write_chunk_fragments

    _, lg = _level(tmp_path, "l.zarrvectors")
    frags = [(0, 3), np.array([4, 2]), np.array([], dtype=np.int64)]
    assert write_chunk_fragments(lg, (0, 0, 0), frags) == [0, 1, 2]
    assert write_chunk_fragments(lg, (0, 0, 0), frags, mode="append") == [3, 4, 5]
    assert write_chunk_fragments(lg, (0, 0, 0), [], mode="append") == []
    assert len(read_vertex_fragment_index(lg, (0, 0, 0))) == 6


def test_an_existing_explicit_arange_stays_explicit_on_append(tmp_path):
    """The one intended difference from the list append.

    A fragment written with ``force_explicit`` whose indices happen to be
    consecutive was silently promoted to a range by the next list append,
    which decoded and re-classified every existing fragment. The CSR
    append keeps existing sections as stored.
    """
    from zarr_vectors.core.arrays import read_vertex_fragment_index, write_chunk_fragments

    _, lg = _level(tmp_path, "e.zarrvectors")
    write_chunk_fragments(lg, (0, 0, 0), [np.arange(3)], force_explicit=True)
    write_chunk_fragments(lg, (0, 0, 0), csr=([7], [0, 1]), mode="append")
    fi = read_vertex_fragment_index(lg, (0, 0, 0))
    assert not fi.is_range(0) and fi.is_range(1)


def test_an_empty_csr_append_writes_nothing_and_reports_the_count(tmp_path):
    from zarr_vectors.core.arrays import write_chunk_fragments

    _, lg = _level(tmp_path, "n.zarrvectors")
    write_chunk_fragments(lg, (0, 0, 0), csr=([1, 2], [0, 1, 2]))
    assert write_chunk_fragments(
        lg, (0, 0, 0), csr=(np.empty(0, np.int64), [0]), mode="append",
    ) == (2, 0)


def test_device_arrays_are_copied_off_once(tmp_path):
    from tests._fake_device import FakeDeviceArray
    from zarr_vectors import _xp
    from zarr_vectors.core.arrays import write_chunk_fragments

    _, lg = _level(tmp_path, "d.zarrvectors")
    indices, offsets = _csr(_random_fragments(np.random.default_rng(2), 20))
    with _xp.count_transfers() as stats:
        write_chunk_fragments(
            lg, (1, 1, 1), csr=(FakeDeviceArray(indices), FakeDeviceArray(offsets)),
        )
    assert stats.d2h_calls == 2


def test_exactly_one_form_is_required(tmp_path):
    from zarr_vectors.core.arrays import write_chunk_fragments

    _, lg = _level(tmp_path, "x.zarrvectors")
    with pytest.raises(ArrayError, match="exactly one"):
        write_chunk_fragments(lg, (0, 0, 0))
    with pytest.raises(ArrayError, match="exactly one"):
        write_chunk_fragments(lg, (0, 0, 0), [(0, 1)], csr=([0], [0, 1]))
