"""ChunkFragmentIndex.gather: many fragments' rows in one vectorised pass.

The definition is the per-fragment accessor: ``rows`` is the
concatenation of ``indices(f)`` over the request and ``lengths`` their
sizes. Checked over ranges, explicit lists and mixtures, on random,
unsorted, repeated and empty requests.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors.encoding.fragments import decode_fragments, encode_fragments


def _reference(fi, frags):
    parts = [fi.indices(int(f)) for f in frags]
    lengths = np.array([p.size for p in parts], dtype=np.int64)
    rows = np.concatenate(parts) if parts else np.empty(0, np.int64)
    return rows.astype(np.int64), lengths


def _index(rng, n_frags, *, explicit_share):
    fragments = []
    cursor = 0
    for _ in range(n_frags):
        count = int(rng.integers(0, 6))
        if rng.random() < explicit_share and count:
            # Not a run, so the encoder keeps it explicit.
            picks = rng.choice(500, size=count, replace=False)
            if count > 1 and np.array_equal(np.diff(np.sort(picks)), np.ones(count - 1)):
                picks[0] += 1000
            fragments.append(picks.astype(np.int64))
        else:
            fragments.append((cursor, count))
            cursor += count
    return decode_fragments(encode_fragments(fragments))


@pytest.mark.parametrize("explicit_share", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("seed", range(4))
def test_gather_equals_the_per_fragment_accessor(seed, explicit_share):
    rng = np.random.default_rng(seed)
    fi = _index(rng, 60, explicit_share=explicit_share)
    requests = [
        np.arange(len(fi)),                              # every fragment, in order
        rng.permutation(len(fi)),                        # unsorted
        rng.integers(0, len(fi), size=200),              # duplicates
        np.array([7, 7, 7, 0, 59, 0]),                   # repeats at both ends
        np.array([], dtype=np.int64),                    # nothing
        [int(rng.integers(0, len(fi)))],                 # one, as a list
    ]
    for frags in requests:
        rows, lengths = fi.gather(frags)
        want_rows, want_lengths = _reference(fi, frags)
        assert rows.dtype == np.int64 and lengths.dtype == np.int64
        np.testing.assert_array_equal(lengths, want_lengths)
        np.testing.assert_array_equal(rows, want_rows)


def test_gather_on_an_empty_index_and_empty_fragments():
    empty = decode_fragments(encode_fragments([]))
    rows, lengths = empty.gather([])
    assert rows.size == 0 and lengths.size == 0
    with pytest.raises(IndexError):
        empty.gather([0])

    fi = decode_fragments(encode_fragments([(5, 0), (5, 3), (8, 0)]))
    rows, lengths = fi.gather([0, 2, 0])
    assert rows.tolist() == [] and lengths.tolist() == [0, 0, 0]
    rows, lengths = fi.gather([2, 1, 0, 1])
    assert rows.tolist() == [5, 6, 7, 5, 6, 7] and lengths.tolist() == [0, 3, 0, 3]


def test_gather_refuses_a_fragment_out_of_range_as_indices_does():
    fi = decode_fragments(encode_fragments([(0, 2), np.array([4, 9])]))
    for bad in (-1, 2, 100):
        with pytest.raises(IndexError):
            fi.indices(bad)
        with pytest.raises(IndexError):
            fi.gather([0, bad])


def test_gather_explicit_rows_come_from_the_explicit_lists():
    fi = decode_fragments(encode_fragments([
        np.array([30, 10, 20]), (100, 2), np.array([7, 3]),
    ]))
    rows, lengths = fi.gather([2, 0, 1, 2])
    assert rows.tolist() == [7, 3, 30, 10, 20, 100, 101, 7, 3]
    assert lengths.tolist() == [2, 3, 2, 2]
