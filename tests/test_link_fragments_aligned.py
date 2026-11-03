"""``link_fragments_aligned``: intra-chunk faces filed by vertex fragment.

Spec §7.5 says each link fragment is "the set of link rows belonging to
one vertex fragment".  ``write_mesh`` wrote one vertex fragment per
object per chunk but every chunk's faces as a single link fragment, so
an object's faces could not be told apart from its neighbours' without
decoding all of them.  These tests pin the grouped layout, the level
claim that tells readers they may rely on it, and the writes that must
withdraw that claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors.core.arrays import (
    _group_intra_by_fragment,
    chunk_link_fragments_aligned,
    list_chunk_keys,
    read_chunk_links,
    read_vertex_fragment_index,
    stamp_link_fragments_aligned,
    write_chunk_links,
)
from zarr_vectors.core.paths import format_offsets, intra_offsets
from zarr_vectors.core.store import (
    get_resolution_level,
    open_store,
    read_level_metadata,
)
from zarr_vectors.types.meshes import read_mesh, write_mesh

from .test_meshes import BOUNDS, CHUNK, _face_set, _patch_mesh


def _flag(store, level=0):
    return read_level_metadata(open_store(store), level).link_fragments_aligned


def _set_flag(store, value=True, level=0):
    """Force the claim, without going through a writer."""
    root = open_store(store, mode="a")
    lg = get_resolution_level(root, level)
    meta = dict(lg.attrs.get("zarr_vectors_level"))
    if value:
        meta["link_fragments_aligned"] = True
    else:
        meta.pop("link_fragments_aligned", None)
    lg.attrs.update({"zarr_vectors_level": meta})


@pytest.fixture
def store(tmp_path):
    v, f, o = _patch_mesh()
    path = tmp_path / "m.zv"
    write_mesh(str(path), v, f, chunk_shape=CHUNK, bounds=BOUNDS, object_ids=o)
    return path, v, f, o


def _multi_object_chunk(lg):
    """A chunk holding intra-chunk faces of more than one object."""
    for cc in list_chunk_keys(lg):
        if (
            read_vertex_fragment_index(lg, cc).num_fragments > 1
            and read_chunk_links(lg, cc)
        ):
            return cc
    raise AssertionError("fixture has no chunk shared by two objects")


# ---------------------------------------------------------------------------
# The layout
# ---------------------------------------------------------------------------


def test_write_mesh_files_faces_by_vertex_fragment(store):
    path, *_ = store
    lg = get_resolution_level(open_store(str(path)), 0)
    keys = list_chunk_keys(lg)
    assert all(chunk_link_fragments_aligned(lg, cc) for cc in keys)
    cc = _multi_object_chunk(lg)
    groups = read_chunk_links(lg, cc)
    assert len(groups) == read_vertex_fragment_index(lg, cc).num_fragments


def test_write_mesh_stamps_the_claim(store):
    path, *_ = store
    assert _flag(path) is True


def test_full_read_is_unchanged(store):
    """Regrouping reorders rows within a cell; the face set must not move."""
    path, v, f, _ = store
    got = read_mesh(str(path))
    assert got["face_count"] == len(f)
    assert _face_set(got["vertices"], got["faces"]) == _face_set(v, f)


def test_grouping_permutes_partition_indices_with_rows():
    """``write_links`` returns the partition link attributes align to; the
    regrouping must carry each row's input index with it."""
    seg = format_offsets(intra_offsets(3, 3))
    src = (0, 0, 0)
    rows = np.array([[5, 6, 7], [0, 1, 2], [3, 4, 5], [6, 7, 8], [1, 2, 3]])
    idx = np.array([10, 11, 12, 13, 14])
    starts = np.array([0, 3, 6])           # fragments [0,3) [3,6) [6,...)
    out, sizes = _group_intra_by_fragment(
        {(seg, src): (rows, idx)}, {src: starts}, sid_ndim=3, link_width=3,
    )
    got_rows, got_idx = out[(seg, src)]
    assert sizes[(seg, src)].tolist() == [2, 2, 1]
    assert got_rows[:, 0].tolist() == [0, 1, 5, 3, 6]
    assert got_idx.tolist() == [11, 14, 10, 12, 13]
    # Each row still sits beside the input index it came with.
    original = {int(i): tuple(r) for i, r in zip(idx, rows.tolist())}
    assert all(original[int(i)] == tuple(r) for i, r in zip(got_idx, got_rows.tolist()))


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def test_object_read_same_with_and_without_the_claim(store):
    path, v, f, o = store
    with_claim = {int(k): read_mesh(str(path), object_ids=[int(k)]) for k in np.unique(o)}
    _set_flag(path, False)
    for k, got in with_claim.items():
        without = read_mesh(str(path), object_ids=[k])
        assert np.array_equal(got["vertices"], without["vertices"])
        assert _face_set(got["vertices"], got["faces"]) == _face_set(
            without["vertices"], without["faces"],
        )
        want = f[(o == k)[f].all(axis=1)]
        assert _face_set(got["vertices"], got["faces"]) == _face_set(v, want)


def test_ungrouped_cell_reads_correctly_once_the_claim_is_gone(store):
    """A cell rewritten as one fragment -- the pre-grouping layout -- must
    still read correctly, which it can only do if the rewrite withdrew
    the claim."""
    path, v, f, o = store
    root = open_store(str(path), mode="a")
    lg = get_resolution_level(root, 0)
    cc = _multi_object_chunk(lg)
    write_chunk_links(lg, cc, [np.concatenate(read_chunk_links(lg, cc))])
    assert _flag(path) is False
    for k in np.unique(o):
        got = read_mesh(str(path), object_ids=[int(k)])
        want = f[(o == k)[f].all(axis=1)]
        assert _face_set(got["vertices"], got["faces"]) == _face_set(v, want)


# ---------------------------------------------------------------------------
# The claim's lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["link_fragments", "vertex_fragments", "intra"])
def test_writes_that_can_break_alignment_clear_the_claim(store, target):
    path, *_ = store
    root = open_store(str(path), mode="a")
    lg = get_resolution_level(root, 0)
    cc = list_chunk_keys(lg)[0]
    key = ".".join(map(str, cc))
    name = {
        "link_fragments": "link_fragments",
        "vertex_fragments": "vertex_fragments",
        "intra": f"links/0/{format_offsets(intra_offsets(3, 3))}",
    }[target]
    lg.write_bytes(name, key, lg.read_bytes(name, key))
    assert _flag(path) is False


def test_cross_chunk_and_vertex_writes_keep_the_claim(store):
    """Neither changes which intra-chunk record belongs to which fragment."""
    path, *_ = store
    root = open_store(str(path), mode="a")
    lg = get_resolution_level(root, 0)
    cross = next(
        seg for seg in lg["links/0"].child_names()
        if seg != format_offsets(intra_offsets(3, 3))
    )
    name = f"links/0/{cross}"
    key = lg.list_chunks(name)[0]
    lg.write_bytes(name, key, lg.read_bytes(name, key))
    cc = list_chunk_keys(lg)[0]
    vkey = ".".join(map(str, cc))
    lg.write_bytes("vertices", vkey, lg.read_bytes("vertices", vkey))
    assert _flag(path) is True


def test_stamp_refuses_an_ungrouped_level(store):
    path, *_ = store
    root = open_store(str(path), mode="a")
    lg = get_resolution_level(root, 0)
    cc = _multi_object_chunk(lg)
    write_chunk_links(lg, cc, [np.concatenate(read_chunk_links(lg, cc))])
    assert stamp_link_fragments_aligned(lg) is False
    assert _flag(path) is False


# ---------------------------------------------------------------------------
# The validator
# ---------------------------------------------------------------------------


def test_validate_consistency_reports_a_lying_claim(store):
    from zarr_vectors.validate.consistency import validate_consistency

    path, *_ = store
    root = open_store(str(path), mode="a")
    lg = get_resolution_level(root, 0)
    cc = _multi_object_chunk(lg)
    # Same number of fragments, wrong rows in each: only a decode can tell.
    groups = read_chunk_links(lg, cc)
    write_chunk_links(lg, cc, groups[::-1])
    _set_flag(path, True)
    report = validate_consistency(str(path))
    assert any("link_fragments_aligned" in e for e in report.errors), report.errors


def test_validate_consistency_accepts_an_honest_claim(store):
    from zarr_vectors.validate.consistency import validate_consistency

    path, *_ = store
    report = validate_consistency(str(path))
    assert not any("link_fragments_aligned" in e for e in report.errors), report.errors
