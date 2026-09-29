"""Names BRIDGE reached into internals for, now on the supported surface (A10).

Each is either the internal object itself, re-exported, or a thin form of
code that already existed; the tests pin that they agree with what they
replace.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._store_compare import assert_stores_identical
from zarr_vectors import building
from zarr_vectors.building import (
    create_link_attributes_array,
    create_link_segments,
    create_links_array,
    create_links_family,
    create_store,
    get_resolution_level,
    read_object_manifests,
    read_object_manifests_csr,
    write_chunk_vertices,
)
from zarr_vectors.core.arrays import write_object_index
from zarr_vectors.core.paths import intra_offsets

pytestmark = pytest.mark.vlen_only  # these pick their layouts themselves


def _level(tmp_path, name, **kw):
    (tmp_path / name).mkdir()
    path = tmp_path / name / "s.zarrvectors"
    root = create_store(
        path, bounds=([0.0] * 3, [100.0] * 3), chunk_shape=(50.0,) * 3, **kw,
    )
    return path, get_resolution_level(root, 0)


# --------------------------------------------------------------------
# Constants


def test_the_format_names_live_in_constants():
    from zarr_vectors import constants
    from zarr_vectors.core import ome
    from zarr_vectors.encoding import categorical

    assert constants.OME_ATTRS_KEY is ome.OME_ATTRS_KEY == "ome"
    assert constants.OME_VERSION is ome.OME_VERSION
    assert constants.DICTIONARY_ENCODING is categorical.DICTIONARY_ENCODING == "dictionary"


def test_the_layout_sentinels_are_the_ones_the_index_is_stamped_with(tmp_path):
    from zarr_vectors.core import dense_manifests

    assert building.OBJECT_INDEX_LAYOUT_DENSE == dense_manifests.OBJECT_INDEX_LAYOUT_DENSE
    assert building.OBJECT_INDEX_LAYOUT_V2 == "vlen_manifests_v2"
    _, lg = _level(tmp_path, "d", manifest_layout="dense")
    write_object_index(lg, {0: [((0, 0, 0), 1)]}, 3)
    assert lg.read_array_meta("object_index")["layout"] == building.OBJECT_INDEX_LAYOUT_DENSE


# --------------------------------------------------------------------
# Small readers


def test_decode_fragment_index_is_the_read_without_the_read(tmp_path):
    _, lg = _level(tmp_path, "f")
    write_chunk_vertices(
        lg, (0, 0, 0), [np.zeros((3, 3), np.float32), np.ones((2, 3), np.float32)],
    )
    blob = lg.read_bytes("vertex_fragments", "0.0.0")
    got = building.decode_fragment_index(blob)
    want = building.read_vertex_fragment_index(lg, (0, 0, 0))
    assert len(got) == len(want) == 2
    for f in range(len(got)):
        np.testing.assert_array_equal(got.indices(f), want.indices(f))


def test_user_metadata_is_what_the_dataset_offers(tmp_path):
    import zarr_vectors as zv

    path, lg = _level(tmp_path, "m")
    root = building.open_store(str(path), mode="r+")
    building.user_metadata(root)["bridge"]["format"] = 2
    ds = zv.open(str(path))
    assert dict(ds.metadata["bridge"]) == {"format": 2}
    assert "bridge" in building.user_metadata(root).names()


@pytest.mark.parametrize("layout", ["vlen", "dense"])
def test_is_dense_index(tmp_path, layout):
    _, lg = _level(tmp_path, layout, manifest_layout=layout)
    assert building.is_dense_index(lg) is False  # no index yet
    write_object_index(lg, {0: [((0, 0, 0), 1)]}, 3)
    assert building.is_dense_index(lg) is (layout == "dense")


# --------------------------------------------------------------------
# read_object_manifests_csr


def _manifests(n, rng):
    return {
        10 * oid: [
            (tuple(int(c) for c in rng.integers(0, 2, 3)), int(rng.integers(0, 2**20)))
            for _ in range(int(rng.integers(0, 4)))
        ]
        for oid in range(n)
    }


@pytest.mark.parametrize("layout", ["vlen", "dense"])
def test_a_subset_as_csr_matches_the_dict_reader(tmp_path, layout):
    rng = np.random.default_rng(0)
    _, lg = _level(tmp_path, layout, manifest_layout=layout)
    manifests = _manifests(50, rng)
    write_object_index(lg, manifests, 3)
    ids = [470, 0, 999, 130, 20, 470]  # unordered, one absent, one repeated
    csr = read_object_manifests_csr(lg, ids)
    by_id = read_object_manifests(lg, ids=ids)
    assert csr.object_ids.tolist() == [470, 0, 130, 20, 470]
    for o, oid in enumerate(csr.object_ids.tolist()):
        lo, hi = int(csr.offsets[o]), int(csr.offsets[o + 1])
        got = [
            (tuple(int(c) for c in csr.chunk_coords[r]), int(csr.fragment_idx[r]))
            for r in range(lo, hi)
        ]
        assert got == [(tuple(c), int(f)) for c, f in by_id[oid]]


@pytest.mark.parametrize("layout", ["vlen", "dense"])
def test_no_ids_found_is_an_empty_csr(tmp_path, layout):
    _, lg = _level(tmp_path, layout, manifest_layout=layout)
    write_object_index(lg, {0: [((0, 0, 0), 1)]}, 3)
    offsets, coords, frags = read_object_manifests_csr(lg, [5, 6])
    assert offsets.tolist() == [0] and coords.shape == (0, 3) and frags.size == 0


# --------------------------------------------------------------------
# create_link_segments


def _loop_like_bridge(lg, *, directed, store, attributes, intra_attributes):
    """What a caller did before: one create call per segment and attribute."""
    import itertools

    create_links_family(lg, delta=0, link_width=2, sid_ndim=3, directed=directed, store=store)
    positive_only = not directed and store == "canonical"
    cross = [
        (d,) for d in itertools.product((-1, 0, 1), repeat=3)
        if any(d) and not (positive_only and next(x for x in d if x) < 0)
    ]
    for offsets in cross:
        create_links_array(lg, 2, dtype="int64", delta=0, sid_ndim=3, offsets=offsets,
                           directed=directed, store=store)
        for name, (dt, shape) in attributes.items():
            create_link_attributes_array(lg, name, dt, delta=0, sid_ndim=3,
                                         offsets=offsets, row_shape=shape)
    intra = intra_offsets(3, 2)
    create_links_array(lg, 2, dtype="int32", delta=0, sid_ndim=3, offsets=intra,
                       directed=directed, store=store)
    for name, (dt, shape) in intra_attributes.items():
        create_link_attributes_array(lg, name, dt, delta=0, sid_ndim=3,
                                     offsets=intra, row_shape=shape)
    return len(cross)


@pytest.mark.parametrize(("directed", "store", "n_cross"), [
    (False, "canonical", 13),
    (True, "canonical", 26),
    (False, "duplicate", 26),
])
def test_one_call_leaves_the_store_the_loop_leaves(tmp_path, directed, store, n_cross):
    attributes = {"w": ("float32", []), "port": ("int8", [2])}
    intra_attributes = {"len": ("float64", [])}
    ref_path, ref = _level(tmp_path, "ref")
    assert _loop_like_bridge(
        ref, directed=directed, store=store,
        attributes=attributes, intra_attributes=intra_attributes,
    ) == n_cross

    path, lg = _level(tmp_path, "one")
    created = create_link_segments(
        lg, sid_ndim=3, directed=directed, store=store, dtype="int64",
        intra_dtype="int32", attributes=attributes, intra_attributes=intra_attributes,
    )
    assert len(created) == (n_cross + 1) + n_cross * 2 + 1
    assert_stores_identical(ref_path, path)
    # Idempotent: a second call changes nothing.
    assert create_link_segments(
        lg, sid_ndim=3, directed=directed, store=store, dtype="int64",
        intra_dtype="int32", attributes=attributes, intra_attributes=intra_attributes,
    ) == created
    assert_stores_identical(ref_path, path)


def test_a_segment_of_another_policy_is_refused(tmp_path):
    from zarr_vectors.exceptions import ArrayError

    _, lg = _level(tmp_path, "p")
    create_link_segments(lg, sid_ndim=3, directed=False, store="canonical")
    with pytest.raises(ArrayError):
        create_link_segments(lg, sid_ndim=3, directed=True, store="canonical")
    with pytest.raises(ArrayError, match="store must be"):
        create_link_segments(lg, sid_ndim=3, store="mirrored")


def test_the_promised_group_methods_include_the_node_cache():
    assert {"cached_nodes", "prime_nodes"} <= building.GROUP_SUPPORTED_METHODS
