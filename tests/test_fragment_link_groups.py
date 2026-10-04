"""``LevelMetadata.fragment_link_groups``: link groups that follow fragments.

When every chunk of a level holds one intra-chunk link group per vertex
fragment, in fragment order, an object's manifest -- which names vertex
fragments -- also names its link groups, and a reader can fetch one object's
links alone.  Invariants under test:

* **G1** ``write_mesh`` groups faces per fragment (whatever the input face
  order), stamps the level and the root capability, and loses no face.
* **G2** The claim is verified, not asserted: a level whose groups do not
  follow its fragments is not stamped.
* **G3** Any later write to ``vertex_fragments``, ``link_fragments`` or the
  intra-chunk link array clears the claim; a writer re-stamps.
* **G4** ``validate_consistency`` reports a claim that does not hold.
* **G5** ``index_fragment_link_groups`` brings an existing level into the
  layout with the least rewriting, refuses levels where a link joins two
  fragments, honours ``dry_run`` and re-checks with ``verify``.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors.constants import CAP_FRAGMENT_LINK_GROUPS
from zarr_vectors.core.arrays import (
    list_chunk_keys,
    read_chunk_links,
    read_vertex_fragment_index,
    write_chunk_links,
)
from zarr_vectors.core.link_groups import (
    fragment_of_rows,
    index_fragment_link_groups,
    intra_links_name,
    split_links_by_fragment,
    stamp_fragment_link_groups,
    verify_fragment_link_groups,
    write_link_groups,
)
from zarr_vectors.core.metadata import LevelMetadata
from zarr_vectors.core.store import (
    get_resolution_level,
    open_store,
    read_level_metadata,
    read_root_metadata,
)
from zarr_vectors.types.meshes import read_mesh, write_mesh
from zarr_vectors.validate.consistency import validate_consistency


def _icosphere(n):
    t = (1 + 5 ** 0.5) / 2
    v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
         [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9],
         [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2],
         [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
         [8, 6, 7], [9, 8, 1]]
    verts = [np.array(x, float) / np.linalg.norm(x) for x in v]
    for _ in range(n):
        cache: dict = {}

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                cache[key] = len(verts)
                verts.append(m / np.linalg.norm(m))
            return cache[key]

        f = [tri for a, b, c in f for tri in (
            [a, mid(a, b), mid(c, a)], [b, mid(b, c), mid(a, b)],
            [c, mid(c, a), mid(b, c)], [mid(a, b), mid(b, c), mid(c, a)])]
    return np.array(verts), np.array(f)


def _spheres(tmp_path, name="s.zv", shuffle=True):
    """Two touching spheres over several chunks, faces in random order."""
    v, f = _icosphere(3)
    verts = np.concatenate([v * 12 + 20, v * 10 + [40, 26, 22]]).astype("float32")
    faces = np.concatenate([f, f + len(v)])
    if shuffle:
        faces = faces[np.random.default_rng(0).permutation(len(faces))]
    oids = np.repeat(np.arange(2), len(v))
    path = str(tmp_path / name)
    write_mesh(path, verts, faces, chunk_shape=(16.0,) * 3, object_ids=oids,
               bounds=([0, 0, 0], [64, 48, 40]))
    return path, verts, faces


def _flag(path, level=0):
    return read_level_metadata(open_store(path), level).fragment_link_groups


def _level(path, level=0, mode="r"):
    return get_resolution_level(open_store(path, mode=mode), level)


def _face_set(vertices, faces):
    v = np.round(np.asarray(vertices, np.float64), 4)
    return {tuple(sorted(map(tuple, v[face]))) for face in np.asarray(faces)}


def _flatten_to_single_groups(path):
    """Re-cut every chunk to one group, as writers before this did."""
    lg = _level(path, mode="r+")
    name, width = intra_links_name(lg)
    for cc in list_chunk_keys(lg, name):
        groups = read_chunk_links(lg, cc, link_width=width)
        write_link_groups(lg, cc, [sum(len(g) for g in groups)])


def test_g1_write_mesh_groups_faces_per_fragment_and_stamps(tmp_path):
    path, verts, faces = _spheres(tmp_path)
    assert _flag(path)
    assert CAP_FRAGMENT_LINK_GROUPS in read_root_metadata(open_store(path)).format_capabilities
    lg = _level(path)
    name, width = intra_links_name(lg)
    checked = 0
    for cc in list_chunk_keys(lg, name):
        groups = read_chunk_links(lg, cc, link_width=width)
        fragments = read_vertex_fragment_index(lg, cc)
        assert len(groups) == fragments.num_fragments
        owner = fragment_of_rows(fragments)
        for k, g in enumerate(groups):
            assert (owner[np.asarray(g)] == k).all()
        checked += 1
    assert checked > 0
    mesh = read_mesh(path)
    assert len(mesh["faces"]) == len(faces)
    assert _face_set(mesh["vertices"], mesh["faces"]) == _face_set(verts, faces)


def test_g1_round_trips_in_level_metadata():
    meta = LevelMetadata(level=1, vertex_count=10, arrays_present=["vertices"],
                         fragment_link_groups=True)
    assert LevelMetadata.from_dict(meta.to_dict()).fragment_link_groups
    plain = LevelMetadata(level=1, vertex_count=10, arrays_present=["vertices"])
    assert "fragment_link_groups" not in plain.to_dict()["zarr_vectors_level"]


def test_g2_a_level_whose_groups_do_not_follow_fragments_is_not_stamped(tmp_path):
    path, _, _ = _spheres(tmp_path)
    _flatten_to_single_groups(path)
    assert not _flag(path)
    lg = _level(path, mode="r+")
    assert verify_fragment_link_groups(lg) is not None
    assert not stamp_fragment_link_groups(lg)
    assert not _flag(path)


def test_g3_writes_clear_the_claim_and_a_writer_restamps(tmp_path):
    path, _, _ = _spheres(tmp_path)
    lg = _level(path, mode="r+")
    name, width = intra_links_name(lg)
    cc = list_chunk_keys(lg, name)[0]
    groups = read_chunk_links(lg, cc, link_width=width)
    # Rewriting a cell -- even with the same groups -- withdraws the claim.
    write_chunk_links(lg, cc, groups, delta=0, link_width=width)
    assert not _flag(path)
    assert stamp_fragment_link_groups(lg, open_store(path, mode="r+"))
    assert _flag(path)
    # So does a write to the fragments the groups follow.
    lg2 = _level(path, mode="r+")
    raw = lg2.read_bytes("vertex_fragments", ".".join(map(str, cc)))
    lg2.write_bytes("vertex_fragments", ".".join(map(str, cc)), raw)
    assert not _flag(path)


def test_g4_the_validator_catches_a_claim_that_does_not_hold(tmp_path):
    path, _, _ = _spheres(tmp_path)
    result = validate_consistency(path)
    assert not any("fragment_link_groups" in e for e in result.errors)
    assert any("fragment_link_groups holds" in p for p in result.passed)
    _flatten_to_single_groups(path)
    # Forced back on without a writer, as a stale claim would be.
    lg = _level(path, mode="r+")
    meta = dict(lg.attrs.get("zarr_vectors_level"))
    meta["fragment_link_groups"] = True
    lg.attrs.update({"zarr_vectors_level": meta})
    result = validate_consistency(path)
    assert any("fragment_link_groups" in e for e in result.errors)


def test_g5_indexing_recuts_groups_without_moving_rows(tmp_path):
    path, verts, faces = _spheres(tmp_path)
    _flatten_to_single_groups(path)
    before = read_mesh(path)
    report = index_fragment_link_groups(path, levels=[0])[0]
    assert report["stamped"] and report["regrouped"] > 0 and report["reordered"] == 0
    assert _flag(path)
    after = read_mesh(path)
    np.testing.assert_array_equal(before["faces"], after["faces"])
    again = index_fragment_link_groups(path, levels=[0])[0]
    assert again["skipped"] == "already stamped"
    rechecked = index_fragment_link_groups(path, levels=[0], verify=True)[0]
    assert rechecked["stamped"] and rechecked["regrouped"] == 0


def test_g5_indexing_reorders_rows_when_it_must(tmp_path):
    path, verts, faces = _spheres(tmp_path)
    lg = _level(path, mode="r+")
    name, width = intra_links_name(lg)
    reversed_any = False
    for cc in list_chunk_keys(lg, name):
        rows = np.concatenate(read_chunk_links(lg, cc, link_width=width))
        if len(np.unique(fragment_of_rows(read_vertex_fragment_index(lg, cc))[rows[:, 0]])) > 1:
            reversed_any = True
        write_chunk_links(lg, cc, [rows[::-1]], delta=0, link_width=width)
    assert reversed_any
    report = index_fragment_link_groups(path, levels=[0])[0]
    assert report["stamped"] and report["reordered"] > 0
    mesh = read_mesh(path)
    assert _face_set(mesh["vertices"], mesh["faces"]) == _face_set(verts, faces)


def test_g5_dry_run_writes_nothing(tmp_path):
    path, _, _ = _spheres(tmp_path)
    _flatten_to_single_groups(path)
    report = index_fragment_link_groups(path, levels=[0], dry_run=True)[0]
    assert report["would_stamp"] and report["regrouped"] > 0
    assert not _flag(path)
    assert verify_fragment_link_groups(_level(path)) is not None


def test_g5_a_link_joining_two_fragments_cannot_be_grouped():
    class Fragments:
        num_fragments = 2

        def is_range(self, f):
            return True

        def range(self, f):
            return (0, 3) if f == 0 else (3, 3)

    owner = fragment_of_rows(Fragments())
    assert owner.tolist() == [0, 0, 0, 1, 1, 1]
    rows = np.array([[3, 4, 5], [0, 1, 2], [2, 1, 0]])
    groups = split_links_by_fragment(rows, owner, 2)
    assert [g.tolist() for g in groups] == [[[0, 1, 2], [2, 1, 0]], [[3, 4, 5]]]
    assert split_links_by_fragment(np.array([[0, 1, 3]]), owner, 2) is None
    assert split_links_by_fragment(np.array([[0, 1, 9]]), owner, 2) is None
    assert [len(g) for g in split_links_by_fragment(np.zeros((0, 3), int), owner, 2)] == [0, 0]


def test_g1_draco_meshes_are_not_stamped(tmp_path):
    pytest.importorskip("DracoPy")
    v, f = _icosphere(2)
    path = str(tmp_path / "d.zv")
    write_mesh(path, (v * 10 + 20).astype("float32"), f, chunk_shape=(16.0,) * 3,
               encoding="draco")
    assert not _flag(path)
