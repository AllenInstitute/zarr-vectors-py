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
* **G6** The rule is the whole rule: link groups hold every row of the cell
  exactly once, and vertex fragments are disjoint.
* **G7** What a writer recorded is never trusted over the store, and a
  claim is withdrawn (or never stamped) whatever handle, batch or helper is
  involved.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors.constants import CAP_FRAGMENT_LINK_GROUPS
from zarr_vectors.core.arrays import (
    list_chunk_keys,
    read_chunk_links,
    read_vertex_fragment_index,
    stamp_fragments_tile,
    write_chunk_links,
)
from zarr_vectors.core.group import Group, _is_intra_links_array
from zarr_vectors.core.link_groups import (
    SHARED_ROW,
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
    update_level_metadata,
)
from zarr_vectors.encoding.fragments import decode_fragments, encode_fragments
from zarr_vectors.exceptions import MetadataError, StoreError
from zarr_vectors.sharding.io import shard_store, unshard_store
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


def _claims(path, level=0):
    meta = read_level_metadata(open_store(path), level)
    return meta.fragment_link_groups, meta.fragments_tile


def _claim_errors(path):
    return [e for e in validate_consistency(path).errors
            if "fragment_link_groups" in e]


def _two_fragment_chunk(lg):
    """A chunk with two vertex fragments that both have faces."""
    name, width = intra_links_name(lg)
    for cc in list_chunk_keys(lg, name):
        fragments = read_vertex_fragment_index(lg, cc)
        groups = read_chunk_links(lg, cc, link_width=width)
        if fragments.num_fragments == 2 and all(len(g) for g in groups):
            return cc, ".".join(map(str, cc)), groups, fragments, width
    raise AssertionError("no chunk with two fragments that both have faces")


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
    owner = fragment_of_rows(decode_fragments(encode_fragments([(0, 3), (3, 3)])))
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


# --------------------------------------------------------------------- G5


def test_g5_indexing_follows_the_stored_row_order(tmp_path):
    """The old groups' concatenation is not the stored order: physical rows
    [fragment 1's, fragment 0's], three ranges listing fragment 0's first.
    A re-cut of the index alone would put fragment 1's rows in group 0."""
    path, verts, faces = _spheres(tmp_path)
    lg = _level(path, mode="r+")
    cc, key, (a0, a1), _, width = _two_fragment_chunk(lg)
    n0, n1 = len(a0), len(a1)
    write_chunk_links(lg, cc, [a1, a0], delta=0, link_width=width)
    lg.write_bytes("link_fragments", key, encode_fragments(
        [(n1, n0 - 5), (n1 + n0 - 5, 5), (0, n1)],
    ))
    report = index_fragment_link_groups(path, levels=[0])[0]
    assert report["stamped"] and report["reordered"] >= 1
    assert _claim_errors(path) == []
    groups = read_chunk_links(_level(path), cc, link_width=width)
    np.testing.assert_array_equal(groups[0], a0)
    np.testing.assert_array_equal(groups[1], a1)
    mesh = read_mesh(path)
    assert _face_set(mesh["vertices"], mesh["faces"]) == _face_set(verts, faces)


def test_g5_indexing_recuts_only_contiguous_groups_over_ordered_rows(tmp_path):
    """Rows already in fragment order under one contiguous group: only the
    index is rewritten, and the stamp takes the recorded groups from the
    stored rows, not from what was intended."""
    path, _, _ = _spheres(tmp_path)
    _flatten_to_single_groups(path)
    lg = _level(path)
    name, _ = intra_links_name(lg)
    before = {k: lg.read_bytes(name, ".".join(map(str, k)))
              for k in list_chunk_keys(lg, name)}
    report = index_fragment_link_groups(path, levels=[0])[0]
    assert report["stamped"] and report["regrouped"] > 0 and report["reordered"] == 0
    after = _level(path)
    for k, raw in before.items():
        assert after.read_bytes(name, ".".join(map(str, k))) == raw
    assert verify_fragment_link_groups(after) is None


def test_g5_verify_withdraws_a_claim_it_cannot_repair(tmp_path):
    path, _, _ = _spheres(tmp_path)
    lg = _level(path, mode="r+")
    cc, _, (g0, g1), _, width = _two_fragment_chunk(lg)
    g0, g1 = g0.copy(), g1.copy()
    g0[0, 0], g1[0, 0] = g1[0, 0], g0[0, 0]      # two faces now span both
    write_chunk_links(lg, cc, [g0, g1], delta=0, link_width=width)
    meta = dict(lg.attrs.get("zarr_vectors_level"))
    meta["fragment_link_groups"] = True          # a stale claim
    lg.attrs.update({"zarr_vectors_level": meta})
    dry = index_fragment_link_groups(path, levels=[0], verify=True, dry_run=True)[0]
    assert dry["would_withdraw"] and _flag(path)
    report = index_fragment_link_groups(path, levels=[0], verify=True)[0]
    assert "joins two vertex fragments" in report["skipped"]
    assert report["withdrawn"] and not _flag(path)


def test_g5_a_failed_stamp_withdraws_a_stale_claim(tmp_path):
    path, _, _ = _spheres(tmp_path)
    _flatten_to_single_groups(path)
    lg = _level(path, mode="r+")
    meta = dict(lg.attrs.get("zarr_vectors_level"))
    meta["fragment_link_groups"] = True
    lg.attrs.update({"zarr_vectors_level": meta})
    assert not stamp_fragment_link_groups(_level(path, mode="r+"))
    assert not _flag(path)


# --------------------------------------------------------------------- G6


def test_g6_groups_must_hold_every_row_of_the_cell(tmp_path):
    """Rows in no group: a reader of the whole cell sees them, a reader of
    one object's groups does not."""
    path, _, _ = _spheres(tmp_path)
    lg = _level(path, mode="r+")
    cc, key, (g0, g1), _, _ = _two_fragment_chunk(lg)
    lg.write_bytes("link_fragments", key, encode_fragments(
        [(0, len(g0)), (len(g0), len(g1) - 10)],
    ))
    fresh = _level(path, mode="r+")
    assert "exactly once" in verify_fragment_link_groups(fresh)
    assert not stamp_fragment_link_groups(fresh)
    report = index_fragment_link_groups(path, levels=[0])[0]
    assert "exactly once" in report["skipped"] and not _flag(path)


def test_g6_no_row_may_be_in_two_groups(tmp_path):
    path, _, _ = _spheres(tmp_path)
    lg = _level(path, mode="r+")
    cc, key, (g0, g1), _, _ = _two_fragment_chunk(lg)
    n0, n1 = len(g0), len(g1)
    lg.write_bytes("link_fragments", key, encode_fragments(
        [(0, n0), (n0 - 1, n1 + 1)],
    ))
    assert "exactly once" in verify_fragment_link_groups(_level(path))


def test_g6_vertex_fragments_must_be_disjoint(tmp_path):
    """Fragment 0 widened over fragment 1: fragment 1's faces lie in both,
    and group 0 cannot hold them too."""
    path, _, _ = _spheres(tmp_path)
    lg = _level(path, mode="r+")
    cc, key, _, fragments, _ = _two_fragment_chunk(lg)
    (s0, n0), (s1, n1) = fragments.range(0), fragments.range(1)
    shared = encode_fragments([(s0, n0 + n1), (s1, n1)])
    assert (fragment_of_rows(decode_fragments(shared)) == SHARED_ROW).sum() == n1
    lg.write_bytes("vertex_fragments", key, shared)
    fresh = _level(path, mode="r+")
    assert "overlap" in verify_fragment_link_groups(fresh)
    assert not stamp_fragment_link_groups(fresh)
    assert "overlap" in index_fragment_link_groups(path, levels=[0])[0]["skipped"]


def test_g6_index_list_fragments_are_checked_row_by_row(tmp_path):
    """Index-list fragments that swap one row: each group's endpoint range
    still fits inside its fragment's span, but one of its rows does not."""
    path, _, _ = _spheres(tmp_path)
    lg = _level(path, mode="r+")
    cc, key, (g0, g1), fragments, width = _two_fragment_chunk(lg)
    (s0, n0), (s1, n1) = fragments.range(0), fragments.range(1)
    a, b = int(g0[0, 0]), int(g1[0, 0])
    rows0, rows1 = np.arange(s0, s0 + n0), np.arange(s1, s1 + n1)
    f0 = np.where(rows0 == a, b, rows0)
    f1 = np.where(rows1 == b, a, rows1)
    write_chunk_links(lg, cc, [g0, g1], delta=0, link_width=width)
    lg.write_bytes("vertex_fragments", key, encode_fragments([f0, f1]))
    assert not stamp_fragment_link_groups(lg)
    assert "leaves its fragment" in verify_fragment_link_groups(_level(path))


# --------------------------------------------------------------------- G7


def test_g7_a_stale_record_is_not_trusted_over_the_store(tmp_path):
    """Recorded groups, then the index re-cut without new ones: the stamp
    must look at what is stored."""
    path, _, _ = _spheres(tmp_path)
    lg = _level(path, mode="r+")
    cc, key, (g0, g1), _, width = _two_fragment_chunk(lg)
    write_chunk_links(lg, cc, [g0, g1], delta=0, link_width=width)
    assert isinstance(lg._link_group_bounds[key], np.ndarray)
    write_link_groups(lg, cc, [len(g0) + 1, len(g1) - 1])
    assert key not in (lg._link_group_bounds or {})
    assert not stamp_fragment_link_groups(lg) and not _flag(path)


def test_g7_a_record_another_handle_made_stale_is_not_trusted(tmp_path):
    path, _, _ = _spheres(tmp_path)
    lg = _level(path, mode="r+")
    cc, key, (g0, g1), _, width = _two_fragment_chunk(lg)
    write_chunk_links(lg, cc, [g0, g1], delta=0, link_width=width)
    write_link_groups(_level(path, mode="r+"), cc, [len(g0) + 1, len(g1) - 1])
    assert not stamp_fragment_link_groups(lg) and not _flag(path)
    assert _claim_errors(path) == []


def test_g7_a_handle_opened_before_a_stamp_still_withdraws_it(tmp_path):
    path, _, _ = _spheres(tmp_path)
    _flatten_to_single_groups(path)
    early = _level(path, mode="r+")              # opened before the stamp
    assert index_fragment_link_groups(path, levels=[0])[0]["stamped"]
    cc, _, groups, _, _ = _two_fragment_chunk(early)
    write_link_groups(early, cc, [sum(len(g) for g in groups)])
    assert not _flag(path)


def test_g7_a_stale_handle_cannot_put_a_withdrawn_claim_back(tmp_path):
    path, _, _ = _spheres(tmp_path)
    stale = _level(path, mode="r+")              # loaded while stamped
    assert stale.attrs.get("zarr_vectors_level")["fragment_link_groups"]
    other = _level(path, mode="r+")
    cc, _, groups, _, _ = _two_fragment_chunk(other)
    write_link_groups(other, cc, [sum(len(g) for g in groups)])
    assert not _flag(path)
    update_level_metadata(stale, add_arrays_present="vertex_attributes")
    assert not _flag(path)
    assert "vertex_attributes" in read_level_metadata(open_store(path), 0).arrays_present
    # ... and the stale handle's own copy caught up.
    assert "fragment_link_groups" not in stale.attrs.get("zarr_vectors_level")


def test_g7_claims_are_not_fields_to_set(tmp_path):
    path, _, _ = _spheres(tmp_path)
    _flatten_to_single_groups(path)
    lg = _level(path, mode="r+")
    for claim in ("fragment_link_groups", "fragments_tile"):
        with pytest.raises(MetadataError, match="claim"):
            update_level_metadata(lg, **{claim: True})
    assert not _flag(path)


def test_g7_stamps_refuse_to_run_with_writes_still_queued(tmp_path):
    path, _, _ = _spheres(tmp_path)
    lg = _level(path, mode="r+")
    cc, _, groups, _, width = _two_fragment_chunk(lg)
    with lg.batched_writes():
        write_chunk_links(lg, cc, [np.concatenate(groups)], delta=0,
                          link_width=width)
        with pytest.raises(StoreError, match="batched_writes"):
            stamp_fragment_link_groups(lg)
        with pytest.raises(StoreError, match="batched_writes"):
            stamp_fragments_tile(lg, 3)
    assert not _flag(path)
    assert _claim_errors(path) == []


def test_g7_the_link_width_one_intra_array_is_the_intra_array():
    assert _is_intra_links_array("links/0/self")
    assert _is_intra_links_array("links/0/0.0.0_0.0.0")
    assert not _is_intra_links_array("links/0/0.0.+1")
    assert not _is_intra_links_array("links/+1/self")


def test_g7_resharding_keeps_the_claims_it_does_not_change(tmp_path):
    path, _, _ = _spheres(tmp_path)
    assert _claims(path) == (True, True)
    shard_store(path, shard_shape=2)
    assert _claims(path) == (True, True)
    assert verify_fragment_link_groups(_level(path)) is None
    unshard_store(path)
    assert _claims(path) == (True, True)
    assert _claim_errors(path) == []
    # A write after the repack still withdraws them.
    lg = _level(path, mode="r+")
    cc, _, groups, _, width = _two_fragment_chunk(lg)
    write_chunk_links(lg, cc, groups, delta=0, link_width=width)
    assert not _flag(path)


def test_g7_an_interrupted_reshard_leaves_the_claims_withdrawn(tmp_path, monkeypatch):
    path, _, _ = _spheres(tmp_path)
    real = Group.create_sharded_chunk_array
    calls = {"n": 0}

    def flaky(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("interrupted")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Group, "create_sharded_chunk_array", flaky)
    with pytest.raises(RuntimeError, match="interrupted"):
        shard_store(path, shard_shape=2)
    assert _claims(path) == (False, False)
