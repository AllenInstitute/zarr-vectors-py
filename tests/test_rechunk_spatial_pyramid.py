"""``rechunk_spatial`` on pyramids: shared fragments, bins and cross-level links.

Coarse levels share fragments between objects (a metavertex sits in the
manifest of every object it aggregates).  The old reader gave each
fragment to its first owner only, so a 20-object level came back with a
handful of objects, and validation still passed.  It also stamped every
coarse level's ``bin_shape`` as ``base * 2**level`` whatever the source
said, and dropped the ``links/±k`` families.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.core.arrays import (
    list_link_deltas,
    read_chunk_attributes,
    read_chunk_vertices,
    read_link_arrays,
    read_object_manifest_rows,
    read_vertex_fragment_index,
)
from zarr_vectors.core.store import (
    get_resolution_level,
    list_resolution_levels,
    open_store,
    read_level_metadata,
)
from zarr_vectors.multiresolution.coarsen import build_pyramid
from zarr_vectors.rechunk import rechunk_spatial
from zarr_vectors.types.polylines import write_polylines
from zarr_vectors.validate import validate


def _buffers(lg, cc):
    """A chunk's vertex buffer, laid out by the fragment index."""
    fi = read_vertex_fragment_index(lg, cc)
    groups = read_chunk_vertices(lg, cc, ndim=3)
    n = sum(len(g) for g in groups)
    buf = np.zeros((n, 3))
    for f, g in enumerate(groups):
        if fi.is_range(f):
            s, c = fi.range(f)
            buf[s:s + c] = g
        else:
            buf[np.asarray(fi.indices(f))] = g
    return groups, buf


def _snapshot(path):
    """Per level: each object's vertex multiset and every link as positions."""
    root = open_store(str(path), mode="r")
    out = {}
    for lvl in list_resolution_levels(root):
        lg = get_resolution_level(root, lvl)
        ids, mans = read_object_manifest_rows(lg)
        objs = {}
        for oid, man in zip(ids.tolist(), mans):
            pts = []
            for cc, f in man:
                groups, _ = _buffers(lg, tuple(cc))
                pts += [tuple(np.round(p, 4)) for p in groups[f]]
            if pts:
                objs[oid] = Counter(pts)
        links = {}
        for delta in list_link_deltas(lg):
            other = get_resolution_level(root, lvl + delta)
            chunks, vi = read_link_arrays(lg, delta=delta)
            recs = []
            for c, v in zip(chunks, vi):
                rec = []
                for k, (ck, vk) in enumerate(zip(c, v)):
                    g = lg if k == 0 else other
                    rec.append(tuple(np.round(_buffers(g, tuple(int(x) for x in ck))[1][vk], 4)))
                # Endpoint order is kept: it is a mesh face's winding.
                recs.append(tuple(rec))
            links[delta] = Counter(recs)
        lm = read_level_metadata(root, lvl)
        out[lvl] = (objs, links, lm.vertex_count, lm.bin_shape)
    return out


@pytest.fixture
def pyramid(tmp_path: Path) -> Path:
    path = tmp_path / "lines.zarr"
    rng = np.random.default_rng(1)
    lines = [np.clip(np.cumsum(rng.normal(0, 3, (40, 3)), 0) + 50, 0.5, 99.5)
             .astype(np.float32) for _ in range(20)]
    write_polylines(
        path, lines, chunk_shape=(25.0,) * 3, bounds=([0.0] * 3, [100.0] * 3),
        vertex_attributes={"radius": [np.arange(len(x), dtype=np.float32)
                                      for x in lines]},
    )
    # Factors of 3 then 2: the old code would have stamped bins of 2x, 4x.
    build_pyramid(path, factors=[(3.0, 1.0), (2.0, 1.0)])
    return path


def test_pyramid_round_trips_exactly(pyramid, tmp_path):
    out = tmp_path / "out.zarr"
    rechunk_spatial(pyramid, out, (50.0,) * 3)
    before, after = _snapshot(pyramid), _snapshot(out)
    assert sorted(before) == sorted(after) == [0, 1, 2]
    for lvl in before:
        objs_b, links_b, vc_b, bin_b = before[lvl]
        objs_a, links_a, vc_a, bin_a = after[lvl]
        assert len(objs_b) == 20
        assert objs_a == objs_b, f"level {lvl} objects differ"
        assert links_a == links_b, f"level {lvl} links differ"
        assert set(links_b) - {0}, "pyramid should carry cross-level links"
        assert vc_a == vc_b
        assert bin_a == bin_b
    assert validate(str(out), level=5).ok


def test_vertex_attributes_follow_their_vertices(pyramid, tmp_path):
    out = tmp_path / "out.zarr"
    rechunk_spatial(pyramid, out, (50.0,) * 3)

    def pairs(path):
        root = open_store(str(path), mode="r")
        lg = get_resolution_level(root, 0)
        got = Counter()
        for cc in lg.list_chunks("vertices"):
            cc = tuple(int(x) for x in cc.split("."))
            groups = read_chunk_vertices(lg, cc, ndim=3)
            attrs = read_chunk_attributes(lg, "radius", cc, dtype="float32", ncols=1)
            for g, a in zip(groups, attrs):
                got.update((tuple(np.round(p, 4)), float(r))
                           for p, r in zip(g, np.ravel(a)))
        return got

    assert pairs(out) == pairs(pyramid)


def test_refusal_leaves_an_existing_output_alone(pyramid, tmp_path):
    from zarr_vectors.core.arrays import (
        create_link_attributes_array,
    )

    root = open_store(str(pyramid), mode="r+")
    create_link_attributes_array(get_resolution_level(root, 0), "w",
                                 dtype="float32", link_width=2, sid_ndim=3)
    out = tmp_path / "out.zarr"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(NotImplementedError, match="link_attributes"):
        rechunk_spatial(pyramid, out, (50.0,) * 3, force=True)
    assert (out / "keep.txt").exists()


def test_mesh_faces_keep_their_winding(tmp_path):
    from zarr_vectors.types.meshes import write_mesh

    rng = np.random.default_rng(3)
    verts = rng.uniform(1.0, 99.0, size=(300, 3)).astype(np.float32)
    faces = np.stack([rng.permutation(300)[:3] for _ in range(400)])
    src = tmp_path / "mesh.zarr"
    write_mesh(src, verts, faces, chunk_shape=(25.0,) * 3,
               bounds=([0.0] * 3, [100.0] * 3))
    out = tmp_path / "out.zarr"
    rechunk_spatial(src, out, (50.0,) * 3)
    before, after = _snapshot(src), _snapshot(out)
    assert before[0][1][0] == after[0][1][0]
    assert sum(before[0][1][0].values()) == 400
    assert before[0][0] == after[0][0]
