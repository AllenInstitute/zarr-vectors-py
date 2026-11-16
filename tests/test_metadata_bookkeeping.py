"""Store creation and metadata bookkeeping regressions."""

from __future__ import annotations

import json

import numpy as np
import pytest

from zarr_vectors.exceptions import ArrayError, DracoError


# --- pre-built Zarr stores keep the writer's create kwargs --------------

def test_prebuilt_store_gets_writer_bounds():
    from zarr.storage import MemoryStore

    from zarr_vectors.core.store import open_store, read_root_metadata
    from zarr_vectors.types.points import read_points, write_points

    pts = np.random.default_rng(0).random((50, 3)).astype("f4") * 500
    store = MemoryStore()
    write_points(store, pts, chunk_shape=(100.0, 100.0, 100.0))
    meta = read_root_metadata(open_store(store))
    assert meta.chunk_shape == (100.0, 100.0, 100.0)
    lo, hi = meta.bounds
    assert np.all(np.asarray(lo) <= pts.min(0))
    assert np.all(np.asarray(hi) >= pts.max(0))
    assert read_points(store)["positions"].shape == (50, 3)


# --- Draco: honest encoding stamp, no half-written store -----------------

def _vertices_attrs(path):
    with open(path / "0" / "vertices" / "zarr.json") as fh:
        return json.load(fh)["attributes"]


def test_draco_on_2d_mesh_is_stamped_raw(tmp_path):
    from zarr_vectors.types.meshes import read_mesh, write_mesh

    path = tmp_path / "m2d.zv"
    v = np.random.default_rng(1).random((20, 2)).astype("f4") * 100
    f = np.array([[0, 1, 2], [3, 4, 5]])
    write_mesh(path, v, f, chunk_shape=(50.0, 50.0), encoding="draco")
    assert _vertices_attrs(path)["encoding"] == "raw"
    out = read_mesh(path, object_ids=[0])  # refused on a Draco store
    assert out["vertices"].shape == (20, 2)


def test_failed_draco_write_creates_nothing(tmp_path, monkeypatch):
    import zarr_vectors.encoding.draco as draco
    from zarr_vectors.types.meshes import write_mesh

    monkeypatch.setattr(draco, "draco_backend_available", lambda: False)
    path = tmp_path / "d.zv"
    v = np.random.default_rng(2).random((6, 3)).astype("f4") * 10
    with pytest.raises(DracoError):
        write_mesh(path, v, np.array([[0, 1, 2]]), chunk_shape=(50.0,) * 3,
                   encoding="draco")
    assert not path.exists()


def test_draco_quads_rejected_before_creation(tmp_path):
    from zarr_vectors.types.meshes import write_mesh

    path = tmp_path / "q.zv"
    v = np.random.default_rng(3).random((8, 3)).astype("f4") * 10
    with pytest.raises(ArrayError, match="triangles"):
        write_mesh(path, v, np.array([[0, 1, 2, 3]]), chunk_shape=(50.0,) * 3,
                   encoding="draco")
    assert not path.exists()


# --- iter_link_cells reads the stored has_perm stamp ---------------------

def test_iter_link_cells_trusts_stored_has_perm(tmp_path, monkeypatch):
    import zarr_vectors.core.arrays as arrays
    from zarr_vectors.core.store import get_resolution_level, open_store
    from zarr_vectors.types.graphs import write_graph

    rng = np.random.default_rng(4)
    pos = rng.random((40, 3)).astype("f4") * 100
    edges = rng.integers(0, 40, size=(60, 2))
    edges = edges[edges[:, 0] != edges[:, 1]]
    path = tmp_path / "g.zv"
    write_graph(path, pos, edges, chunk_shape=(30.0, 30.0, 30.0))
    level = get_resolution_level(open_store(path), 0)

    def widths():
        return {
            (seg, cc): tuple(np.asarray(g).shape[1] for g in groups)
            for seg, _off, cc, groups in arrays.iter_link_cells(level, 0)
        }

    before = widths()
    assert any(w and w[0] == 3 for w in before.values())  # perm column present
    expected = len(arrays.read_links(level))
    # A policy that disagrees with the stamps must not change the decode.
    monkeypatch.setattr(arrays, "links_has_perm", lambda *a, **k: False)
    assert widths() == before
    assert len(arrays.read_links(level)) == expected


# --- object_row_count is object_count ------------------------------------

def test_object_row_count_matches_object_count(tmp_path):
    from zarr_vectors.core.arrays import object_count, object_row_count
    from zarr_vectors.core.store import get_resolution_level, open_store
    from zarr_vectors.types.points import write_points

    path = tmp_path / "p.zv"
    pts = np.random.default_rng(5).random((30, 3)).astype("f4") * 100
    write_points(path, pts, chunk_shape=(50.0,) * 3,
                 object_ids=np.repeat(np.arange(6), 5))
    level = get_resolution_level(open_store(path), 0)
    assert object_row_count(level) == object_count(level) == 6


# --- arrays_present lists every family a writer puts on disk -------------

def _writers():
    import zarr_vectors.types.graphs as G
    import zarr_vectors.types.lines as L
    import zarr_vectors.types.meshes as M
    import zarr_vectors.types.points as PT
    import zarr_vectors.types.polylines as P

    rng = np.random.default_rng(6)
    pos = rng.random((40, 3)).astype("f4") * 100
    va = {"w": rng.random(40).astype("f4")}
    ends = rng.random((10, 2, 3)) * 100
    pl = [rng.random((8, 3)).astype("f4") * 100 for _ in range(5)]
    edges = np.array([[i, i + 1] for i in range(39)])
    return {
        "points": lambda p, c: PT.write_points(
            p, pos, chunk_shape=c, attributes=va,
            object_ids=np.repeat(np.arange(8), 5),
            object_attributes={"s": np.ones(8, "f4")},
            groups={0: [0, 1]}, group_attributes={"g": np.ones(1, "f4")}),
        "lines": lambda p, c: L.write_lines(
            p, ends, chunk_shape=c,
            vertex_attributes={"a": rng.random((10, 2)).astype("f4")}),
        "polyline": lambda p, c: P.write_polylines(
            p, pl, chunk_shape=c, geometry_type="polyline"),
        "streamline": lambda p, c: P.write_polylines(
            p, pl, chunk_shape=c, geometry_type="streamline",
            vertex_attributes={"a": [np.ones(8, "f4")] * 5}),
        "mesh": lambda p, c: M.write_mesh(
            p, pos, np.array([[0, 1, 2], [3, 4, 5]]), chunk_shape=c,
            vertex_attributes=va),
        "graph": lambda p, c: G.write_graph(
            p, pos, edges, chunk_shape=c, vertex_attributes=va,
            link_attributes={"w": np.ones(39, "f4")}),
    }


@pytest.mark.parametrize("chunk", [30.0, 1000.0])
@pytest.mark.parametrize("kind", ["points", "lines", "polyline", "streamline",
                                  "mesh", "graph"])
def test_arrays_present_matches_disk(tmp_path, kind, chunk):
    from zarr_vectors.building import refresh_arrays_present
    from zarr_vectors.core.store import (
        get_resolution_level,
        open_store,
        read_level_metadata,
    )

    path = tmp_path / f"{kind}.zv"
    _writers()[kind](path, (chunk,) * 3)
    root = open_store(path, mode="r+")
    declared = sorted(read_level_metadata(root, 0).arrays_present)
    assert declared == refresh_arrays_present(get_resolution_level(root, 0))
    if kind in ("lines", "polyline", "streamline") and chunk == 30.0:
        assert "links" in declared


# --- RootMetadata.to_dict and _write_root_attrs agree on key presence ----

def test_root_to_dict_key_presence_matches_writer(tmp_path):
    from zarr_vectors.core.store import create_store, read_root_metadata

    root = create_store(tmp_path / "r.zv", bounds=([0, 0, 0], [10, 10, 10]),
                        chunk_shape=(5.0, 5.0, 5.0))
    stored = root.attrs.to_dict()["zarr_vectors"]
    wire = read_root_metadata(root).to_dict()["zarr_vectors"]
    always = {"zv_version", "format_capabilities", "chunk_shape", "bounds",
              "geometry_types"}
    assert always <= set(stored) and always <= set(wire)
    # Everything the writer stores, to_dict emits too; to_dict's extras
    # are only the policy fields a store leaves to their defaults.
    assert set(stored) <= set(wire)
    assert set(wire) - set(stored) <= {
        "links_convention", "object_index_convention", "cross_chunk_strategy",
        "cross_level_depth", "cross_level_storage", "reduction_factor",
    }
    assert wire["format_capabilities"] == stored["format_capabilities"] == []
    assert "crs" not in stored and "crs" not in wire


# --- point clouds carry no links_convention ------------------------------

def test_point_cloud_is_not_stamped_with_links_convention(tmp_path):
    from zarr_vectors.core.store import open_store, read_root_metadata
    from zarr_vectors.types.points import write_points

    path = tmp_path / "p.zv"
    write_points(path, np.random.default_rng(7).random((20, 3)) * 10,
                 chunk_shape=(5.0, 5.0, 5.0))
    root = open_store(path)
    assert "links_convention" not in root.attrs.to_dict()["zarr_vectors"]
    # Readers still see the documented default.
    assert read_root_metadata(root).links_convention == "implicit_sequential"


# --- zv_array discriminators ---------------------------------------------

def test_parametric_arrays_carry_zv_array(tmp_path):
    from zarr_vectors.core.arrays import read_zv_array_tag
    from zarr_vectors.core.store import get_parametric_group, open_store
    from zarr_vectors.types.parametric import (
        read_parametric_objects,
        write_parametric_objects,
    )

    path = str(tmp_path / "par.zv")
    write_parametric_objects(
        path,
        [{"type": "plane", "coefficients": [0, 0, 1, -5]},
         {"type": "sphere", "coefficients": [1, 2, 3, 4]}],
        object_attributes={"w": np.array([1.0, 2.0])},
        groups={0: [0, 1]},
        group_attributes={"g": np.array([7.0])},
        create_new_store=True,
        store_kwargs={"bounds": ([0, 0, 0], [10, 10, 10]),
                      "chunk_shape": (5.0, 5.0, 5.0)},
    )
    para = get_parametric_group(open_store(path))
    tags = {
        name: read_zv_array_tag(para.read_array_meta(name) or {})
        for name in ("objects", "names", "object_attributes/w", "groups",
                     "group_attributes/g")
    }
    assert tags == {
        "objects": "parametric_objects",
        "names": "parametric_names",
        "object_attributes/w": "object_attribute",
        "groups": "groups",
        "group_attributes/g": "groupings_attribute",
    }
    assert len(read_parametric_objects(path)) == 2
