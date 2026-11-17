"""Tests for write-time attribute chunking (``chunk_by_attribute``).

Covers the five geometry writers (points, polylines, lines, graphs,
meshes), the per-vertex split rule (points/polylines) vs per-object
uniformity rule (lines/graphs/meshes), metadata round-trips, validation
errors, and the ``attribute_filter`` read fast path.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._deprecated import open_zv
from zarr_vectors.core.attr_chunking import assign_attribute_bins
from zarr_vectors.core.store import open_store, read_level_metadata
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.types.graphs import read_graph, write_graph
from zarr_vectors.types.lines import read_lines, write_lines
from zarr_vectors.types.meshes import read_mesh, write_mesh
from zarr_vectors.types.points import read_points, write_points
from zarr_vectors.types.polylines import read_polylines, write_polylines

# ===================================================================
# assign_attribute_bins unit tests
# ===================================================================


def test_assign_attribute_bins_strings():
    vals = np.array(["B", "A", "A", "C", "B"])
    bins, mapping = assign_attribute_bins(vals)
    # np.unique sorts: A, B, C
    assert mapping == ["A", "B", "C"]
    assert bins.tolist() == [1, 0, 0, 2, 1]


def test_assign_attribute_bins_ints():
    vals = np.array([5, 1, 1, 9, 5], dtype=np.int32)
    bins, mapping = assign_attribute_bins(vals)
    assert mapping == [1, 5, 9]
    assert bins.tolist() == [1, 0, 0, 2, 1]


def test_assign_attribute_bins_rejects_floats():
    with pytest.raises(ArrayError, match="categorical-only"):
        assign_attribute_bins(np.array([1.0, 2.0, 3.0]))


def test_assign_attribute_bins_rejects_2d():
    with pytest.raises(ArrayError, match="1D"):
        assign_attribute_bins(np.array([[1, 2], [3, 4]]))


# ===================================================================
# write_points + chunk_by_attribute
# ===================================================================


def _make_two_gene_cloud(seed: int = 0, n: int = 300):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 100, (n, 3)).astype("f4")
    gene = np.array(["A"] * (n * 3 // 5) + ["B"] * (n - n * 3 // 5))
    rng.shuffle(gene)
    return pos, gene


def test_points_attr_chunking_round_trip(tmp_path):
    pos, gene = _make_two_gene_cloud(seed=0, n=300)
    store = tmp_path / "g.zv"

    write_points(
        str(store), pos,
        chunk_shape=(50.0, 50.0, 50.0),
        vertex_attributes={"gene": gene},
        chunk_by_attribute="gene",
    )

    # Level metadata is correctly populated
    root = open_store(str(store))
    lm = read_level_metadata(root, 0)
    assert lm.chunk_dims == ["gene", "x", "y", "z"]
    assert lm.chunk_attribute_name == "gene"
    assert lm.chunk_attribute_values == ["A", "B"]

    # Chunk keys all start with a bin index 0 or 1
    keys = root["0"].list_chunks("vertices")
    assert keys, "no chunks written"
    for k in keys:
        parts = k.split(".")
        assert len(parts) == 4, f"expected 4D chunk key, got {k}"
        assert parts[0] in ("0", "1")


def test_points_attr_chunking_full_read_returns_all(tmp_path):
    pos, gene = _make_two_gene_cloud(seed=1, n=400)
    store = tmp_path / "g.zv"

    write_points(
        str(store), pos,
        chunk_shape=(50.0, 50.0, 50.0),
        vertex_attributes={"gene": gene},
        chunk_by_attribute="gene",
    )
    out = read_points(str(store))
    assert out["vertex_count"] == 400


def test_points_attr_filter_selectivity(tmp_path):
    pos, gene = _make_two_gene_cloud(seed=2, n=600)
    store = tmp_path / "g.zv"

    write_points(
        str(store), pos,
        chunk_shape=(50.0, 50.0, 50.0),
        vertex_attributes={"gene": gene},
        chunk_by_attribute="gene",
    )

    out_a = read_points(str(store), attribute_filter={"gene": "A"})
    out_b = read_points(str(store), attribute_filter={"gene": "B"})

    assert out_a["vertex_count"] == int((gene == "A").sum())
    assert out_b["vertex_count"] == int((gene == "B").sum())
    assert out_a["vertex_count"] + out_b["vertex_count"] == len(pos)


def test_points_attr_filter_unknown_value_returns_empty(tmp_path):
    pos, gene = _make_two_gene_cloud(seed=3, n=200)
    store = tmp_path / "g.zv"
    write_points(
        str(store), pos,
        chunk_shape=(50.0, 50.0, 50.0),
        vertex_attributes={"gene": gene},
        chunk_by_attribute="gene",
    )
    out = read_points(str(store), attribute_filter={"gene": "ZZZ"})
    assert out["vertex_count"] == 0


def test_points_chunk_by_attribute_missing_attribute_raises(tmp_path):
    pos = np.random.default_rng(0).uniform(0, 10, (50, 3)).astype("f4")
    with pytest.raises(ArrayError, match="must name a key"):
        write_points(
            str(tmp_path / "x.zv"), pos,
            chunk_shape=(10.0, 10.0, 10.0),
            chunk_by_attribute="nonexistent",
        )


def test_points_chunk_by_attribute_rejects_float(tmp_path):
    pos = np.random.default_rng(0).uniform(0, 10, (50, 3)).astype("f4")
    with pytest.raises(ArrayError, match="categorical-only"):
        write_points(
            str(tmp_path / "x.zv"), pos,
            chunk_shape=(10.0, 10.0, 10.0),
            vertex_attributes={"score": np.random.default_rng(0).uniform(0, 1, 50)},
            chunk_by_attribute="score",
        )


def test_points_attribute_filter_on_non_attr_store_raises(tmp_path):
    """attribute_filter only makes sense for attribute-chunked stores."""
    pos = np.random.default_rng(0).uniform(0, 10, (50, 3)).astype("f4")
    store = tmp_path / "plain.zv"
    write_points(str(store), pos, chunk_shape=(10.0, 10.0, 10.0))
    with pytest.raises(ArrayError, match="chunk_attribute_name"):
        read_points(str(store), attribute_filter={"gene": "A"})


def test_points_attribute_filter_mismatched_name_raises(tmp_path):
    pos, gene = _make_two_gene_cloud(seed=4, n=100)
    store = tmp_path / "g.zv"
    write_points(
        str(store), pos,
        chunk_shape=(50.0, 50.0, 50.0),
        vertex_attributes={"gene": gene},
        chunk_by_attribute="gene",
    )
    with pytest.raises(ArrayError, match="does not match"):
        read_points(str(store), attribute_filter={"wrong_name": "A"})


# ===================================================================
# Polylines + chunk_by_attribute, including per-vertex split
# ===================================================================


def test_polylines_attr_chunking_splits_mixed_polyline(tmp_path):
    """Polyline 0 has all 'A' vertices; polyline 1 has 5 A then 5 B.

    Filtering by 'A' should yield 10 + 5 vertices; filtering by 'B'
    should yield 5 vertices.  Both polyline IDs should appear in
    filter='A' but only polyline 1 in filter='B'.
    """
    rng = np.random.default_rng(0)
    polys = [
        rng.uniform(0, 100, (10, 3)).astype("f4"),
        rng.uniform(0, 100, (10, 3)).astype("f4"),
    ]
    labels = [
        np.array(["A"] * 10),
        np.array(["A"] * 5 + ["B"] * 5),
    ]
    store = tmp_path / "tr.zv"
    write_polylines(
        str(store), polys,
        chunk_shape=(50.0, 50.0, 50.0),
        bin_shape=(50.0, 50.0, 50.0),
        vertex_attributes={"bundle": labels},
        chunk_by_attribute="bundle",
    )

    # Metadata
    root = open_store(str(store))
    lm = read_level_metadata(root, 0)
    assert lm.chunk_attribute_name == "bundle"
    assert lm.chunk_attribute_values == ["A", "B"]

    # Full read keeps everything
    out_all = read_polylines(str(store))
    assert out_all["vertex_count"] == 20

    # Filter A
    out_a = read_polylines(str(store), attribute_filter={"bundle": "A"})
    assert out_a["vertex_count"] == 15
    # Both polylines have at least some A vertices
    assert out_a["polyline_count"] == 2

    # Filter B
    out_b = read_polylines(str(store), attribute_filter={"bundle": "B"})
    assert out_b["vertex_count"] == 5
    assert out_b["polyline_count"] == 1


def test_polylines_attr_chunking_chunk_keys_are_4d(tmp_path):
    rng = np.random.default_rng(5)
    polys = [rng.uniform(0, 100, (8, 3)).astype("f4") for _ in range(4)]
    labels = [np.array(["X"] * 8) if i % 2 == 0 else np.array(["Y"] * 8) for i in range(4)]
    store = tmp_path / "tr.zv"
    write_polylines(
        str(store), polys,
        chunk_shape=(50.0, 50.0, 50.0),
        bin_shape=(50.0, 50.0, 50.0),
        vertex_attributes={"bundle": labels},
        chunk_by_attribute="bundle",
    )
    root = open_store(str(store))
    files = root["0"].list_chunks("vertices")
    assert files, "no chunks written"
    for f in files:
        assert f.count(".") == 3, f"expected 4-arity key, got {f}"


# ===================================================================
# Lazy reader accessors
# ===================================================================


def test_zv_level_attribute_values(tmp_path):
    pos, gene = _make_two_gene_cloud(seed=6, n=120)
    store = tmp_path / "g.zv"
    write_points(
        str(store), pos,
        chunk_shape=(50.0, 50.0, 50.0),
        vertex_attributes={"gene": gene},
        chunk_by_attribute="gene",
    )
    zv = open_zv(str(store))
    lvl = zv[0]
    assert lvl.chunk_attribute_name == "gene"
    assert lvl.attribute_values == ["A", "B"]
    assert lvl.chunk_dims is not None
    assert lvl.chunk_dims[0] == "gene"


def test_zv_level_read_attribute_chunk(tmp_path):
    pos, gene = _make_two_gene_cloud(seed=7, n=200)
    store = tmp_path / "g.zv"
    write_points(
        str(store), pos,
        chunk_shape=(50.0, 50.0, 50.0),
        vertex_attributes={"gene": gene},
        chunk_by_attribute="gene",
    )
    zv = open_zv(str(store))
    a_groups = zv[0].read_attribute_chunk("A")
    total_a_verts = sum(len(g) for g in a_groups)
    assert total_a_verts == int((gene == "A").sum())


def test_zv_level_read_attribute_chunk_unknown_value(tmp_path):
    pos, gene = _make_two_gene_cloud(seed=8, n=80)
    store = tmp_path / "g.zv"
    write_points(
        str(store), pos,
        chunk_shape=(50.0, 50.0, 50.0),
        vertex_attributes={"gene": gene},
        chunk_by_attribute="gene",
    )
    zv = open_zv(str(store))
    assert zv[0].read_attribute_chunk("missing") == []


# ===================================================================
# Lines + chunk_by_attribute (per-line, in line_attributes)
# ===================================================================


def _make_categorised_lines(seed: int = 0, n: int = 12):
    rng = np.random.default_rng(seed)
    eps = np.stack([
        rng.uniform(0, 100, (n, 3)),
        rng.uniform(0, 100, (n, 3)),
    ], axis=1).astype("f4")
    cat = np.array(["X"] * (n // 2) + ["Y"] * (n - n // 2))
    rng.shuffle(cat)
    return eps, cat


def test_lines_attr_chunking_line_attribute_round_trip(tmp_path):
    eps, cat = _make_categorised_lines(seed=0, n=16)
    store = tmp_path / "lines.zv"
    write_lines(
        str(store), eps,
        chunk_shape=(50.0, 50.0, 50.0),
        bin_shape=(50.0, 50.0, 50.0),
        object_attributes={"cat": cat},
        chunk_by_attribute="cat",
    )
    lm = read_level_metadata(open_store(str(store)), 0)
    assert lm.chunk_attribute_name == "cat"
    assert sorted(lm.chunk_attribute_values) == ["X", "Y"]

    n_x = int((cat == "X").sum())
    n_y = int((cat == "Y").sum())
    assert read_lines(str(store))["line_count"] == n_x + n_y
    assert read_lines(str(store), attribute_filter={"cat": "X"})["line_count"] == n_x
    assert read_lines(str(store), attribute_filter={"cat": "Y"})["line_count"] == n_y


def test_lines_per_endpoint_attribute_must_match_per_line(tmp_path):
    """If the chunk-by attribute is in per-endpoint `attributes`, both
    endpoints of every line must share the same value."""
    eps, _ = _make_categorised_lines(seed=1, n=8)
    # Endpoints disagree → expect error.
    bad = np.array([["A", "B"]] * 8)
    with pytest.raises(ArrayError, match="endpoints"):
        write_lines(
            str(tmp_path / "x.zv"), eps,
            chunk_shape=(50.0, 50.0, 50.0),
            vertex_attributes={"cat": bad},
            chunk_by_attribute="cat",
        )


def test_lines_chunk_by_attribute_missing_raises(tmp_path):
    eps, _ = _make_categorised_lines(seed=2, n=6)
    with pytest.raises(ArrayError, match="must name a key"):
        write_lines(
            str(tmp_path / "x.zv"), eps,
            chunk_shape=(50.0, 50.0, 50.0),
            chunk_by_attribute="nonexistent",
        )


# ===================================================================
# Graphs + chunk_by_attribute (per-object uniformity)
# ===================================================================


def _make_categorised_graph(seed: int = 0, n_objs: int = 4, per_obj: int = 5):
    rng = np.random.default_rng(seed)
    n = n_objs * per_obj
    pos = rng.uniform(0, 100, (n, 3)).astype("f4")
    obj_ids = np.repeat(np.arange(n_objs, dtype=np.int64), per_obj)
    # First object is type "I", rest are "E".
    cell_type = np.where(obj_ids == 0, "I", "E")
    edges = np.array(
        [[i, i + 1] for i in range(n - 1) if obj_ids[i] == obj_ids[i + 1]],
        dtype=np.int64,
    )
    return pos, edges, obj_ids, cell_type


def test_graph_attr_chunking_round_trip(tmp_path):
    pos, edges, obj_ids, cell_type = _make_categorised_graph(seed=0)
    store = tmp_path / "graph.zv"
    write_graph(
        str(store), pos, edges,
        chunk_shape=(50.0, 50.0, 50.0),
        object_ids=obj_ids,
        vertex_attributes={"cell_type": cell_type},
        chunk_by_attribute="cell_type",
    )
    lm = read_level_metadata(open_store(str(store)), 0)
    assert lm.chunk_attribute_name == "cell_type"
    assert sorted(lm.chunk_attribute_values) == ["E", "I"]

    n_i = int((cell_type == "I").sum())
    n_e = int((cell_type == "E").sum())
    assert read_graph(str(store))["node_count"] == n_i + n_e
    assert read_graph(str(store), attribute_filter={"cell_type": "I"})["node_count"] == n_i
    assert read_graph(str(store), attribute_filter={"cell_type": "E"})["node_count"] == n_e


def test_graph_attr_chunking_rejects_mixed_object(tmp_path):
    """Per-object uniformity: all nodes of one object must share the value."""
    pos, edges, obj_ids, cell_type = _make_categorised_graph(seed=1)
    # Break uniformity: flip one node's type within object 0.
    cell_type = cell_type.copy()
    cell_type[1] = "E"  # object 0 now has nodes "I", "E", "I", "I", "I"
    with pytest.raises(ArrayError, match="per-object uniformity"):
        write_graph(
            str(tmp_path / "x.zv"), pos, edges,
            chunk_shape=(50.0, 50.0, 50.0),
            object_ids=obj_ids,
            vertex_attributes={"cell_type": cell_type},
            chunk_by_attribute="cell_type",
        )


def test_graph_attr_chunking_default_object_per_node(tmp_path):
    """With chunk_by_attribute and no object_ids, each node becomes its
    own object (so uniformity is trivially satisfied)."""
    rng = np.random.default_rng(0)
    n = 8
    pos = rng.uniform(0, 100, (n, 3)).astype("f4")
    edges = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int64)
    cell_type = np.array(["A"] * 4 + ["B"] * 4)
    store = tmp_path / "g.zv"
    write_graph(
        str(store), pos, edges,
        chunk_shape=(50.0, 50.0, 50.0),
        vertex_attributes={"cell_type": cell_type},
        chunk_by_attribute="cell_type",
    )
    assert read_graph(str(store), attribute_filter={"cell_type": "A"})["node_count"] == 4


# ===================================================================
# Meshes + chunk_by_attribute (per-object uniformity)
# ===================================================================


def test_mesh_attr_chunking_round_trip(tmp_path):
    # Two tetrahedra in non-overlapping regions, labelled differently.
    v0 = np.array([[10, 10, 10], [20, 10, 10], [15, 20, 10], [15, 15, 20]], dtype="f4")
    v1 = np.array([[60, 60, 60], [70, 60, 60], [65, 70, 60], [65, 65, 70]], dtype="f4")
    verts = np.concatenate([v0, v1], axis=0)
    faces = np.array(
        [[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3],
         [4, 5, 6], [4, 5, 7], [5, 6, 7], [4, 6, 7]],
        dtype=np.int64,
    )
    obj_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    tissue = np.array(["cortex"] * 4 + ["stem"] * 4)

    store = tmp_path / "m.zv"
    write_mesh(
        str(store), verts, faces,
        chunk_shape=(50.0, 50.0, 50.0),
        object_ids=obj_ids,
        vertex_attributes={"tissue": tissue},
        chunk_by_attribute="tissue",
    )
    lm = read_level_metadata(open_store(str(store)), 0)
    assert lm.chunk_attribute_name == "tissue"
    assert sorted(lm.chunk_attribute_values) == ["cortex", "stem"]
    assert read_mesh(str(store), attribute_filter={"tissue": "cortex"})["vertex_count"] == 4
    assert read_mesh(str(store), attribute_filter={"tissue": "stem"})["vertex_count"] == 4


def test_mesh_attr_chunking_rejects_mixed_object(tmp_path):
    verts = np.array(
        [[10, 10, 10], [20, 10, 10], [15, 20, 10], [15, 15, 20]],
        dtype="f4",
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]], dtype=np.int64)
    obj_ids = np.zeros(4, dtype=np.int64)
    # One mesh object with mixed tissue labels → uniformity check fires.
    tissue = np.array(["cortex", "stem", "cortex", "cortex"])
    with pytest.raises(ArrayError, match="per-object uniformity"):
        write_mesh(
            str(tmp_path / "x.zv"), verts, faces,
            chunk_shape=(50.0, 50.0, 50.0),
            object_ids=obj_ids,
            vertex_attributes={"tissue": tissue},
            chunk_by_attribute="tissue",
        )
