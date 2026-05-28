"""Integration tests for lazy API, headers, sharding, rechunking, and composite stores.

Each test exercises a multi-feature pipeline end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _make_streamlines(rng, n=100, ndim=3):
    """Generate synthetic streamlines."""
    polys = []
    for i in range(n):
        nn = rng.integers(15, 50)
        start = rng.uniform(10, 290, size=(1, ndim)).astype(np.float32)
        steps = rng.normal(0, 2.5, size=(nn - 1, ndim)).astype(np.float32)
        polys.append(
            np.clip(np.concatenate([start, start + np.cumsum(steps, axis=0)]),
                    0, 399).astype(np.float32)
        )
    return polys


class TestLazyFilterChain:
    """Lazy API: open → filter by group → filter by bbox → compute."""

    def test_lazy_filter_pipeline(self, tmp_path: Path) -> None:
        from zarr_vectors.types.polylines import write_polylines
        from zarr_vectors.lazy import open_zv

        rng = np.random.default_rng(42)
        polys = _make_streamlines(rng, 100)
        store = str(tmp_path / "tracts.zv")
        write_polylines(
            store, polys, chunk_shape=(200., 200., 200.),
            groups={0: list(range(50)), 1: list(range(50, 100))},
        )

        zv = open_zv(store)
        assert zv[0].vertex_count > 0

        # Chain: group → bbox
        view = zv[0].filter(group_ids=[0]).filter(
            bbox=(np.array([0, 0, 0]), np.array([200, 200, 200])),
        )
        result = view.compute()
        assert result["vertex_count"] > 0
        assert result["vertex_count"] < zv[0].vertex_count

        # Polyline access
        poly_5 = zv[0].polylines[5].compute()
        assert poly_5.ndim == 2 and poly_5.shape[1] == 3

        # Compute all polylines in parallel
        all_polys = zv[0].polylines.compute()
        assert len(all_polys) == 100


class TestLazyDaskParallel:
    """Lazy API with explicit dask.compute parallelism."""

    def test_dask_compute_chunks(self, tmp_path: Path) -> None:
        import dask
        from zarr_vectors.types.points import write_points
        from zarr_vectors.lazy import open_zv

        rng = np.random.default_rng(42)
        store = str(tmp_path / "pts.zv")
        positions = rng.uniform(0, 400, size=(5000, 3)).astype(np.float32)
        write_points(store, positions, chunk_shape=(100., 100., 100.))

        zv = open_zv(store)
        delayed_chunks = zv[0].vertices.to_delayed()
        assert len(delayed_chunks) > 1

        # Custom per-chunk processing
        delayed_means = [
            dask.delayed(lambda c: c.mean(axis=0))(ch)
            for ch in delayed_chunks
        ]
        results = dask.compute(*delayed_means)
        assert all(r.shape == (3,) for r in results)


class TestShardReshardChain:
    """Shard → reshard → unshard round-trip with data integrity.

    Asserts the resulting layout uses Zarr v3's native
    ``sharding_indexed`` codec — the on-disk format is readable by
    any standards-compliant Zarr v3 reader, not just zarr-vectors-py.
    """

    def test_shard_chain(self, tmp_path: Path) -> None:
        import json
        import zarr
        from zarr_vectors.types.points import write_points, read_points
        from zarr_vectors.sharding import reshard, is_sharded, get_shard_info
        from zarr_vectors.validate import validate

        rng = np.random.default_rng(42)
        store = str(tmp_path / "pts.zv")
        positions = rng.uniform(0, 400, size=(2000, 3)).astype(np.float32)
        write_points(store, positions, chunk_shape=(100., 100., 100.))
        r_before = read_points(store)

        # flat → sharded (8x8x8 = 512 chunks per shard)
        reshard(store, 8)
        assert is_sharded(store)
        info = get_shard_info(store)
        assert info["sharded"] and info["arrays"], info
        # Every sharded array uses the native codec.
        zg = zarr.open_group(store)
        vertices_arr = zg["0/vertices"]
        assert isinstance(vertices_arr, zarr.Array)
        codec_names = {
            c.to_dict().get("name") for c in vertices_arr.metadata.codecs
        }
        assert "sharding_indexed" in codec_names, codec_names

        # Reshard to a different shape — still native.
        reshard(store, 4)
        info4 = get_shard_info(store)
        assert info4["sharded"]
        for entry in info4["arrays"]:
            assert all(s == 4 for s in entry["shard_shape"])

        # sharded → flat
        reshard(store, None)
        assert not is_sharded(store)

        # Data survives
        r_after = read_points(store)
        assert r_after["vertex_count"] == 2000
        np.testing.assert_allclose(
            np.sort(r_before["positions"], axis=0),
            np.sort(r_after["positions"], axis=0),
            atol=1e-5,
        )

        # Validates after full chain
        assert validate(store, level=4).ok


class TestBornShardedWrites:
    """Type writers can produce native-sharded stores directly via
    ``shard_shape=`` — no post-hoc ``shard_store`` conversion needed.
    """

    def test_points_born_sharded(self, tmp_path: Path) -> None:
        import zarr
        from zarr_vectors.types.points import write_points, read_points

        rng = np.random.default_rng(0)
        store = str(tmp_path / "pts.zv")
        positions = rng.uniform(0, 400, (500, 3)).astype(np.float32)
        write_points(
            store, positions,
            chunk_shape=(100., 100., 100.),
            shard_shape=2,
        )

        # vertices and vertex_fragments are sharded Zarr arrays at
        # creation time — no migration step needed.
        zg = zarr.open_group(store)
        for name in ("0/vertices", "0/vertex_fragments"):
            node = zg[name]
            assert isinstance(node, zarr.Array), (name, type(node))
            codec_names = {
                c.to_dict().get("name") for c in node.metadata.codecs
            }
            assert "sharding_indexed" in codec_names, (name, codec_names)
            assert node.shards == (2, 2, 2)

        r = read_points(store)
        assert r["vertex_count"] == 500
        np.testing.assert_allclose(
            np.sort(r["positions"], 0), np.sort(positions, 0), atol=1e-5,
        )

    def test_graph_born_sharded(self, tmp_path: Path) -> None:
        import zarr
        from zarr_vectors.types.graphs import write_graph, read_graph

        store = str(tmp_path / "g.zv")
        # Three clusters of nodes in three distinct chunks, with edges
        # inside each cluster (guarantees intra-chunk edges so the
        # ``link_fragments`` array is materialised) plus one cross-chunk
        # edge.
        nodes = np.array([
            [10., 10., 10.], [20., 20., 20.], [30., 30., 30.],   # chunk (0,0,0)
            [110., 110., 110.], [120., 120., 120.],              # chunk (1,1,1)
            [210., 210., 210.], [220., 220., 220.],              # chunk (2,2,2)
        ], dtype=np.float32)
        edges = np.array([
            [0, 1], [1, 2],   # intra (0,0,0)
            [3, 4],           # intra (1,1,1)
            [5, 6],           # intra (2,2,2)
            [2, 3],           # cross (0,0,0) -> (1,1,1)
        ], dtype=np.int64)
        write_graph(
            store, nodes, edges,
            chunk_shape=(100., 100., 100.),
            shard_shape=(2, 2, 2),
        )

        zg = zarr.open_group(store)
        for name in ("0/vertices", "0/vertex_fragments",
                     "0/links/0", "0/link_fragments"):
            node = zg[name]
            assert isinstance(node, zarr.Array), (name, type(node))

        r = read_graph(store)
        assert r["node_count"] == 7


class TestShardedPyramid:
    """Build pyramid then shard — all levels survive."""

    def test_pyramid_then_shard(self, tmp_path: Path) -> None:
        from zarr_vectors.types.points import write_points, read_points
        from zarr_vectors.multiresolution.coarsen import build_pyramid
        from zarr_vectors.sharding import reshard, is_sharded
        from zarr_vectors.core.store import open_store, list_resolution_levels
        from zarr_vectors.validate import validate

        rng = np.random.default_rng(42)
        store = str(tmp_path / "pyr.zv")
        write_points(
            store,
            rng.uniform(0, 1000, size=(10000, 3)).astype(np.float32),
            chunk_shape=(100., 100., 100.),
        )
        build_pyramid(store, factors=[(2.0, 1.0), (2.0, 1.0)])
        levels_before = list_resolution_levels(open_store(store))

        reshard(store, 8)
        assert is_sharded(store)

        reshard(store, None)
        levels_after = list_resolution_levels(open_store(store))
        assert levels_after == levels_before

        for lvl in levels_after:
            r = read_points(store, level=lvl)
            assert r["vertex_count"] > 0

        assert validate(store, level=5).ok


class TestRechunkByGroup:
    """Rechunk by group → prefix-scan reads."""

    def test_group_rechunk(self, tmp_path: Path) -> None:
        from zarr_vectors.types.polylines import write_polylines
        from zarr_vectors.rechunk import rechunk, RechunkSpec
        from zarr_vectors.core.store import open_store
        from zarr_vectors.core.arrays import list_chunk_keys, read_chunk_vertices

        rng = np.random.default_rng(42)
        polys = _make_streamlines(rng, 60)
        store = str(tmp_path / "tracts.zv")
        write_polylines(
            store, polys, chunk_shape=(200., 200., 200.),
            groups={0: list(range(30)), 1: list(range(30, 60))},
        )

        out = str(tmp_path / "grouped.zv")
        result = rechunk(store, RechunkSpec(by="group"), output=out)
        assert result["bins_created"] == 2
        assert result["objects_rechunked"] == 60

        # Verify 4D chunk keys
        keys = list_chunk_keys(open_store(out)["0"])
        assert all(len(k) == 4 for k in keys)

        # Group 0 prefix scan
        g0_keys = [k for k in keys if k[0] == 0]
        g0_verts = 0
        for ck in g0_keys:
            groups = read_chunk_vertices(
                open_store(out)["0"], ck,
                dtype=np.float32, ndim=3,
            )
            g0_verts += sum(len(g) for g in groups)
        assert g0_verts > 0


class TestRechunkByAttribute:
    """Rechunk by attribute:length with explicit bins."""

    def test_length_rechunk(self, tmp_path: Path) -> None:
        from zarr_vectors.types.polylines import write_polylines
        from zarr_vectors.rechunk import rechunk, RechunkSpec
        from zarr_vectors.core.store import open_store
        from zarr_vectors.core.arrays import list_chunk_keys

        rng = np.random.default_rng(42)
        polys = _make_streamlines(rng, 80)
        store = str(tmp_path / "tracts.zv")
        write_polylines(store, polys, chunk_shape=(200., 200., 200.))

        out = str(tmp_path / "by_length.zv")
        result = rechunk(
            store,
            RechunkSpec(by="attribute:length", bins=[0, 30, 80, float("inf")]),
            output=out,
        )
        assert result["bins_created"] >= 2
        assert result["objects_rechunked"] == 80

        # 4D keys with prefix = length bin
        keys = list_chunk_keys(open_store(out)["0"])
        assert all(len(k) == 4 for k in keys)
        bin_prefixes = sorted(set(k[0] for k in keys))
        assert len(bin_prefixes) >= 2


class TestRechunkViaLazy:
    """Rechunk via lazy API: zv[0].rechunk()."""

    def test_lazy_rechunk(self, tmp_path: Path) -> None:
        from zarr_vectors.types.polylines import write_polylines
        from zarr_vectors.lazy import open_zv

        rng = np.random.default_rng(42)
        polys = _make_streamlines(rng, 40)
        store = str(tmp_path / "tracts.zv")
        write_polylines(
            store, polys, chunk_shape=(200., 200., 200.),
            groups={0: list(range(20)), 1: list(range(20, 40))},
        )

        zv = open_zv(store)
        out = str(tmp_path / "rechunked.zv")
        result = zv[0].rechunk(by="group", output=out)
        assert result["bins_created"] == 2

        # Read rechunked store via lazy API
        zv_rc = open_zv(out)
        assert zv_rc[0].vertices.compute().shape[0] > 0


class TestCompositeStore:
    """Composite: points + graph + mesh in one store."""

    def test_composite_pipeline(self, tmp_path: Path) -> None:
        from zarr_vectors.types.points import write_points, read_points
        from zarr_vectors.composite import add_geometry, read_composite
        from zarr_vectors.validate import validate
        from zarr_vectors.lazy import open_zv

        rng = np.random.default_rng(42)
        store = str(tmp_path / "brain.zv")
        positions = rng.uniform(0, 200, size=(1000, 3)).astype(np.float32)
        write_points(store, positions, chunk_shape=(200., 200., 200.))

        # Add graph
        nodes = rng.uniform(0, 200, size=(80, 3)).astype(np.float32)
        edges = np.array([[i, i + 1] for i in range(79)], dtype=np.int64)
        add_geometry(store, "graph", positions=nodes, edges=edges)

        # Add mesh
        mesh_v = rng.uniform(0, 200, size=(200, 3)).astype(np.float32)
        mesh_f = rng.integers(0, 200, size=(300, 3)).astype(np.int64)
        add_geometry(store, "mesh", positions=mesh_v, faces=mesh_f)

        # Read individual types
        assert read_points(store)["vertex_count"] == 1000

        # Read composite
        comp = read_composite(store)
        assert comp["point_cloud"]["vertex_count"] == 1000
        assert comp["graph"]["vertex_count"] == 80
        assert comp["mesh"]["vertex_count"] == 200
        assert len(comp["graph"]["links"]) == 79
        assert comp["mesh"]["face_count"] == 300

        # Validates
        assert validate(store, level=4).ok

        # Lazy API
        zv = open_zv(store)
        assert len(zv.geometry_types) == 3
        assert zv[0].vertices.compute().shape[0] == 1000


class TestBackwardCompat:
    """All original store types work unchanged with new code."""

    def test_all_types(self, tmp_path: Path) -> None:
        from zarr_vectors.types.points import write_points, read_points
        from zarr_vectors.types.lines import write_lines, read_lines
        from zarr_vectors.types.polylines import write_polylines, read_polylines
        from zarr_vectors.types.meshes import write_mesh, read_mesh
        from zarr_vectors.types.graphs import write_graph, read_graph
        from zarr_vectors.validate import validate
        from zarr_vectors.multiresolution.coarsen import build_pyramid

        rng = np.random.default_rng(42)

        # Points
        s = str(tmp_path / "pts.zv")
        write_points(s, rng.uniform(0, 100, size=(200, 3)).astype(np.float32),
                     chunk_shape=(100., 100., 100.))
        assert read_points(s)["vertex_count"] == 200
        assert validate(s, level=5).ok

        # Lines
        s = str(tmp_path / "lin.zv")
        write_lines(s, np.array([[[10, 10, 10], [20, 20, 20]]], dtype=np.float32),
                    chunk_shape=(100., 100., 100.))
        assert read_lines(s)["line_count"] == 1

        # Polylines
        s = str(tmp_path / "pl.zv")
        write_polylines(s, [np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)],
                        chunk_shape=(100., 100., 100.))
        assert read_polylines(s)["polyline_count"] == 1

        # Mesh
        s = str(tmp_path / "m.zv")
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        f = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
        write_mesh(s, v, f, chunk_shape=(100., 100., 100.))
        assert read_mesh(s)["vertex_count"] == 4

        # Graph
        s = str(tmp_path / "g.zv")
        write_graph(s, np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32),
                    np.array([[0, 1]], dtype=np.int64), chunk_shape=(100., 100., 100.))
        assert read_graph(s)["node_count"] == 2

        # Pyramid
        s = str(tmp_path / "pyr.zv")
        write_points(s, rng.uniform(0, 1000, size=(10000, 3)).astype(np.float32),
                     chunk_shape=(100., 100., 100.))
        build_pyramid(s, factors=[(2.0, 1.0)])
        assert validate(s, level=5).ok
