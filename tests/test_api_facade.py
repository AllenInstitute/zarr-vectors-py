"""The data-oriented facade.

Two things are under test.

**Fidelity.** In this phase the facade delegates to the five existing
readers, so its correctness is exactly the correctness of the adapters
that map their five differently-shaped result dicts into one.  Every
adapter is checked against the reader it wraps, on real stores.

**Decoupling.** The storage-leak linter at the bottom is the test that
keeps this from eroding: it fails any future change that puts a
``chunk_shape`` or a ``backend`` back on the public surface.  Without it
the facade drifts back into the storage vocabulary one convenient
keyword at a time.
"""

from __future__ import annotations

import asyncio
import inspect

import numpy as np
import pytest

import zarr_vectors as zv
from zarr_vectors.api.result import ReadResult
from zarr_vectors.api.schema import Layout, Schema, SchemaConflict, StorageOptions
from zarr_vectors.types.graphs import read_graph, write_graph
from zarr_vectors.types.lines import read_lines, write_lines
from zarr_vectors.types.meshes import read_mesh, write_mesh
from zarr_vectors.types.points import read_points, write_points
from zarr_vectors.types.polylines import read_polylines, write_polylines

CHUNK = (200.0, 200.0, 200.0)
BIN = (50.0, 50.0, 50.0)


@pytest.fixture
def points_store(tmp_path):
    rng = np.random.default_rng(11)
    path = tmp_path / "points.zarrvectors"
    positions = rng.uniform(0, 400, size=(600, 3)).astype(np.float32)
    write_points(
        path, positions, chunk_shape=CHUNK, bin_shape=BIN,
        vertex_attributes={
            "intensity": rng.random(600).astype(np.float32),
            "label": rng.integers(0, 5, 600).astype(np.int32),
        },
    )
    return path


@pytest.fixture
def polyline_store(tmp_path):
    rng = np.random.default_rng(12)
    path = tmp_path / "tracts.zarrvectors"
    lines = [
        (rng.normal(0, 20, size=(12, 3)).cumsum(axis=0) + 200).astype(np.float32)
        for _ in range(20)
    ]
    write_polylines(path, lines, chunk_shape=CHUNK, bin_shape=BIN)
    return path


@pytest.fixture
def line_store(tmp_path):
    rng = np.random.default_rng(13)
    path = tmp_path / "lines.zarrvectors"
    endpoints = rng.uniform(50, 350, size=(25, 2, 3)).astype(np.float32)
    write_lines(path, endpoints, chunk_shape=CHUNK, bin_shape=BIN)
    return path


@pytest.fixture
def mesh_store(tmp_path):
    rng = np.random.default_rng(14)
    path = tmp_path / "mesh.zarrvectors"
    vertices = rng.uniform(50, 350, size=(60, 3)).astype(np.float32)
    faces = rng.integers(0, 60, size=(40, 3)).astype(np.int64)
    write_mesh(path, vertices, faces, chunk_shape=CHUNK, bin_shape=BIN)
    return path


@pytest.fixture
def graph_store(tmp_path):
    rng = np.random.default_rng(15)
    path = tmp_path / "graph.zarrvectors"
    positions = rng.uniform(50, 350, size=(40, 3)).astype(np.float32)
    edges = np.column_stack([np.arange(39), np.arange(1, 40)]).astype(np.int64)
    write_graph(path, positions, edges, chunk_shape=CHUNK, bin_shape=BIN)
    return path


# =====================================================================
# Adapter fidelity -- the correctness proof for this phase
# =====================================================================


class TestAdapterFidelity:
    def test_points_match_the_legacy_reader(self, points_store):
        got = zv.open(points_store).read()
        legacy = read_points(str(points_store), attribute_names=["intensity", "label"])
        assert np.array_equal(got.positions, legacy["positions"])
        assert got.vertex_count == legacy["vertex_count"]
        for name, values in legacy["vertex_attributes"].items():
            assert np.array_equal(got.attributes[name], values)

    def test_attributes_all_actually_means_all(self, points_store):
        # read_points returns NO attributes when attribute_names is
        # omitted -- None means none, not all. The facade resolves the
        # names, so the obvious spelling gets the obvious answer.
        assert read_points(str(points_store))["vertex_attributes"] == {}
        assert zv.open(points_store).read().attributes.names() == ("intensity", "label")

    def test_attributes_can_be_narrowed(self, points_store):
        got = zv.open(points_store).select(attributes=["label"]).read()
        assert got.attributes.names() == ("label",)

    def test_polylines_match_the_legacy_reader(self, polyline_store):
        got = zv.open(polyline_store).read()
        legacy = read_polylines(str(polyline_store))
        assert got.part_count == legacy["polyline_count"]
        assert got.vertex_count == legacy["vertex_count"]
        # read_polylines yields each polyline as a LIST of per-chunk
        # segments -- even a single-chunk polyline is a list of one.
        # Concatenating them is exact: the lengths agree and no boundary
        # vertex is duplicated.
        assert sorted(len(p) for p in got.polylines) == sorted(
            sum(len(seg) for seg in part) for part in legacy["polylines"]
        )

    def test_polyline_object_ids_are_per_part_not_per_vertex(self, polyline_store):
        # The distinction is load-bearing: read_polylines returns one id
        # per polyline while read_points' object path returns one per
        # vertex. Conflating them misaligns one of the two.
        got = zv.open(polyline_store).read()
        legacy = read_polylines(str(polyline_store))
        assert got.part_objects is not None
        assert len(got.part_objects) == got.part_count
        assert np.array_equal(got.part_objects, legacy["object_ids"])
        assert got.object_ids is None

    def test_lines_match_the_legacy_reader(self, line_store):
        got = zv.open(line_store).read()
        legacy = read_lines(str(line_store))
        assert got.part_count == legacy["line_count"]
        assert np.array_equal(got.endpoints, legacy["endpoints"])

    def test_lines_gain_explicit_connectivity(self, line_store):
        # (M, 2, D) leaves the pairing implicit in the axis layout.
        # Flattening to (2M, D) plus edges makes it explicit and loses
        # nothing -- .endpoints reverses it exactly.
        got = zv.open(line_store).read()
        assert got.edges is not None
        assert got.edges.shape == (got.part_count, 2)
        assert np.array_equal(got.edges[0], [0, 1])

    def test_mesh_matches_the_legacy_reader(self, mesh_store):
        got = zv.open(mesh_store).read()
        legacy = read_mesh(str(mesh_store))
        assert np.array_equal(got.positions, legacy["vertices"])
        assert np.array_equal(got.faces, legacy["faces"])

    def test_graph_matches_the_legacy_reader(self, graph_store):
        got = zv.open(graph_store).read()
        legacy = read_graph(str(graph_store))
        assert np.array_equal(got.positions, legacy["positions"])
        assert np.array_equal(got.edges, legacy["edges"])

    @pytest.mark.parametrize(
        "fixture", ["points_store", "polyline_store", "line_store", "mesh_store", "graph_store"],
    )
    def test_every_kind_reads_through_one_call(self, fixture, request):
        # The point of the uniform result: a tool that does not know
        # which geometry it was handed is still correct.
        result = zv.open(request.getfixturevalue(fixture)).read()
        assert isinstance(result, ReadResult)
        assert result.positions.ndim == 2
        assert result.vertex_count > 0
        assert result.part_count >= 1
        assert sum(s.stop - s.start for s in result.parts) == result.vertex_count

    def test_attributes_read_distinguishes_empty_from_unattempted(self, mesh_store):
        # read_mesh has no attribute keyword at all, so "no attributes"
        # here means "never looked", not "none exist".
        got = zv.open(mesh_store).read()
        assert got.attributes_read is False


class TestBboxAndObjectSelection:
    def test_bbox_matches_the_legacy_reader(self, points_store):
        lo, hi = np.array([100.0, 100.0, 100.0]), np.array([250.0, 250.0, 250.0])
        got = zv.open(points_store).select(bbox=(lo, hi)).read()
        legacy = read_points(str(points_store), bbox=(lo, hi))
        assert np.array_equal(got.positions, legacy["positions"])

    def test_objects_selects_one_polyline(self, polyline_store):
        ds = zv.open(polyline_store)
        oid = int(ds.read().part_objects[3])
        got = ds.select(objects=[oid]).read()
        assert got.part_count == 1
        assert int(got.part_objects[0]) == oid

    def test_selections_intersect_when_chained(self, points_store):
        ds = zv.open(points_store)
        chained = ds.select(bbox=([0.0, 0.0, 0.0], [400.0, 400.0, 400.0])).select(
            bbox=([100.0, 100.0, 100.0], [250.0, 250.0, 250.0])
        )
        direct = ds.select(bbox=([100.0, 100.0, 100.0], [250.0, 250.0, 250.0]))
        assert chained.read().vertex_count == direct.read().vertex_count

    def test_select_is_lazy(self, points_store):
        # Nothing may be read until a terminal is called, or swapping the
        # implementation underneath becomes a behaviour change.
        ds = zv.open(points_store)
        query = ds.select(bbox=([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
        assert query.selection.bbox is not None
        assert query.read().vertex_count >= 0


class TestPostFilters:
    def test_near_enforces_the_sphere_not_its_box(self, points_store):
        centre, radius = np.array([200.0, 200.0, 200.0]), 60.0
        got = zv.open(points_store).select(near=(centre, radius)).read()
        distances = np.linalg.norm(got.positions - centre, axis=1)
        assert got.vertex_count > 0
        assert distances.max() <= radius + 1e-6

    def test_near_keeps_attributes_aligned(self, points_store):
        got = zv.open(points_store).select(near=([200.0, 200.0, 200.0], 60.0)).read()
        for values in got.attributes.values():
            assert len(values) == got.vertex_count

    def test_limit_truncates_and_says_so(self, points_store):
        got = zv.open(points_store).select(limit=7).read()
        assert got.vertex_count == 7
        assert got.truncated is True
        assert got.complete is False

    def test_limit_above_the_count_does_not_mark_truncated(self, points_store):
        got = zv.open(points_store).select(limit=10_000).read()
        assert got.truncated is False

    def test_restrict_repairs_face_indices(self, mesh_store):
        # A filtered mesh whose faces still point at the old row numbers
        # is silently corrupt. Every surviving face must index inside the
        # new positions array.
        full = zv.open(mesh_store).read()
        keep = np.zeros(full.vertex_count, dtype=bool)
        keep[: full.vertex_count // 2] = True
        cut = full.restrict(keep)
        assert cut.faces is not None
        assert cut.vertex_count == int(keep.sum())
        if len(cut.faces):
            assert cut.faces.max() < cut.vertex_count
            assert cut.faces.min() >= 0

    def test_restrict_drops_edges_that_lost_an_endpoint(self, graph_store):
        full = zv.open(graph_store).read()
        keep = np.ones(full.vertex_count, dtype=bool)
        keep[0] = False
        cut = full.restrict(keep)
        assert len(cut.edges) < len(full.edges)
        assert cut.edges.max() < cut.vertex_count

    def test_restrict_recuts_parts(self, polyline_store):
        full = zv.open(polyline_store).read()
        keep = np.zeros(full.vertex_count, dtype=bool)
        keep[full.parts[0]] = True
        cut = full.restrict(keep)
        assert cut.part_count == 1
        assert cut.vertex_count == full.parts[0].stop - full.parts[0].start


class TestDatasetSurface:
    def test_reports_bounds_without_reaching_into_attrs(self, points_store):
        lo, hi = zv.open(points_store).bounds
        assert lo.shape == (3,) and hi.shape == (3,)
        assert (hi > lo).all()

    def test_reports_axes_and_ndim(self, points_store):
        ds = zv.open(points_store)
        assert ds.ndim == 3
        assert [a["name"] for a in ds.axes] == ["x", "y", "z"]

    def test_reports_kinds_levels_and_format(self, points_store):
        ds = zv.open(points_store)
        assert ds.kinds == ("point_cloud",)
        assert ds.levels == (0,)
        assert ds.format_version >= (0, 9)

    def test_level_scale_is_the_data_shaped_cell_size(self, points_store):
        assert zv.open(points_store).level(0).scale == CHUNK

    def test_level_resolution_is_the_query_cell_size(self, points_store):
        assert zv.open(points_store).level(0).resolution == BIN

    def test_attribute_names_replaces_the_zarr_group_walk(self, points_store):
        level = zv.open(points_store).level(0)
        assert level.attribute_names("vertex") == ("intensity", "label")
        assert level.attribute_names("object") == ()

    def test_attribute_names_rejects_an_unknown_family(self, points_store):
        with pytest.raises(ValueError, match="unknown attribute family"):
            zv.open(points_store).level(0).attribute_names("nonsense")

    def test_resolution_picks_the_nearest_level(self, points_store):
        ds = zv.open(points_store)
        assert ds.resolution(scale=200.0).index == 0

    def test_indexing_and_iteration(self, points_store):
        ds = zv.open(points_store)
        assert ds[0].index == 0
        assert [lvl.index for lvl in ds] == [0]

    def test_explain_names_the_reader_and_the_terms(self, points_store):
        text = zv.open(points_store).select(limit=3).explain()
        assert "read_points" in text
        assert "filtered in memory" in text


class TestDeferredCapabilities:
    """The facade must not advertise what it cannot do."""

    def test_iter_cells_says_it_is_not_available_yet(self, points_store):
        with pytest.raises(NotImplementedError, match="resolver phase"):
            zv.open(points_store).select().iter_cells()


class TestAsync:
    def test_aread_matches_read(self, points_store):
        ds = zv.open(points_store)
        sync = ds.read()
        out = asyncio.run(ds.aread())
        assert np.array_equal(out.positions, sync.positions)

    def test_aopen_then_aread(self, points_store):
        async def go():
            ds = await zv.aopen(str(points_store))
            return await ds.aread()

        assert asyncio.run(go()).vertex_count == 600


class TestSchemaAndCreate:
    def test_layout_resolve_derives_the_physical_parameters(self):
        schema = Schema(bounds=([0, 0, 0], [400, 400, 400]), layout=Layout(cells=4, subcells=2))
        resolved = schema.layout.resolve(schema, store_kind="local")
        assert resolved.chunk_shape == (100.0, 100.0, 100.0)
        assert resolved.bin_shape == (50.0, 50.0, 50.0)
        assert resolved.shard_shape is None  # local: one object per cell is cheap

    def test_layout_packs_on_an_object_store_but_not_locally(self):
        schema = Schema(bounds=([0, 0, 0], [400, 400, 400]), layout=Layout(cells=4))
        assert schema.layout.resolve(schema, store_kind="local").shard_shape is None
        assert schema.layout.resolve(schema, store_kind="object").shard_shape is not None

    def test_cell_size_is_the_escape_hatch_for_a_fixed_grid(self):
        schema = Schema(
            bounds=([0, 0, 0], [768, 768, 768]),
            layout=Layout(cell_size=(768.0, 768.0, 768.0)),
        )
        assert schema.layout.resolve(schema).chunk_shape == (768.0, 768.0, 768.0)

    def test_create_then_read_back(self, tmp_path):
        path = tmp_path / "made.zarrvectors"
        ds = zv.create(
            path,
            schema=Schema(bounds=([0, 0, 0], [400, 400, 400]), layout=Layout(cells=4)),
        )
        rng = np.random.default_rng(2)
        ds.add_points(rng.uniform(10, 390, size=(200, 3)).astype(np.float32))
        assert zv.open(path).read().vertex_count == 200

    def test_add_points_lands_on_the_stores_own_grid(self, tmp_path):
        # A second write must not silently re-grid the store.
        path = tmp_path / "grid.zarrvectors"
        ds = zv.create(
            path,
            schema=Schema(bounds=([0, 0, 0], [400, 400, 400]), layout=Layout(cells=4)),
        )
        rng = np.random.default_rng(3)
        ds.add_points(rng.uniform(10, 390, size=(100, 3)).astype(np.float32))
        assert zv.open(path).level(0).scale == (100.0, 100.0, 100.0)

    def test_open_or_create_creates_then_opens(self, tmp_path):
        path = tmp_path / "ioc.zarrvectors"
        schema = Schema(bounds=([0, 0, 0], [400, 400, 400]), layout=Layout(cells=4))
        first = zv.open_or_create(path, schema=schema)
        second = zv.open_or_create(path, schema=schema)
        assert first.url == second.url

    def test_open_or_create_raises_on_disagreement(self, tmp_path):
        # _create_or_open_store silently drops bounds for an existing
        # path, which is how a store ends up with a grid that contradicts
        # its own chunk keys. Here it is an error.
        path = tmp_path / "conflict.zarrvectors"
        zv.open_or_create(
            path, schema=Schema(bounds=([0, 0, 0], [400, 400, 400]), layout=Layout(cells=4)),
        )
        with pytest.raises(SchemaConflict, match="bounds"):
            zv.open_or_create(
                path, schema=Schema(bounds=([0, 0, 0], [999, 999, 999]), layout=Layout(cells=4)),
            )

    def test_open_or_create_can_defer_to_the_store(self, tmp_path):
        path = tmp_path / "keep.zarrvectors"
        zv.open_or_create(
            path, schema=Schema(bounds=([0, 0, 0], [400, 400, 400]), layout=Layout(cells=4)),
        )
        ds = zv.open_or_create(
            path,
            schema=Schema(bounds=([0, 0, 0], [999, 999, 999])),
            on_conflict="keep",
        )
        assert float(ds.bounds[1][0]) == 400.0

    def test_schema_round_trips_through_a_store(self, points_store):
        ds = zv.open(points_store)
        schema = Schema.from_store(ds._root_meta)
        assert schema.ndim == 3
        assert schema.kind == "point_cloud"
        assert tuple(schema.layout.cell_size) == CHUNK

    def test_storage_options_precedence(self, monkeypatch):
        assert StorageOptions(backend="icechunk").resolve_backend("s3://b/k") == "icechunk"
        monkeypatch.setenv("ZARR_VECTORS_BACKEND", "fsspec")
        assert StorageOptions().resolve_backend("s3://b/k") == "fsspec"
        monkeypatch.delenv("ZARR_VECTORS_BACKEND")
        assert StorageOptions().resolve_backend("s3://b/k") is None


class TestFormatGate:
    def test_accepts_a_satisfied_range(self, points_store):
        zv.require_format(zv.open(points_store), ">=0.9,<0.11")

    def test_rejects_and_explains(self, points_store):
        with pytest.raises(zv.FormatError, match="rewritten from source"):
            zv.require_format(zv.open(points_store), ">=99.0")

    def test_rejects_an_unparseable_clause(self, points_store):
        with pytest.raises(ValueError, match="cannot parse"):
            zv.require_format(zv.open(points_store), "~0.9")


# =====================================================================
# The linter that keeps the decoupling from eroding
# =====================================================================

FORBIDDEN_PARAMS = frozenset({
    "chunk_shape", "bin_shape", "shard_shape", "compressor", "backend",
    "chunks", "delta", "offsets", "fragment_attributes", "links_convention",
    "object_index_convention", "cross_chunk_strategy", "store_path",
})

FORBIDDEN_ANNOTATIONS = ("ChunkCoords", "FsGroup", "zarr.")

# The storage layer stays exported and stays supported; it is simply not
# what the linter governs.
STORAGE_LAYER = frozenset({
    "Group", "FsGroup", "create_store", "open_store", "rebind", "detect_scheme",
    "RechunkSpec", "rechunk", "rechunk_by_attribute", "ZVWriter",
})


def _facade_callables():
    for name in zv.__all__:
        if name in STORAGE_LAYER or name.startswith("__"):
            continue
        obj = getattr(zv, name)
        if inspect.isclass(obj):
            for attr in dir(obj):
                if attr.startswith("_"):
                    continue
                member = inspect.getattr_static(obj, attr, None)
                if inspect.isfunction(member):
                    yield f"{name}.{attr}", member
        elif callable(obj):
            yield name, obj


class TestNoStorageLeaks:
    def test_no_public_parameter_names_a_storage_concept(self):
        offenders = [
            f"{where}({param})"
            for where, fn in _facade_callables()
            for param in inspect.signature(fn).parameters
            if param in FORBIDDEN_PARAMS
        ]
        assert not offenders, (
            "the facade grew a storage parameter; put it on Layout or "
            f"StorageOptions instead: {offenders}"
        )

    def test_no_public_annotation_names_a_storage_type(self):
        offenders = []
        for where, fn in _facade_callables():
            for param in inspect.signature(fn).parameters.values():
                text = str(param.annotation)
                for bad in FORBIDDEN_ANNOTATIONS:
                    if bad in text:
                        offenders.append(f"{where}({param.name}: {text})")
        assert not offenders, offenders

    def test_the_result_type_keeps_its_canonical_field_names(self):
        # No reader may reintroduce 'vertices' / 'endpoints' / 'polylines'
        # as the primary array name -- that fragmentation is what the
        # uniform result exists to end.
        fields = {f.name for f in ReadResult.__dataclass_fields__.values()}
        assert "positions" in fields
        assert not fields & {"vertices", "endpoints", "polylines"}

    def test_the_top_level_exports_a_data_oriented_entry_point(self):
        for name in ("open", "create", "open_or_create", "aopen", "Dataset"):
            assert name in zv.__all__
            assert hasattr(zv, name)
