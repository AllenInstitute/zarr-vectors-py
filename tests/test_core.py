"""Step 03 tests: metadata dataclasses, serialisation, conventions, and store lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from zarr_vectors.core.metadata import (
    ArrayMetadata,
    LevelMetadata,
    ParametricTypeDef,
    RootMetadata,
    build_axes_metadata,
    build_coordinate_transforms,
    deserialise_parametric_types,
    serialise_parametric_types,
    validate_axes,
    validate_conventions,
    requires_links_array,
    requires_object_index,
    DEFAULT_PARAMETRIC_TYPES,
    PARAMETRIC_PLANE,
    PARAMETRIC_LINE,
    PARAMETRIC_SPHERE,
)
from zarr_vectors.core.store import (
    FsGroup,
    create_resolution_level,
    create_store,
    get_parametric_group,
    get_resolution_level,
    list_resolution_levels,
    open_store,
    read_level_metadata,
    read_parametric_types,
    read_root_metadata,
    set_bounds,
    store_info,
    write_parametric_types,
)
from zarr_vectors.constants import (
    LINKS_EXPLICIT,
    LINKS_IMPLICIT_SEQUENTIAL,
    LINKS_IMPLICIT_BRANCHES,
    OBJIDX_IDENTITY,
    OBJIDX_STANDARD,
    CROSS_CHUNK_EXPLICIT,
)
from zarr_vectors.exceptions import ConventionError, MetadataError, StoreError


# ===================================================================
# Helpers
# ===================================================================

def _make_root_meta(**overrides) -> RootMetadata:
    """Create a valid RootMetadata with sensible defaults, applying overrides.

    Used by the ``TestRootMetadata`` suite (which validates the
    dataclass directly).  Callers that just want to seed a store via
    ``create_store`` should use :func:`_root_kwargs` instead — passing
    a fully-populated :class:`RootMetadata` to ``create_store`` is no
    longer supported.
    """
    defaults = dict(
        spatial_index_dims=[
            {"name": "x", "type": "space", "unit": "um"},
            {"name": "y", "type": "space", "unit": "um"},
            {"name": "z", "type": "space", "unit": "um"},
        ],
        chunk_shape=(100.0, 100.0, 100.0),
        bounds=([0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]),
        geometry_types=["point_cloud"],
    )
    defaults.update(overrides)
    return RootMetadata(**defaults)


def _root_kwargs(**overrides) -> dict:
    """Flat kwargs for ``create_store`` with the same sensible defaults
    that :func:`_make_root_meta` uses for the dataclass."""
    defaults: dict = dict(
        axes=[
            {"name": "x", "type": "space", "unit": "um"},
            {"name": "y", "type": "space", "unit": "um"},
            {"name": "z", "type": "space", "unit": "um"},
        ],
        chunk_shape=(100.0, 100.0, 100.0),
        bounds=([0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]),
        geometry_types=["point_cloud"],
    )
    # Translate the in-memory dataclass field name to the create_store
    # kwarg name on the fly so tests can keep using the familiar
    # ``spatial_index_dims=...`` override.
    if "spatial_index_dims" in overrides:
        overrides["axes"] = overrides.pop("spatial_index_dims")
    defaults.update(overrides)
    return defaults


def _make_level_meta(level: int = 0, **overrides) -> LevelMetadata:
    """Create a valid LevelMetadata."""
    defaults: dict = dict(
        level=level,
        vertex_count=1000,
        arrays_present=["vertices"],
    )
    if level > 0:
        defaults["bin_shape"] = (200.0, 200.0, 200.0)
        defaults["coarsening_method"] = "per_object"
        defaults["parent_level"] = level - 1
    defaults.update(overrides)
    return LevelMetadata(**defaults)


# ===================================================================
# Axes / CRS helpers
# ===================================================================

class TestAxesMetadata:

    def test_build_3d(self) -> None:
        axes = build_axes_metadata(
            ["x", "y", "z"], ["space", "space", "space"], ["um", "um", "um"]
        )
        assert len(axes) == 3
        assert axes[0] == {"name": "x", "type": "space", "unit": "um"}

    def test_build_xyzt(self) -> None:
        axes = build_axes_metadata(
            ["x", "y", "z", "t"],
            ["space", "space", "space", "time"],
            ["um", "um", "um", "s"],
        )
        assert len(axes) == 4
        assert axes[3]["type"] == "time"

    def test_mismatched_lengths(self) -> None:
        try:
            build_axes_metadata(["x", "y"], ["space"], ["um", "um"])
            assert False, "Should raise"
        except MetadataError:
            pass

    def test_no_space_axes(self) -> None:
        try:
            build_axes_metadata(["t", "c"], ["time", "channel"], ["s", ""])
            assert False, "Should raise"
        except MetadataError:
            pass

    def test_validate_axes_ok(self) -> None:
        axes = [
            {"name": "x", "type": "space", "unit": "um"},
            {"name": "y", "type": "space", "unit": "um"},
        ]
        validate_axes(axes)  # should not raise

    def test_validate_axes_too_few(self) -> None:
        try:
            validate_axes([{"name": "x", "type": "space", "unit": "um"}])
            assert False
        except MetadataError:
            pass

    def test_validate_axes_missing_name(self) -> None:
        try:
            validate_axes([
                {"type": "space", "unit": "um"},
                {"name": "y", "type": "space", "unit": "um"},
            ])
            assert False
        except MetadataError:
            pass


class TestCoordinateTransforms:

    def test_scale_only(self) -> None:
        t = build_coordinate_transforms([0.5, 0.5, 0.5])
        assert len(t) == 1
        assert t[0]["type"] == "scale"
        assert t[0]["scale"] == [0.5, 0.5, 0.5]

    def test_scale_and_translation(self) -> None:
        t = build_coordinate_transforms([1.0, 1.0, 1.0], [10.0, 20.0, 30.0])
        assert len(t) == 2
        assert t[1]["type"] == "translation"
        assert t[1]["translation"] == [10.0, 20.0, 30.0]


# ===================================================================
# Convention validation
# ===================================================================

class TestConventions:

    def test_valid_conventions(self) -> None:
        validate_conventions(
            LINKS_IMPLICIT_SEQUENTIAL, OBJIDX_STANDARD, CROSS_CHUNK_EXPLICIT
        )

    def test_invalid_links_convention(self) -> None:
        try:
            validate_conventions("bogus", OBJIDX_STANDARD, CROSS_CHUNK_EXPLICIT)
            assert False
        except ConventionError:
            pass

    def test_invalid_objidx_convention(self) -> None:
        try:
            validate_conventions(LINKS_IMPLICIT_SEQUENTIAL, "bogus", CROSS_CHUNK_EXPLICIT)
            assert False
        except ConventionError:
            pass

    def test_identity_with_multiple_chunks(self) -> None:
        try:
            validate_conventions(
                LINKS_IMPLICIT_SEQUENTIAL, OBJIDX_IDENTITY,
                CROSS_CHUNK_EXPLICIT, num_spatial_chunks=10
            )
            assert False
        except ConventionError:
            pass

    def test_identity_with_single_chunk_ok(self) -> None:
        validate_conventions(
            LINKS_IMPLICIT_SEQUENTIAL, OBJIDX_IDENTITY,
            CROSS_CHUNK_EXPLICIT, num_spatial_chunks=1
        )

    def test_requires_links(self) -> None:
        assert requires_links_array(LINKS_EXPLICIT) is True
        assert requires_links_array(LINKS_IMPLICIT_SEQUENTIAL) is False
        assert requires_links_array(LINKS_IMPLICIT_BRANCHES) is False

    def test_requires_object_index(self) -> None:
        assert requires_object_index(OBJIDX_STANDARD, num_chunks=5) is True
        assert requires_object_index(OBJIDX_IDENTITY, num_chunks=1) is False


# ===================================================================
# RootMetadata
# ===================================================================

class TestRootMetadata:

    def test_create_and_validate(self) -> None:
        m = _make_root_meta()
        m.validate()

    def test_sid_ndim(self) -> None:
        m = _make_root_meta()
        assert m.sid_ndim == 3

    def test_round_trip(self) -> None:
        m = _make_root_meta(
            geometry_types=["point_cloud", "streamline"],
            links_convention=LINKS_IMPLICIT_SEQUENTIAL,
            reduction_factor=8,
        )
        d = m.to_dict()
        # 0.5.0+: axes live in NGFF ``multiscales[0].axes`` rather
        # than under ``zarr_vectors.spatial_index_dims``, so a round
        # trip through to_dict / from_dict requires injecting the
        # NGFF block on the side.
        d["multiscales"] = [{
            "version": "0.4",
            "axes": list(m.spatial_index_dims),
            "datasets": [{"path": "0", "coordinateTransformations": [
                {"type": "scale", "scale": [1.0] * m.sid_ndim},
            ]}],
            "metadata": {"format": "zarr_vectors"},
        }]
        m2 = RootMetadata.from_dict(d)
        assert m2.zv_version == m.zv_version
        assert m2.chunk_shape == m.chunk_shape
        assert m2.geometry_types == m.geometry_types
        assert m2.reduction_factor == 8
        assert m2.sid_ndim == 3

    def test_missing_zarr_vectors_key(self) -> None:
        try:
            RootMetadata.from_dict({"other": "stuff"})
            assert False
        except MetadataError:
            pass

    def test_missing_required_field(self) -> None:
        try:
            RootMetadata.from_dict({"zarr_vectors": {"zv_version": "0.2"}})
            assert False
        except MetadataError:
            pass

    def test_bad_chunk_shape_length(self) -> None:
        m = _make_root_meta(chunk_shape=(100.0, 100.0))  # 2D shape, 3D axes
        try:
            m.validate()
            assert False
        except MetadataError:
            pass

    def test_negative_chunk_shape(self) -> None:
        m = _make_root_meta(chunk_shape=(100.0, -50.0, 100.0))
        try:
            m.validate()
            assert False
        except MetadataError:
            pass

    def test_bad_geometry_type(self) -> None:
        m = _make_root_meta(geometry_types=["unicorn"])
        try:
            m.validate()
            assert False
        except MetadataError:
            pass

    def test_reduction_factor_too_low(self) -> None:
        m = _make_root_meta(reduction_factor=1)
        try:
            m.validate()
            assert False
        except MetadataError:
            pass

    def test_defaults(self) -> None:
        m = _make_root_meta()
        assert m.links_convention == LINKS_IMPLICIT_SEQUENTIAL
        assert m.object_index_convention == OBJIDX_STANDARD
        assert m.cross_chunk_strategy == CROSS_CHUNK_EXPLICIT
        assert m.reduction_factor == 8


# ===================================================================
# LevelMetadata
# ===================================================================

class TestLevelMetadata:

    def test_level0(self) -> None:
        m = _make_level_meta(0)
        m.validate()
        assert m.bin_shape is None
        assert m.parent_level is None

    def test_level1(self) -> None:
        m = _make_level_meta(1)
        m.validate()
        assert m.bin_shape == (200.0, 200.0, 200.0)
        assert m.parent_level == 0

    def test_round_trip(self) -> None:
        m = _make_level_meta(1)
        d = m.to_dict()
        m2 = LevelMetadata.from_dict(d)
        assert m2.level == 1
        assert m2.bin_shape == (200.0, 200.0, 200.0)
        assert m2.vertex_count == 1000
        assert m2.parent_level == 0

    def test_level0_with_bin_size_invalid(self) -> None:
        m = _make_level_meta(0, bin_shape=(100.0, 100.0, 100.0))
        try:
            m.validate()
            assert False
        except MetadataError:
            pass

    def test_level1_without_bin_size_invalid(self) -> None:
        m = LevelMetadata(level=1, vertex_count=500, arrays_present=["vertices"])
        try:
            m.validate()
            assert False
        except MetadataError:
            pass

    def test_missing_key(self) -> None:
        try:
            LevelMetadata.from_dict({"zarr_vectors_level": {"level": 0}})
            assert False
        except MetadataError:
            pass


# ===================================================================
# ArrayMetadata
# ===================================================================

class TestArrayMetadata:

    def test_basic(self) -> None:
        m = ArrayMetadata(name="vertices", dtype="float32")
        d = m.to_dict()
        m2 = ArrayMetadata.from_dict(d)
        assert m2.name == "vertices"
        assert m2.dtype == "float32"
        assert m2.encoding == "raw"

    def test_with_channels(self) -> None:
        m = ArrayMetadata(
            name="attributes/gene_expression",
            dtype="float16",
            channel_names=["SNAP25", "GAD1", "SLC17A7"],
            channel_dtype="float16",
        )
        d = m.to_dict()
        m2 = ArrayMetadata.from_dict(d)
        assert m2.channel_names == ["SNAP25", "GAD1", "SLC17A7"]

    def test_draco_encoding(self) -> None:
        m = ArrayMetadata(name="vertices", dtype="float32", encoding="draco")
        m.validate()

    def test_bad_encoding(self) -> None:
        m = ArrayMetadata(name="vertices", dtype="float32", encoding="lz4")
        try:
            m.validate()
            assert False
        except MetadataError:
            pass


# ===================================================================
# ParametricTypeDef
# ===================================================================

class TestParametricTypes:

    def test_standard_types(self) -> None:
        assert PARAMETRIC_PLANE.type_id == 0
        assert PARAMETRIC_PLANE.name == "plane"
        assert PARAMETRIC_PLANE.coefficients == ["A", "B", "C", "D"]
        assert PARAMETRIC_LINE.type_id == 1
        assert PARAMETRIC_SPHERE.type_id == 2

    def test_serialise_round_trip(self) -> None:
        d = serialise_parametric_types(DEFAULT_PARAMETRIC_TYPES)
        assert "parametric_types" in d
        assert "0" in d["parametric_types"]
        types = deserialise_parametric_types(d)
        assert len(types) == 3
        assert types[0].name == "plane"
        assert types[1].name == "line"
        assert types[2].name == "sphere"

    def test_empty_registry(self) -> None:
        types = deserialise_parametric_types({})
        assert types == []

    def test_single_type(self) -> None:
        d = serialise_parametric_types([PARAMETRIC_PLANE])
        types = deserialise_parametric_types(d)
        assert len(types) == 1
        assert types[0].coefficients == ["A", "B", "C", "D"]


# ===================================================================
# FsGroup
# ===================================================================

class TestFsGroup:

    def test_create_and_attrs(self, tmp_store_path: Path) -> None:
        g = FsGroup(tmp_store_path, create=True)
        g.attrs["foo"] = "bar"
        assert g.attrs["foo"] == "bar"

    def test_create_subgroup(self, tmp_store_path: Path) -> None:
        g = FsGroup(tmp_store_path, create=True)
        child = g.create_group("child")
        assert "child" in g
        assert isinstance(child, FsGroup)

    def test_getitem(self, tmp_store_path: Path) -> None:
        g = FsGroup(tmp_store_path, create=True)
        g.create_group("a")
        a = g["a"]
        assert isinstance(a, FsGroup)

    def test_getitem_nested(self, tmp_store_path: Path) -> None:
        g = FsGroup(tmp_store_path, create=True)
        g.create_group("a").create_group("b")
        b = g["a/b"]
        assert isinstance(b, FsGroup)

    def test_getitem_missing(self, tmp_store_path: Path) -> None:
        g = FsGroup(tmp_store_path, create=True)
        try:
            g["nonexistent"]
            assert False
        except StoreError:
            pass

    def test_iter(self, tmp_store_path: Path) -> None:
        g = FsGroup(tmp_store_path, create=True)
        g.create_group("b")
        g.create_group("a")
        g.create_group("c")
        names = list(g)
        assert names == ["a", "b", "c"]  # sorted

    def test_chunk_io(self, tmp_store_path: Path) -> None:
        g = FsGroup(tmp_store_path, create=True)
        data = b"hello chunk"
        g.write_bytes("vertices", "0.0.0", data)
        assert g.chunk_exists("vertices", "0.0.0")
        assert g.read_bytes("vertices", "0.0.0") == data

    def test_chunk_missing(self, tmp_store_path: Path) -> None:
        g = FsGroup(tmp_store_path, create=True)
        try:
            g.read_bytes("vertices", "9.9.9")
            assert False
        except StoreError:
            pass

    def test_list_chunks(self, tmp_store_path: Path) -> None:
        g = FsGroup(tmp_store_path, create=True)
        g.write_bytes("vertices", "0.0.1", b"a")
        g.write_bytes("vertices", "0.0.0", b"b")
        g.write_bytes("vertices", "1.0.0", b"c")
        assert g.list_chunks("vertices") == ["0.0.0", "0.0.1", "1.0.0"]

    def test_array_meta(self, tmp_store_path: Path) -> None:
        g = FsGroup(tmp_store_path, create=True)
        g.write_array_meta("vertices", {"dtype": "float32", "encoding": "raw"})
        meta = g.read_array_meta("vertices")
        assert meta["dtype"] == "float32"


# ===================================================================
# Store API
# ===================================================================

class TestStoreCreate:

    def test_create_store(self, tmp_store_path: Path) -> None:
        root = create_store(tmp_store_path, **_root_kwargs())
        assert tmp_store_path.is_dir()
        assert "zarr_vectors" in root.attrs.to_dict()
        assert f"0" in root
        # /parametric is no longer auto-created; it is created lazily by
        # get_parametric_group() on first use.
        assert "parametric" not in root
        # The warm shell already contains the empty ragged-vertex pair.
        res0 = root["0"]
        assert "vertices" in res0
        assert "vertex_fragments" in res0

    def test_create_store_minimal(self, tmp_store_path: Path) -> None:
        """create_store(path) with no kwargs produces a warm 3D shell
        with default bounds = ([0,0,0], [128,128,128])."""
        root = create_store(tmp_store_path)
        assert tmp_store_path.is_dir()
        attrs = root.attrs.to_dict()
        zv = attrs["zarr_vectors"]
        assert zv["zv_version"]
        assert zv["bounds"] == [[0.0, 0.0, 0.0], [128.0, 128.0, 128.0]]
        assert zv["chunk_shape"] == [128.0, 128.0, 128.0]
        # Axes now live in the NGFF multiscales block, not under
        # zarr_vectors.spatial_index_dims (0.5.0+).
        assert "spatial_index_dims" not in zv
        assert len(attrs["multiscales"][0]["axes"]) == 3
        assert zv["geometry_types"] == []
        assert "0" in root
        assert "parametric" not in root
        res0 = root["0"]
        assert "vertices" in res0
        assert "vertex_fragments" in res0

    def test_create_store_ndim_2d(self, tmp_store_path: Path) -> None:
        """ndim kwarg resolves to a 2D store with default 2D bounds."""
        root = create_store(tmp_store_path, ndim=2)
        attrs = root.attrs.to_dict()
        zv = attrs["zarr_vectors"]
        assert zv["bounds"] == [[0.0, 0.0], [128.0, 128.0]]
        # Axes live in NGFF ``multiscales[0].axes`` as of format 0.5.0.
        assert len(attrs["multiscales"][0]["axes"]) == 2

    def test_create_store_explicit_bounds(self, tmp_store_path: Path) -> None:
        """Explicit bounds kwarg overrides the default."""
        root = create_store(
            tmp_store_path,
            bounds=([-10.0, -10.0, -10.0], [50.0, 50.0, 50.0]),
        )
        zv = root.attrs.to_dict()["zarr_vectors"]
        assert zv["bounds"] == [[-10.0, -10.0, -10.0], [50.0, 50.0, 50.0]]

    def test_create_store_ndim_mismatch(self, tmp_store_path: Path) -> None:
        """Inconsistent kwargs raise MetadataError."""
        try:
            create_store(
                tmp_store_path,
                ndim=3,
                chunk_shape=(100.0, 100.0),
            )
            assert False
        except MetadataError:
            pass

    def test_create_store_already_exists(self, tmp_store_path: Path) -> None:
        create_store(tmp_store_path, **_root_kwargs())
        try:
            create_store(tmp_store_path, **_root_kwargs())
            assert False
        except StoreError:
            pass

    def test_create_store_invalid_metadata(self, tmp_store_path: Path) -> None:
        try:
            # chunk_shape arity (2) inconsistent with axes arity (3).
            create_store(tmp_store_path, **_root_kwargs(chunk_shape=(100.0, 100.0)))
            assert False
        except MetadataError:
            pass


class TestStoreOpen:

    def test_open_store(self, tmp_store_path: Path) -> None:
        create_store(tmp_store_path, **_root_kwargs())
        root = open_store(tmp_store_path)
        assert isinstance(root, FsGroup)

    def test_open_nonexistent(self, tmp_store_path: Path) -> None:
        try:
            open_store(tmp_store_path)
            assert False
        except StoreError:
            pass

    def test_open_invalid_store(self, tmp_store_path: Path) -> None:
        tmp_store_path.mkdir(parents=True)
        (tmp_store_path / ".zattrs").write_text("{}")
        try:
            open_store(tmp_store_path)
            assert False
        except StoreError:
            pass


class TestResolutionLevels:

    def test_create_and_list(self, tmp_store_path: Path) -> None:
        root = create_store(tmp_store_path, **_root_kwargs())
        lm0 = _make_level_meta(0)
        create_resolution_level(root, 0, lm0)

        lm1 = _make_level_meta(1)
        create_resolution_level(root, 1, lm1)

        levels = list_resolution_levels(root)
        assert levels == [0, 1]

    def test_get_level(self, tmp_store_path: Path) -> None:
        root = create_store(tmp_store_path, **_root_kwargs())
        lm0 = _make_level_meta(0)
        create_resolution_level(root, 0, lm0)
        lvl = get_resolution_level(root, 0)
        assert isinstance(lvl, FsGroup)

    def test_get_missing_level(self, tmp_store_path: Path) -> None:
        root = create_store(tmp_store_path, **_root_kwargs())
        try:
            get_resolution_level(root, 99)
            assert False
        except StoreError:
            pass

    def test_read_level_metadata(self, tmp_store_path: Path) -> None:
        root = create_store(tmp_store_path, **_root_kwargs())
        lm = _make_level_meta(0, vertex_count=42)
        create_resolution_level(root, 0, lm)
        read_back = read_level_metadata(root, 0)
        assert read_back.vertex_count == 42


class TestRootMetadataReadWrite:

    def test_read_root_metadata(self, tmp_store_path: Path) -> None:
        root = create_store(tmp_store_path, **_root_kwargs(geometry_types=["mesh"]))
        read_back = read_root_metadata(root)
        assert read_back.geometry_types == ["mesh"]
        assert read_back.chunk_shape == (100.0, 100.0, 100.0)


class TestParametricTypesStore:

    def test_write_and_read(self, tmp_store_path: Path) -> None:
        root = create_store(tmp_store_path, **_root_kwargs())
        write_parametric_types(root, DEFAULT_PARAMETRIC_TYPES)
        types = read_parametric_types(root)
        assert len(types) == 3
        assert types[0].name == "plane"

    def test_empty(self, tmp_store_path: Path) -> None:
        root = create_store(tmp_store_path, **_root_kwargs())
        types = read_parametric_types(root)
        assert types == []


class TestStoreInfo:

    def test_basic_info(self, tmp_store_path: Path) -> None:
        root = create_store(
            tmp_store_path,
            **_root_kwargs(geometry_types=["point_cloud", "skeleton"]),
        )
        lm = _make_level_meta(0, vertex_count=5000)
        create_resolution_level(root, 0, lm)
        write_parametric_types(root, [PARAMETRIC_PLANE])

        info = store_info(root)
        assert info["zv_version"].startswith("0.8")
        assert info["geometry_types"] == ["point_cloud", "skeleton"]
        assert info["chunk_shape"] == [100.0, 100.0, 100.0]
        assert len(info["levels"]) == 1
        assert info["levels"][0]["vertex_count"] == 5000
        assert len(info["parametric_types"]) == 1


class TestSetBounds:

    def test_expand(self, tmp_store_path: Path) -> None:
        """Expanding bounds is allowed without force."""
        root = create_store(tmp_store_path)
        set_bounds(root, ([0.0, 0.0, 0.0], [200.0, 200.0, 200.0]))
        zv = root.attrs.to_dict()["zarr_vectors"]
        assert zv["bounds"] == [[0.0, 0.0, 0.0], [200.0, 200.0, 200.0]]

    def test_contract_requires_force(self, tmp_store_path: Path) -> None:
        """Contracting bounds without force raises."""
        root = create_store(tmp_store_path)
        try:
            set_bounds(root, ([10.0, 10.0, 10.0], [100.0, 100.0, 100.0]))
            assert False
        except StoreError:
            pass

    def test_contract_with_force(self, tmp_store_path: Path) -> None:
        """Contracting with force=True is allowed."""
        root = create_store(tmp_store_path)
        set_bounds(
            root,
            ([10.0, 10.0, 10.0], [100.0, 100.0, 100.0]),
            force=True,
        )
        zv = root.attrs.to_dict()["zarr_vectors"]
        assert zv["bounds"] == [[10.0, 10.0, 10.0], [100.0, 100.0, 100.0]]

    def test_shape_mismatch_raises(self, tmp_store_path: Path) -> None:
        root = create_store(tmp_store_path)  # 3D store
        try:
            set_bounds(root, ([0.0, 0.0], [100.0, 100.0]))  # 2D
            assert False
        except MetadataError:
            pass
