"""Written metadata against the LinkML model of the Zarr Vectors convention.

``schema/zarr_vectors.linkml.yaml`` (and ``schema/build.py``) are copies of
``convention/`` in AllenInstitute/zarr_vectors, where the convention's
LinkML model is maintained; ``schema/regen.py`` generates
``schema/zarr_vectors.schema.json`` from them.  These tests check that:

* the generated schema is the one the convention publishes (by a hash
  of its canonical JSON, so the LinkML copy here has not drifted from
  it);
* the ``zarr_vectors`` objects ``to_dict()`` and the array writers emit
  validate against the class for their ``kind``, and broken ones do not;
* every member of every ``VALID_*`` constant set is a member of the
  matching schema enum, and the reverse.

If the LinkML source changes upstream, copy it (and ``build.py``) here,
run ``python schema/regen.py``, commit the regenerated schema, and set
``PUBLISHED_SHA256`` to the hash the failing test prints.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

from zarr_vectors.constants import (  # noqa: E402
    VALID_COARSENING_METHODS,
    VALID_CROSS_CHUNK_STRATEGIES,
    VALID_ENCODINGS,
    VALID_LINKS_CONVENTIONS,
    VALID_OBJIDX_CONVENTIONS,
    VALID_XLEVEL_STORAGE,
)
from zarr_vectors.core.metadata import LevelMetadata, RootMetadata  # noqa: E402
from zarr_vectors.exceptions import MetadataError  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schema" / "zarr_vectors.schema.json"
#: SHA-256 of the convention's published schema.json as canonical JSON
#: (sorted keys, no whitespace): the convention repository reformats it
#: with Prettier, so equality is as JSON, not bytes.
PUBLISHED_SHA256 = "82c10b17c22386ac67f71c56d3a450e9a78c35eb0a0fe0f0973a44f3e0173d3b"


def canonical_sha256(doc: dict) -> str:
    return hashlib.sha256(
        json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.fixture(scope="module")
def schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _validate(schema: dict, defs_name: str, instance: dict) -> None:
    """Validate ``instance`` against the named class in the schema."""
    fragment = {
        "$schema": schema.get("$schema"),
        "$defs": schema["$defs"],
        "$ref": f"#/$defs/{defs_name}",
    }
    jsonschema.Draft202012Validator(fragment).validate(instance)


def test_generated_schema_is_the_published_convention_schema(schema):
    """The convention repository publishes the schema generated from the
    same LinkML (reformatted by Prettier), so the two are equal as JSON."""
    got = canonical_sha256(schema)
    assert got == PUBLISHED_SHA256, (
        "schema/zarr_vectors.schema.json differs from the convention's "
        "schema.json: re-copy convention/ from AllenInstitute/zarr_vectors "
        "and run `python schema/regen.py`; if the convention itself moved, "
        f"set PUBLISHED_SHA256 = {got!r}"
    )


# ===================================================================
# Round-trip happy paths
# ===================================================================


def _minimal_root_md():
    return RootMetadata(
        spatial_index_dims=[
            {"name": "x", "type": "space", "unit": "um"},
            {"name": "y", "type": "space", "unit": "um"},
            {"name": "z", "type": "space", "unit": "um"},
        ],
        chunk_shape=(50.0, 50.0, 50.0),
        bounds=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
        geometry_types=["point_cloud"],
    )


def test_root_metadata_round_trip_validates(schema):
    wire = _minimal_root_md().to_dict()["zarr_vectors"]
    _validate(schema, "RootMetadata", wire)


def test_root_metadata_with_octree_chunks_validates(schema):
    rm = _minimal_root_md()
    rm.format_capabilities = ["octree_chunks"]
    rm.required_capabilities = ["octree_chunks"]
    wire = rm.to_dict()["zarr_vectors"]
    assert wire["required_capabilities"] == ["octree_chunks"]
    _validate(schema, "RootMetadata", wire)


def test_root_metadata_with_dense_int32_manifests_validates(schema):
    rm = _minimal_root_md()
    rm.manifest_layout, rm.manifest_block_dtype = "dense", "int32"
    wire = rm.to_dict()["zarr_vectors"]
    assert wire["manifest_block_dtype"] == "int32"
    _validate(schema, "RootMetadata", wire)


@pytest.mark.parametrize("shard_shape", [4, [4, 4, 2]])
def test_root_metadata_shard_shape_scalar_or_per_axis_validates(schema, shard_shape):
    """``create_store`` stamps a bare int when every axis agrees."""
    wire = _minimal_root_md().to_dict()["zarr_vectors"]
    wire["shard_shape"] = shard_shape
    _validate(schema, "RootMetadata", wire)


def test_level_metadata_minimal_validates(schema):
    lm = LevelMetadata(level=0, vertex_count=0, arrays_present=[])
    wire = lm.to_dict()["zarr_vectors"]
    _validate(schema, "LevelMetadata", wire)


def test_level_metadata_with_attribute_chunking_validates(schema):
    lm = LevelMetadata(
        level=0,
        vertex_count=500,
        arrays_present=["vertices", "object_index"],
        chunk_dims=["gene", "x", "y", "z"],
        chunk_attribute_name="gene",
        chunk_attribute_values=["A", "B"],
    )
    wire = lm.to_dict()["zarr_vectors"]
    _validate(schema, "LevelMetadata", wire)


@pytest.mark.parametrize("values", [[0, 2, -1], [30.0, 80.0], [3, 7]])
def test_level_metadata_with_numeric_bin_labels_validates(schema, values):
    """Categorical ints, lower edges and group indices are all numbers."""
    lm = LevelMetadata(
        level=0,
        vertex_count=500,
        arrays_present=["vertices", "object_index"],
        chunk_dims=["group", "x", "y", "z"],
        chunk_attribute_name="group",
        chunk_attribute_values=values,
    )
    wire = lm.to_dict()["zarr_vectors"]
    _validate(schema, "LevelMetadata", wire)


# ===================================================================
# Per-array ``zarr_vectors`` objects, sampled from core/arrays.py writers
# ===================================================================


@pytest.mark.parametrize(
    "defs_name,instance",
    [
        ("VerticesMetadata",
         {"kind": "vertices", "dtype": "float32", "encoding": "raw"}),
        # Intra-chunk (all-zero offsets): rows are L ints, no perm_idx.
        ("LinksMetadata",
         {"kind": "links", "dtype": "int64", "offsets": [[0, 0, 0]],
          "has_perm": False, "link_width": 2, "level_delta": 0}),
        # Cross-level: source is always input endpoint 0, so no perm_idx.
        ("LinksMetadata",
         {"kind": "links", "dtype": "int64", "offsets": [[0, 0, 0]],
          "has_perm": False, "link_width": 2, "level_delta": 1}),
        # Non-intra at delta 0, undirected => a non-identity placement is
        # possible, so rows widen to 1 + L with a leading perm_idx.
        ("LinksMetadata",
         {"kind": "links", "dtype": "int64", "offsets": [[0, 0, 1]],
          "has_perm": True, "link_width": 2, "level_delta": 0}),
        # L=3 triangle face spanning two neighbours.
        ("LinksMetadata",
         {"kind": "links", "dtype": "int64",
          "offsets": [[0, 0, 1], [0, 1, 0]], "has_perm": True,
          "link_width": 3, "level_delta": 0}),
        # link_width == 1 (parent refs): no other endpoint, so the offsets
        # list is empty and the array sits at the ``self`` segment.
        ("LinksMetadata",
         {"kind": "links", "dtype": "int64", "offsets": [],
          "has_perm": False, "link_width": 1, "level_delta": 1}),
        ("VertexAttributeMetadata",
         {"kind": "vertex_attribute", "name": "intensity", "dtype": "float32",
          "row_shape": []}),
        ("ObjectIndexMetadata",
         {"kind": "object_index", "num_objects": 42, "sid_ndim": 3,
          "layout": "vlen_manifests_v2"}),
        # An index created before any object is written carries only kind.
        ("ObjectIndexMetadata", {"kind": "object_index"}),
        ("ObjectAttributeMetadata",
         {"kind": "object_attribute", "name": "volume",
          "dtype": "float32", "shape": [42]}),
        ("GroupsMetadata", {"kind": "groups", "num_groups": 5}),
        ("GroupAttributeMetadata",
         {"kind": "group_attribute", "name": "label",
          "dtype": "int32", "shape": [5]}),
        # --- links family group: policy, pre-finalize (no counts yet) ---
        ("LinksFamilyMetadata",
         {"kind": "links_family", "level_delta": 0, "link_width": 2,
          "directed": False, "store": "canonical", "sid_ndim": 3}),
        # Post-finalize: duplicate store, physical rows exceed logical.
        ("LinksFamilyMetadata",
         {"kind": "links_family", "level_delta": 0, "link_width": 2,
          "directed": False, "store": "duplicate", "sid_ndim": 3,
          "num_links": 12, "num_physical_records": 20}),
        ("LinksFamilyMetadata",
         {"kind": "links_family", "level_delta": -1, "link_width": 1,
          "directed": True, "store": "canonical", "sid_ndim": 3,
          "num_links": 5, "num_physical_records": 5}),
        # create_links_family stamps no sid_ndim when it isn't known, and
        # finalize_links preserves that absence rather than inventing one.
        ("LinksFamilyMetadata",
         {"kind": "links_family", "level_delta": 0, "link_width": 2,
          "directed": False, "store": "canonical"}),
        # --- link attributes -------------------------------------------
        ("LinkAttributeMetadata",
         {"kind": "link_attribute", "name": "weight", "dtype": "float32",
          "row_shape": [], "offsets": [[0, 0, 1]], "level_delta": 0}),
        ("LinkAttributeMetadata",
         {"kind": "link_attribute", "name": "rgb", "dtype": "uint8",
          "row_shape": [3], "offsets": [[0, 0, 0]], "level_delta": 1}),
        ("LinkAttributeFamilyMetadata",
         {"kind": "link_attribute_family", "name": "weight",
          "level_delta": 1}),
        ("LinkAttributeFamilyMetadata",
         {"kind": "link_attribute_family", "name": "weight",
          "level_delta": 1, "num_links": 7}),
    ],
)
def test_per_array_metadata_validates(schema, defs_name, instance):
    _validate(schema, defs_name, instance)


@pytest.mark.parametrize(
    "offsets,delta,directed,store,expected",
    [
        # Intra (all-zero offsets): identity placement, input order kept.
        ([(0, 0, 0)], 0, False, "canonical", False),
        ([(0, 0, 0)], 0, False, "duplicate", False),
        ([(0, 0, 0)], 0, True, "canonical", False),
        # link_width == 1: empty offsets counts as intra.
        ([], 0, False, "canonical", False),
        # Cross-level: endpoints are distinguished by level, so the source
        # is always input endpoint 0 => identity placement.
        ([(0, 0, 1)], 1, False, "canonical", False),
        ([(0, 0, 1)], -1, False, "duplicate", False),
        # Non-intra at delta 0: perm_idx exactly when a non-identity
        # placement is possible.
        ([(0, 0, 1)], 0, False, "canonical", True),
        ([(0, 0, 1)], 0, False, "duplicate", True),
        ([(0, 0, 1)], 0, True, "duplicate", True),
        ([(0, 0, 1)], 0, True, "canonical", False),
    ],
)
def test_has_perm_rule_matches_shipped_writer(
    offsets, delta, directed, store, expected,
):
    """The ``has_perm`` rule documented on ``LinksMeta`` must match the
    writer's own discriminator.

    ``links_has_perm`` is the single definition the writer and reader
    consult to agree on record width, so if it moves, the schema prose
    (and the ``has_perm`` values in the cases above) are stale.
    """
    from zarr_vectors.core.arrays import links_has_perm

    assert links_has_perm(
        offsets, delta=delta, directed=directed, store=store,
    ) is expected


# ===================================================================
# Rejection cases
# ===================================================================


def test_reduction_factor_below_two_rejected(schema):
    rm = _minimal_root_md()
    rm.reduction_factor = 1  # < 2
    with pytest.raises(jsonschema.ValidationError):
        _validate(schema, "RootMetadata", rm.to_dict()["zarr_vectors"])


def test_root_without_a_space_axis_rejected(schema):
    rm = _minimal_root_md()
    wire = rm.to_dict()["zarr_vectors"]
    wire["axes"] = [{"name": "t", "type": "time"}]
    with pytest.raises(jsonschema.ValidationError):
        _validate(schema, "RootMetadata", wire)


def test_unknown_geometry_type_accepted(schema):
    """The convention leaves ``geometry_types`` open, so a reader meeting a
    newer geometry type is not made to reject the whole store."""
    wire = _minimal_root_md().to_dict()["zarr_vectors"]
    wire["geometry_types"] = ["pancake"]
    _validate(schema, "RootMetadata", wire)


def test_object_sparsity_out_of_range_rejected(schema):
    lm = LevelMetadata(level=0, vertex_count=0, arrays_present=[])
    wire = lm.to_dict()["zarr_vectors"]
    wire["object_sparsity"] = 2.0
    with pytest.raises(jsonschema.ValidationError):
        _validate(schema, "LevelMetadata", wire)


def test_per_array_wrong_kind_rejected(schema):
    # Otherwise-valid links metadata whose only defect is the tag: kind
    # says "vertices" but LinksMetadata pins kind to "links".  Keep every
    # other slot present so this fails on the discriminator alone and not
    # incidentally on a missing required slot.
    instance = {"kind": "vertices", "dtype": "int64",
                "offsets": [[0, 0, 0]], "has_perm": False,
                "link_width": 2, "level_delta": 0}
    with pytest.raises(jsonschema.ValidationError):
        _validate(schema, "LinksMetadata", instance)


def test_links_family_kind_not_interchangeable_with_links(schema):
    """The family group and its offsets arrays are distinct shapes.

    Both live under ``links/<delta>/`` and share slot names, so a reader
    that confused the two would silently decode policy as records.
    """
    family = {"kind": "links_family", "level_delta": 0, "link_width": 2,
              "directed": False, "store": "canonical", "sid_ndim": 3}
    with pytest.raises(jsonschema.ValidationError):
        _validate(schema, "LinksMetadata", family)


def test_links_missing_has_perm_rejected(schema):
    """``has_perm`` is the record-width discriminator: rows are L ints, or
    1 + L when true.  A writer that drops it leaves readers guessing a
    width, which mis-parses silently rather than raising."""
    instance = {"kind": "links", "dtype": "int64",
                "offsets": [[0, 0, 1]], "link_width": 2, "level_delta": 0}
    with pytest.raises(jsonschema.ValidationError):
        _validate(schema, "LinksMetadata", instance)


def test_unknown_kind_rejected(schema):
    """``zarr_vectors`` must match exactly one kind class."""
    with pytest.raises(jsonschema.ValidationError):
        _validate(schema, "ZarrVectorsBlock", {"kind": "pancake"})


# ===================================================================
# Constants <-> schema enum agreement
# ===================================================================


@pytest.mark.parametrize(
    "enum_name,members",
    [
        ("LinksConvention", VALID_LINKS_CONVENTIONS),
        ("ObjectIndexConvention", VALID_OBJIDX_CONVENTIONS),
        ("CrossChunkStrategy", VALID_CROSS_CHUNK_STRATEGIES),
        ("CrossLevelStorage", VALID_XLEVEL_STORAGE),
        ("VertexEncoding", VALID_ENCODINGS),
        ("CoarseningMethod", VALID_COARSENING_METHODS),
    ],
)
def test_constant_set_matches_schema_enum(schema, enum_name, members):
    """Every member of ``VALID_*`` must appear in the corresponding enum."""
    schema_enum = schema["$defs"][enum_name]["enum"]
    missing = set(members) - set(schema_enum)
    extra = set(schema_enum) - set(members)
    assert not missing, (
        f"{enum_name}: constant members not in schema: {sorted(missing)}"
    )
    assert not extra, (
        f"{enum_name}: schema members not in constants: {sorted(extra)}"
    )


# ===================================================================
# Format-version cutoff (0.9.0 hard break: single per-chunk-array layout).
# ===================================================================


def test_format_version_is_0_10_0():
    """The current ZV writer stamps 0.10.0; bump tests here when bumping."""
    from zarr_vectors.constants import FORMAT_VERSION

    assert FORMAT_VERSION == "0.10.0", (
        f"FORMAT_VERSION drifted to {FORMAT_VERSION!r}; if intentional "
        f"update this test and the version-cutoff check in "
        f"zarr_vectors.core.metadata.RootMetadata.validate()."
    )


@pytest.mark.parametrize("zv_version", ["0.9.0", "0.9.1", "0.9.2", "0.9.3", "0.9.4"])
def test_0_10_0_moves_the_read_cutoff(zv_version):
    """0.10.0 moved every metadata key onto the Zarr Vectors convention,
    so the cutoff moves with it: a 0.9.x store is migrated in place
    (``migrate_metadata_0_10``), not read.  Within 0.9 every addition was
    additive and the cutoff stayed at 0.9.0."""
    from zarr_vectors.core.metadata import RootMetadata

    axes = [{"name": n, "type": "space"} for n in "xyz"]
    meta = RootMetadata(
        spatial_index_dims=axes,
        chunk_shape=(100.0, 100.0, 100.0),
        bounds=([0.0] * 3, [100.0] * 3),
        geometry_types=["point_cloud"],
        zv_version=zv_version,
    )
    with pytest.raises(MetadataError, match=zv_version):
        meta.validate()


def test_an_absent_shard_shape_reads_as_unsharded():
    """What every store written before 0.9.3 looks like."""
    from zarr_vectors.core.metadata import RootMetadata

    axes = [{"name": n, "type": "space"} for n in "xyz"]
    meta = RootMetadata.from_dict({
        "zarr_vectors": {
            "kind": "root",
            "zv_version": "0.10.0",
            "axes": axes,
            "chunk_shape": [100.0] * 3,
            "bounds": [[0.0] * 3, [100.0] * 3],
            "geometry_types": ["point_cloud"],
        },
    })
    assert meta.shard_shape is None
    assert meta.manifest_layout is None  # absent: the vlen layout


@pytest.mark.parametrize(
    "stale_version",
    ["0.3", "0.4.1", "0.5.0", "0.6.0", "0.7.0", "0.8.0", "0.8.1"],
)
def test_pre_0_9_0_stores_rejected(stale_version):
    """Pre-0.9.0 stores (Option-G per-chunk sub-arrays) must fail
    validate() with a clear message mentioning ``zv_version``."""
    from zarr_vectors.exceptions import MetadataError

    md = _minimal_root_md()
    md.zv_version = stale_version
    with pytest.raises(MetadataError, match="zv_version"):
        md.validate()


def test_0_9_0_store_is_refused_by_validate():
    md = _minimal_root_md()
    md.zv_version = "0.9.0"
    with pytest.raises(MetadataError, match="0.9.0"):
        md.validate()
