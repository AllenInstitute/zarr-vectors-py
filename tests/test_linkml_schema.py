"""Sanity-check: hand-written dataclasses round-trip against the LinkML schema.

The LinkML schema in ``schema/zarr_vectors.linkml.yaml`` is a *reference*
model.  The runtime source of truth is the Python dataclass tree in
``zarr_vectors/core/metadata.py``.  These tests prevent the two from
drifting silently:

* Examples emitted by ``to_dict()`` validate against the generated
  JSON Schema (modulo a documented representation bridge — see
  :func:`_to_linkml_logical_form`).
* Deliberately-broken examples are rejected by the JSON Schema.
* Every member of every ``VALID_*`` constant appears as an enum
  member in the schema.

If the LinkML source changes, run ``python schema/regen.py`` and
commit the regenerated JSON Schema; this test re-loads it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

from zarr_vectors.constants import (
    VALID_CROSS_CHUNK_STRATEGIES,
    VALID_ENCODINGS,
    VALID_GEOMETRY_TYPES,
    VALID_LINKS_CONVENTIONS,
    VALID_OBJIDX_CONVENTIONS,
    VALID_XLEVEL_STORAGE,
)
from zarr_vectors.core.metadata import LevelMetadata, RootMetadata


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema" / "zarr_vectors.schema.json"


@pytest.fixture(scope="module")
def schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _to_linkml_logical_form(root_dict: dict) -> dict:
    """Bridge wire-format ``.zattrs`` content into the LinkML object model.

    The on-disk dict envelopes the payload under ``zarr_vectors`` /
    ``zarr_vectors_level``; LinkML models the inner payload directly.
    A couple of slot representations also differ:

    * ``bounds`` on disk is ``[[min...], [max...]]`` (two parallel lists).
      LinkML models it as a :class:`BoundingBox` object with
      ``min_corner`` / ``max_corner`` properties.
    * ``crs`` on disk may be a dict, scalar, or null; LinkML models it
      as the open ``CRS`` class (``additionalProperties: true``).  The
      JSON Schema accepts any of those, so no bridging is needed
      anymore.

    This is the only place that bridges the two representations; if it
    grows beyond a handful of lines we should tighten the schema.
    """
    out = dict(root_dict)
    if isinstance(out.get("bounds"), list) and len(out["bounds"]) == 2 \
            and isinstance(out["bounds"][0], list):
        out["bounds"] = {
            "min_corner": list(out["bounds"][0]),
            "max_corner": list(out["bounds"][1]),
        }
    return out


def _validate(schema: dict, defs_name: str, instance: dict) -> None:
    """Validate ``instance`` against the named class in the schema."""
    fragment = {
        "$schema": schema.get("$schema"),
        "$defs": schema["$defs"],
        "$ref": f"#/$defs/{defs_name}",
    }
    jsonschema.validate(instance, fragment)


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
    rm = _minimal_root_md()
    wire = rm.to_dict()["zarr_vectors"]
    _validate(schema, "RootMetadata", _to_linkml_logical_form(wire))


def test_level_metadata_minimal_validates(schema):
    lm = LevelMetadata(level=0, vertex_count=0, arrays_present=[])
    wire = lm.to_dict()["zarr_vectors_level"]
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
    wire = lm.to_dict()["zarr_vectors_level"]
    _validate(schema, "LevelMetadata", wire)


# ===================================================================
# Per-array .zattrs shapes — sampled from the writers in core/arrays.py
# ===================================================================


@pytest.mark.parametrize(
    "defs_name,instance",
    [
        ("VerticesMeta",
         {"zv_array": "vertices", "dtype": "float32", "encoding": "raw"}),
        # Intra-chunk (all-zero offsets): rows are L ints, no perm_idx.
        ("LinksMeta",
         {"zv_array": "links", "dtype": "int64", "offsets": [[0, 0, 0]],
          "has_perm": False, "link_width": 2, "level_delta": 0}),
        # Cross-level: source is always input endpoint 0, so no perm_idx.
        ("LinksMeta",
         {"zv_array": "links", "dtype": "int64", "offsets": [[0, 0, 0]],
          "has_perm": False, "link_width": 2, "level_delta": 1}),
        # Non-intra at delta 0, undirected => a non-identity placement is
        # possible, so rows widen to 1 + L with a leading perm_idx.
        ("LinksMeta",
         {"zv_array": "links", "dtype": "int64", "offsets": [[0, 0, 1]],
          "has_perm": True, "link_width": 2, "level_delta": 0}),
        # L=3 triangle face spanning two neighbours.
        ("LinksMeta",
         {"zv_array": "links", "dtype": "int64",
          "offsets": [[0, 0, 1], [0, 1, 0]], "has_perm": True,
          "link_width": 3, "level_delta": 0}),
        # link_width == 1 (parent refs): no other endpoint, so the offsets
        # list is empty and the array sits at the ``self`` segment.
        ("LinksMeta",
         {"zv_array": "links", "dtype": "int64", "offsets": [],
          "has_perm": False, "link_width": 1, "level_delta": 1}),
        ("AttributeMeta",
         {"zv_array": "attribute", "name": "intensity", "dtype": "float32"}),
        ("ObjectIndexMeta",
         {"zv_array": "object_index", "num_objects": 42, "sid_ndim": 3}),
        ("ObjectAttributeMeta",
         {"zv_array": "object_attribute", "name": "volume",
          "dtype": "float32", "shape": [42]}),
        ("GroupingsMeta",
         {"zv_array": "groupings", "num_groups": 5}),
        ("GroupingsAttributeMeta",
         {"zv_array": "groupings_attribute", "name": "label",
          "dtype": "int32", "shape": [5]}),
        # --- links family group: policy, pre-finalize (no counts yet) ---
        ("LinksFamilyMeta",
         {"zv_array": "links_family", "level_delta": 0, "link_width": 2,
          "directed": False, "store": "canonical", "sid_ndim": 3}),
        # Post-finalize: duplicate store, physical rows exceed logical.
        ("LinksFamilyMeta",
         {"zv_array": "links_family", "level_delta": 0, "link_width": 2,
          "directed": False, "store": "duplicate", "sid_ndim": 3,
          "num_links": 12, "num_physical_records": 20}),
        ("LinksFamilyMeta",
         {"zv_array": "links_family", "level_delta": -1, "link_width": 1,
          "directed": True, "store": "canonical", "sid_ndim": 3,
          "num_links": 5, "num_physical_records": 5}),
        # create_links_family stamps no sid_ndim when it isn't known, and
        # finalize_links preserves that absence rather than inventing one.
        ("LinksFamilyMeta",
         {"zv_array": "links_family", "level_delta": 0, "link_width": 2,
          "directed": False, "store": "canonical"}),
        # --- link attributes -------------------------------------------
        ("LinkAttributeMeta",
         {"zv_array": "link_attribute", "name": "weight", "dtype": "float32",
          "offsets": [[0, 0, 0]], "level_delta": 0}),
        # write_link_attributes also stamps row_shape (tail dims per row).
        ("LinkAttributeMeta",
         {"zv_array": "link_attribute", "name": "weight", "dtype": "float32",
          "row_shape": [], "offsets": [[0, 0, 1]], "level_delta": 0}),
        ("LinkAttributeMeta",
         {"zv_array": "link_attribute", "name": "rgb", "dtype": "uint8",
          "row_shape": [3], "offsets": [[0, 0, 0]], "level_delta": 1}),
        ("LinkAttributeFamilyMeta",
         {"zv_array": "link_attribute_family", "name": "weight",
          "level_delta": 1}),
        ("LinkAttributeFamilyMeta",
         {"zv_array": "link_attribute_family", "name": "weight",
          "level_delta": 1, "num_links": 7}),
    ],
)
def test_per_array_zattrs_shapes_validate(schema, defs_name, instance):
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


def test_negative_reduction_factor_rejected(schema):
    rm = _minimal_root_md()
    rm.reduction_factor = 1  # < 2
    wire = _to_linkml_logical_form(rm.to_dict()["zarr_vectors"])
    with pytest.raises(jsonschema.ValidationError):
        _validate(schema, "RootMetadata", wire)


def test_unknown_geometry_type_rejected(schema):
    wire = _to_linkml_logical_form(_minimal_root_md().to_dict()["zarr_vectors"])
    wire["geometry_types"] = ["pancake"]
    with pytest.raises(jsonschema.ValidationError):
        _validate(schema, "RootMetadata", wire)


def test_object_sparsity_out_of_range_rejected(schema):
    lm = LevelMetadata(level=0, vertex_count=0, arrays_present=[])
    wire = lm.to_dict()["zarr_vectors_level"]
    wire["object_sparsity"] = 2.0
    with pytest.raises(jsonschema.ValidationError):
        _validate(schema, "LevelMetadata", wire)


def test_per_array_wrong_discriminator_rejected(schema):
    # Otherwise-valid LinksMeta whose only defect is the tag: zv_array
    # says "vertices" but LinksMeta has equals_string: links.  Keep every
    # other slot present so this fails on the discriminator alone and not
    # incidentally on a missing required slot.
    instance = {"zv_array": "vertices", "dtype": "int64",
                "offsets": [[0, 0, 0]], "has_perm": False,
                "link_width": 2, "level_delta": 0}
    with pytest.raises(jsonschema.ValidationError):
        _validate(schema, "LinksMeta", instance)


def test_links_family_tag_not_interchangeable_with_links(schema):
    """The family group and its offsets arrays are distinct shapes.

    Both live under ``links/<delta>/`` and share slot names, so a reader
    that confused the two would silently decode policy as records.
    """
    family = {"zv_array": "links_family", "level_delta": 0, "link_width": 2,
              "directed": False, "store": "canonical", "sid_ndim": 3}
    with pytest.raises(jsonschema.ValidationError):
        _validate(schema, "LinksMeta", family)


def test_links_meta_missing_has_perm_rejected(schema):
    """``has_perm`` is the record-width discriminator: rows are L ints, or
    1 + L when true.  A writer that drops it leaves readers guessing a
    width, which mis-parses silently rather than raising."""
    instance = {"zv_array": "links", "dtype": "int64",
                "offsets": [[0, 0, 1]], "link_width": 2, "level_delta": 0}
    with pytest.raises(jsonschema.ValidationError):
        _validate(schema, "LinksMeta", instance)


# ===================================================================
# Constants ↔ schema enum agreement
# ===================================================================


@pytest.mark.parametrize(
    "enum_name,members",
    [
        ("LinksConvention", VALID_LINKS_CONVENTIONS),
        ("ObjectIndexConvention", VALID_OBJIDX_CONVENTIONS),
        ("CrossChunkStrategy", VALID_CROSS_CHUNK_STRATEGIES),
        ("CrossLevelStorage", VALID_XLEVEL_STORAGE),
        ("GeometryType", VALID_GEOMETRY_TYPES),
        ("Encoding", VALID_ENCODINGS),
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


def test_format_version_is_0_9_0():
    """The current ZV writer stamps 0.9.0; bump tests here when bumping."""
    from zarr_vectors.constants import FORMAT_VERSION

    assert FORMAT_VERSION == "0.9.0", (
        f"FORMAT_VERSION drifted to {FORMAT_VERSION!r}; if intentional "
        f"update this test and the version-cutoff check in "
        f"zarr_vectors.core.metadata.RootMetadata.validate()."
    )


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


def test_0_9_0_store_passes_validate():
    md = _minimal_root_md()
    md.zv_version = "0.9.0"
    md.validate()  # should not raise
