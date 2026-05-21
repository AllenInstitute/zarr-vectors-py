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
        ("LinksMeta",
         {"zv_array": "links", "dtype": "int64", "link_width": 2,
          "level_delta": 0}),
        ("LinksMeta",
         {"zv_array": "links", "dtype": "int64", "link_width": 2,
          "level_delta": 1}),
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
        ("CrossChunkLinksMeta",
         {"zv_array": "cross_chunk_links", "sid_ndim": 3,
          "level_delta": 0, "link_width": 2, "layout": "partitioned_v1"}),
        ("CrossChunkLinksMeta",
         {"zv_array": "cross_chunk_links", "sid_ndim": 3,
          "level_delta": -1, "link_width": 1, "layout": "partitioned_v1"}),
        ("CrossChunkLinksMeta",
         {"zv_array": "cross_chunk_links", "sid_ndim": 3,
          "level_delta": 0, "link_width": 3, "layout": "partitioned_v1"}),
        ("LinkAttributeMeta",
         {"zv_array": "link_attribute", "name": "weight", "dtype": "float32",
          "level_delta": 0}),
        ("CrossChunkLinkAttributeMeta",
         {"zv_array": "cross_chunk_link_attribute", "name": "weight",
          "dtype": "float32", "level_delta": 1, "layout": "partitioned_v1"}),
    ],
)
def test_per_array_zattrs_shapes_validate(schema, defs_name, instance):
    _validate(schema, defs_name, instance)


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
    # zv_array says "vertices" but we're validating against LinksMeta —
    # the LinksMeta class has equals_string: links, so the wrong tag
    # must be rejected.
    instance = {"zv_array": "vertices", "dtype": "int64", "link_width": 2}
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
# Format-version cutoff (0.7.0 hard break: per-level chunk_shape).
# ===================================================================


def test_format_version_is_0_7_0():
    """The current ZV writer stamps 0.7.0; bump tests here when bumping."""
    from zarr_vectors.constants import FORMAT_VERSION

    assert FORMAT_VERSION == "0.7.0", (
        f"FORMAT_VERSION drifted to {FORMAT_VERSION!r}; if intentional "
        f"update this test and the version-cutoff check in "
        f"zarr_vectors.core.metadata.RootMetadata.validate()."
    )


@pytest.mark.parametrize(
    "stale_version",
    ["0.3", "0.3.5", "0.4", "0.4.0", "0.4.1", "0.5.0", "0.6.0"],
)
def test_pre_0_7_0_stores_rejected(stale_version):
    """Pre-0.7.0 stores must fail validate() with a clear message that
    mentions the rename so the user knows what changed."""
    from zarr_vectors.exceptions import MetadataError

    md = _minimal_root_md()
    md.zv_version = stale_version
    with pytest.raises(MetadataError, match="zv_version"):
        md.validate()


def test_0_7_0_store_passes_validate():
    md = _minimal_root_md()
    md.zv_version = "0.7.0"
    md.validate()  # should not raise
