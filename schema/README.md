# `schema/` — machine-readable reference

This directory holds the canonical machine-readable schema for the
Zarr Vectors (ZV) on-disk format.  The runtime Python source of truth
remains [`zarr_vectors/core/metadata.py`]; the files here are a
parallel reference that other languages and tools can consume.

## Files

| File | Status | Purpose |
|---|---|---|
| `zarr_vectors.linkml.yaml` | hand-edited | [LinkML] source.  Edit this to change the schema. |
| `zarr_vectors.schema.json` | generated | JSON Schema (Draft 2020-12).  Downstream validators consume this. |
| `reference.md`             | generated | Markdown reference rendered from the LinkML source. |
| `regen.py`                 | hand-edited | CLI that re-runs the generators. |

## Regenerating

Install the dev tooling once:

```bash
pip install "zarr-vectors[dev]"   # brings in linkml + jsonschema
```

Then:

```bash
python schema/regen.py            # regenerate artefacts in place
python schema/regen.py --check    # exit non-zero if regen would change files
```

CI is expected to run `--check`; if it fails, run without `--check`
locally and commit the regenerated files.  Only the JSON Schema is
strict-checked — LinkML's Markdown doc generator currently emits
class-row lists in non-deterministic order, so `reference.md` is
regenerated freely and is not byte-pinned.

The dev extras pin `linkml>=1.7,<2` and `jsonschema>=4.0,<5` to keep
generated diffs between contributors quiet.

## Coverage

| Schema item | Source |
|---|---|
| `RootMetadata`            | `zarr_vectors.core.metadata.RootMetadata` |
| `LevelMetadata`           | `zarr_vectors.core.metadata.LevelMetadata` |
| 12 per-array attribute shapes | `zarr_vectors.core.arrays.write_array_meta` call-sites — discriminator slot `zv_array` |
| Enums                     | `zarr_vectors.constants.VALID_*` frozensets and `CAP_*` tokens |

The `ParametricTypeDef` registry (plane/line/sphere) is implemented in
`zarr_vectors.core.metadata` (with built-in `PARAMETRIC_PLANE`,
`PARAMETRIC_LINE`, `PARAMETRIC_SPHERE` defaults); the LinkML schema
itself does not yet model it — adding `ParametricTypeDef` to the
schema is a follow-up.

## Representation bridge

A handful of slots have a different representation in the LinkML
object model than in the wire-format attributes dict:

- **`bounds`** — on disk it is the two-element list `[[min...],
  [max...]]`; the LinkML schema models it as a `BoundingBox` class
  with `min_corner` / `max_corner` properties.
- **`crs`** — on disk it is `None` or a free-form dict; the LinkML
  schema models this as the open `CRS` class
  (`additionalProperties: true`), so any of `null` / dict / scalar
  validates.

`tests/test_linkml_schema.py` documents the bridge in one place
(`_to_linkml_logical_form`); if it grows beyond a few lines we should
tighten the schema instead.

## Vocabulary mappings

The schema declares prefixes for [OME-Zarr NGFF] and [Schema.org] and
attaches `class_uri` / `slot_uri` to the obvious slots:

- `Axis` → `ngff:Axis` (mirrors OME-Zarr RFC 4/5 axis objects).
- `zv_version` → `schema:version`.
- `crs` → `schema:coordinateReferenceSystem` (loose match).

Wider ontology mapping (OBO / EFO / BIDS) is intentionally out of
scope for v1.

## Cross-language consumers

The generated JSON Schema is the public contract.  To validate an
attributes dict from any JSON-Schema-capable language:

```python
import json, jsonschema
schema = json.load(open("schema/zarr_vectors.schema.json"))
fragment = {
    "$defs": schema["$defs"],
    "$ref": "#/$defs/RootMetadata",
}
jsonschema.validate(my_root_metadata_dict, fragment)
```

(Note the bridge for `bounds` / `crs` described above.)

## Versioning

The schema's `$id` carries the format version
(`https://w3id.org/zarr-vectors/schema/0.4`); the value is nominal
until the project owns the `w3id.org` namespace prefix.  No code
reads it — it's a stable identifier for downstream tools.

[LinkML]:                       https://linkml.io/
[OME-Zarr NGFF]:                https://ngff.openmicroscopy.org/
[Schema.org]:                   https://schema.org/
[`zarr_vectors/core/metadata.py`]: ../zarr_vectors/core/metadata.py
