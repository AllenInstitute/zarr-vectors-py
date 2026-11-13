# `schema/`: machine-readable reference

This directory holds a machine-readable schema for the Zarr Vectors
on-disk metadata, for tools and languages that do not use the Python
package. The Python source of truth is
[`zarr_vectors/core/metadata.py`]; the files here describe the same
metadata in [LinkML] and JSON Schema.

## Files

| File | Status | Purpose |
|---|---|---|
| `zarr_vectors.linkml.yaml` | hand-edited | [LinkML] source. Edit this to change the schema. |
| `zarr_vectors.schema.json` | generated | JSON Schema (`$schema` Draft 2019-09). Downstream validators consume this. |
| `reference.md`             | generated | Markdown reference rendered from the LinkML source. |
| `regen.py`                 | hand-edited | Command that runs the generators. |

## Regenerating

Install the development extra, which brings in `linkml>=1.7,<2` and
`jsonschema>=4.0,<5`:

```bash
pip install "zarr-vectors[dev]"
```

Then, from the repository root:

```bash
python schema/regen.py            # regenerate both artefacts in place
python schema/regen.py --check    # exit 1 if the JSON Schema is stale
```

`--check` regenerates only the JSON Schema and compares it byte for
byte with the committed file; it prints `schema artefacts up to date`
on success. `reference.md` is not compared, because the LinkML
Markdown generator does not emit its class rows in a fixed order. The
test workflow (`.github/workflows/tests.yml`) runs `--check` on every
push to `main` and every pull request; if it fails, run `regen.py`
without `--check` and commit the regenerated files.

## Coverage

| Schema item | Source |
|---|---|
| `RootMetadata` | `zarr_vectors.core.metadata.RootMetadata` (the `zarr_vectors` block in the root attributes) |
| `LevelMetadata` | `zarr_vectors.core.metadata.LevelMetadata` (the `zarr_vectors_level` block in each level's attributes) |
| 11 per-array classes (`VerticesMeta` to `LinkAttributeMeta`) | the array and group attributes written by `zarr_vectors/core/arrays.py`; discriminator slot `zv_array` (enum `ZvArrayTag`) |
| Enums | `zarr_vectors.constants` `VALID_*` sets and `CAP_*` tokens |

The parametric object registry (`ParametricTypeDef` and the built-in
`PARAMETRIC_PLANE`, `PARAMETRIC_LINE` and `PARAMETRIC_SPHERE` in
`zarr_vectors.core.metadata`) is not modelled in the schema.

## Representation bridge

Some slots are shaped differently in the LinkML model than in the
stored attributes:

- **`bounds`**: stored as the two-element list `[[min...], [max...]]`;
  modelled as a `BoundingBox` class with `min_corner` and `max_corner`.
- **`shard_shape`**: stored as a single integer when every axis agrees;
  modelled as a list.
- **`crs`**: stored as `null` or a free-form value; modelled as the open
  `CRS` class, so any value validates.

The stored blocks are wrapped in a `zarr_vectors` or
`zarr_vectors_level` key; the schema models the inner payload.
`_to_linkml_logical_form` in `tests/test_linkml_schema.py` performs
this conversion and is the reference for it.

## Vocabulary mappings

The schema declares prefixes for [OME-Zarr NGFF] (`ngff:`, version 0.4)
and [Schema.org] (`schema:`) and maps three items:

- `Axis` → `ngff:Axis`
- `zv_version` → `schema:version`
- `crs` → `schema:coordinateReferenceSystem`

## Validating from other languages

The generated JSON Schema is the contract for consumers outside Python.
Every class is under `$defs`. To validate a root block, read the
`zarr_vectors` entry from the root `zarr.json` attributes, apply the
bridge above, and validate it against `#/$defs/RootMetadata`:

```python
import json

import jsonschema

schema = json.load(open("schema/zarr_vectors.schema.json"))
root = json.load(open("my_store.zarrvectors/zarr.json"))
block = dict(root["attributes"]["zarr_vectors"])

lo, hi = block["bounds"]
block["bounds"] = {"min_corner": lo, "max_corner": hi}
if isinstance(block.get("shard_shape"), int):
    block["shard_shape"] = [block["shard_shape"]]

jsonschema.validate(
    block,
    {"$schema": schema["$schema"], "$defs": schema["$defs"],
     "$ref": "#/$defs/RootMetadata"},
)
```

## Identifier

The schema's `id` (the JSON Schema `$id`) is
`https://w3id.org/zarr-vectors/schema/0.5`. It identifies the schema
for downstream tools; no code in this package reads it, and it is not
the format version, which is `zv_version` in the root attributes.

[LinkML]:                       https://linkml.io/
[OME-Zarr NGFF]:                https://ngff.openmicroscopy.org/
[Schema.org]:                   https://schema.org/
[`zarr_vectors/core/metadata.py`]: ../zarr_vectors/core/metadata.py
