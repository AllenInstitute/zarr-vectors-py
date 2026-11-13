# `schema/`: machine-readable reference

A schema for the Zarr Vectors on-disk metadata, for tools that do not
use the Python package. The source of truth is
[`zarr_vectors/core/metadata.py`]; these files describe the same
metadata in [LinkML] and JSON Schema.

| File | Status | Purpose |
|---|---|---|
| `zarr_vectors.linkml.yaml` | hand-edited | [LinkML] source. Edit this to change the schema. |
| `zarr_vectors.schema.json` | generated | JSON Schema (Draft 2019-09), for downstream validators. |
| `reference.md`             | generated | Markdown reference. |
| `regen.py`                 | hand-edited | Runs the generators. |

## Regenerating

From the repository root, with the `dev` extra installed
(`pip install "zarr-vectors[dev]"`):

```bash
python schema/regen.py            # regenerate both files in place
python schema/regen.py --check    # exit 1 if the JSON Schema is stale
```

`--check` compares only the JSON Schema, byte for byte; `reference.md`
is not compared because its row order is not fixed. The test workflow
runs `--check`. If it fails, run `regen.py` and commit the result.

## Coverage

| Schema item | Source |
|---|---|
| `RootMetadata` | `RootMetadata`: the `zarr_vectors` block in the root attributes |
| `LevelMetadata` | `LevelMetadata`: the `zarr_vectors_level` block in each level's attributes |
| 11 per-array classes (`VerticesMeta` to `LinkAttributeMeta`) | array and group attributes written by `zarr_vectors/core/arrays.py`; discriminator slot `zv_array` |
| Enums | `zarr_vectors.constants` `VALID_*` sets and `CAP_*` tokens |

The parametric object registry (`ParametricTypeDef`) is not modelled.

The schema maps three items to outside vocabularies: `Axis` →
`ngff:Axis` ([OME-Zarr NGFF] 0.4), `zv_version` → `schema:version` and
`crs` → `schema:coordinateReferenceSystem` ([Schema.org]).

## Stored form and schema form

The schema models the payload inside the `zarr_vectors` or
`zarr_vectors_level` key. Three slots are shaped differently:

| Slot | Stored | Schema |
|---|---|---|
| `bounds` | `[[min...], [max...]]` | `BoundingBox` with `min_corner`, `max_corner` |
| `shard_shape` | an integer when every axis agrees | a list |
| `crs` | `null` or any value | the open `CRS` class |

`_to_linkml_logical_form` in `tests/test_linkml_schema.py` is the
reference conversion.

## Validating from other languages

Every class is under `$defs`. To validate a root block, read
`zarr_vectors` from the root `zarr.json` attributes, convert it as
above, and validate against `#/$defs/RootMetadata`:

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

The schema `$id` is `https://w3id.org/zarr-vectors/schema/0.5`. It
names the schema only; the format version is `zv_version` in the root
attributes.

[LinkML]:                       https://linkml.io/
[OME-Zarr NGFF]:                https://ngff.openmicroscopy.org/
[Schema.org]:                   https://schema.org/
[`zarr_vectors/core/metadata.py`]: ../zarr_vectors/core/metadata.py
