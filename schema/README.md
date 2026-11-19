# `schema/`: machine-readable reference

The LinkML model and JSON Schema of the
[Zarr Vectors convention] v0.1.0, the metadata of format 0.10.0, for
tools that do not use the Python package. The model is maintained in
[`convention/` of AllenInstitute/zarr_vectors][upstream], beside the
format specification; this directory holds a copy. The convention
repository publishes the same generated schema at its `schema_url`, and
that published schema is the normative definition. The runtime source of
truth for what this package writes is [`zarr_vectors/core/metadata.py`].

| File | Status | Purpose |
|---|---|---|
| `zarr_vectors.linkml.yaml` | copied | [LinkML] model of one node's `zarr.json`. Edit it upstream, not here. |
| `build.py` | copied | The upstream generator; `regen.py` calls its `build()`. |
| `zarr_vectors.schema.json` | generated | JSON Schema (Draft 2020-12): the convention's schema. |
| `reference.md` | generated, not committed | Markdown reference; `regen.py` writes it locally. |
| `regen.py` | hand-edited | Runs the generators. |

## Updating

When the model changes upstream, copy `convention/zarr_vectors.linkml.yaml`
and `convention/build.py` here, then, from the repository root with the
`dev` extra installed (`pip install "zarr-vectors[dev]"`):

```bash
python schema/regen.py            # regenerate both files in place
python schema/regen.py --check    # exit 1 if the JSON Schema is stale
```

`--check` compares only the JSON Schema, byte for byte. `reference.md`
is not committed: its row order is not fixed, so it could not be
checked, and the convention's README and the LinkML model are the
readable reference. The test workflow
runs `--check`. The generator is pinned through `uv.lock` (LinkML
1.12.1, as upstream), so the output matches upstream byte for byte.

`tests/test_linkml_schema.py` checks that the generated schema equals the
convention's published schema (by a hash of its canonical JSON,
`PUBLISHED_SHA256`), that the metadata the writers emit validates against the class
for its `kind`, and that the `VALID_*` constant sets match the schema
enums.

## Coverage

The schema validates a whole `zarr.json` (`#/$defs/ZarrNode`, the
document root): the `zarr_conventions` registrations, the root's
`multiscales` layout, and the `zarr_vectors` object, which must match
exactly one kind class (`RootMetadata`, `LevelMetadata`,
`VerticesMetadata`, `LinksMetadata`, …). Other attributes, such as
`proj:projjson`, are allowed but not checked. `geometry_types` is a list
of free strings, so a store with a newer geometry type still validates;
`RootMetadata.validate` checks the names this package knows. Rules that
span several nodes (axis counts, bin divisibility, the `parent_level`
chain, link offsets) are not checked by the schema; see the convention
README.

## Validating from other languages

Validate each node's `zarr.json` against the schema as it is:

```python
import json

import jsonschema

schema = json.load(open("schema/zarr_vectors.schema.json"))
node = json.load(open("my_store.zarrvectors/zarr.json"))
jsonschema.Draft202012Validator(schema).validate(node)
```

To check only a `zarr_vectors` object, validate it against its class,
for example `{"$schema": schema["$schema"], "$defs": schema["$defs"],
"$ref": "#/$defs/RootMetadata"}`.

[LinkML]:                       https://linkml.io/
[Zarr Vectors convention]:      https://github.com/Andrew-Keenlyside/zarr-vectors
[upstream]:                     https://github.com/AllenInstitute/zarr_vectors/tree/main/convention
[`zarr_vectors/core/metadata.py`]: ../zarr_vectors/core/metadata.py
