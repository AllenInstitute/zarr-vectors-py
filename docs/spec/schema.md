# Machine-readable schema

The Zarr Vectors metadata is also described as a schema, so stores can
be validated from any language with a JSON Schema validator. The files
are in the [`schema/` directory](https://github.com/AllenInstitute/zarr-vectors-py/tree/main/schema):

| File | Content |
|---|---|
| `zarr_vectors.linkml.yaml` | The [LinkML](https://linkml.io/) source, edited by hand. |
| `zarr_vectors.schema.json` | JSON Schema generated from it. |
| `reference.md` | Markdown reference generated from it. |
| `README.md` | Coverage, how stored attributes map onto the schema, and how to regenerate it. |

The source of truth is the dataclasses in
`zarr_vectors/core/metadata.py`. `tests/test_linkml_schema.py` checks
that written metadata validates against the JSON Schema. The test
workflow runs `python schema/regen.py --check` to catch a stale schema.
