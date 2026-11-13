# Machine-readable schema

This page points to the machine-readable description of the Zarr
Vectors metadata, for validating stores from any language with a JSON
Schema validator. The files live in the `schema/` directory at the
repository root:

- `schema/zarr_vectors.linkml.yaml`: the [LinkML](https://linkml.io/)
  source, edited by hand.
- `schema/zarr_vectors.schema.json`: the JSON Schema generated from it.
- `schema/reference.md`: a Markdown reference generated from it.

The Python source of truth is the dataclasses in
`zarr_vectors/core/metadata.py`. `tests/test_linkml_schema.py` checks
that metadata written by the package validates against the generated
JSON Schema, and the test workflow runs `python schema/regen.py --check`
to catch a stale JSON Schema.

`schema/README.md` describes what the schema covers, how stored
attributes map onto it, and how to regenerate it. All four files are in
the [`schema/` directory on GitHub](https://github.com/AllenInstitute/zarr-vectors-py/tree/main/schema).
