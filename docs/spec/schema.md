# Machine-readable schema

The Zarr Vectors metadata is also described as a schema, so stores can
be validated from any language with a JSON Schema validator. The files
are in the [`schema/` directory](https://github.com/AllenInstitute/zarr-vectors-py/tree/main/schema):

| File | Content |
|---|---|
| `zarr_vectors.linkml.yaml` | The [LinkML](https://linkml.io/) model, copied from [`convention/` in AllenInstitute/zarr_vectors](https://github.com/AllenInstitute/zarr_vectors/tree/main/convention), where it is maintained. |
| `zarr_vectors.schema.json` | JSON Schema generated from it. |
| `reference.md` | Markdown reference generated from it. |
| `README.md` | Coverage, how to validate a node, and how to update the copy. |

The generated JSON Schema is the schema the
[Zarr Vectors convention](https://github.com/Andrew-Keenlyside/zarr-vectors)
publishes. It validates one node's whole `zarr.json`: the convention
registrations, the root's `multiscales` layout and the `zarr_vectors`
object, which must match exactly one `kind`. It does not check rules
that span several nodes, or `proj:projjson` and the `ome` node; see
[Root metadata](layout/root_metadata.md) for those.

The source of truth for what the package writes is the dataclasses in
`zarr_vectors/core/metadata.py`. `tests/test_linkml_schema.py` checks
that written metadata validates against the JSON Schema and that the
generated schema equals the convention's published one;
`tests/test_convention_conformance.py` validates every node of stores the
writers produce. The test workflow runs `python schema/regen.py --check`
to catch a stale schema.
