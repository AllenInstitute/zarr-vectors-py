# Machine-readable schema

The narrative spec pages in `docs/spec/` describe the Zarr Vectors
on-disk format in prose.  For a machine-readable reference — usable
from any language with a JSON Schema validator — see the [`schema/`]
directory in the repository root:

- [`schema/zarr_vectors.linkml.yaml`] — the canonical [LinkML] source.
- [`schema/zarr_vectors.schema.json`] — generated JSON Schema (Draft
  2020-12).
- [`schema/reference.md`] — generated Markdown reference.

The runtime Python source of truth is the dataclass tree in
[`zarr_vectors/core/metadata.py`] (`RootMetadata`, `LevelMetadata`).  A pytest
sanity check (`tests/test_linkml_schema.py`) prevents the LinkML schema from
drifting from the runtime dataclasses.

What the schema models, and what it does not:

- It describes the **payload** of the root `zarr_vectors` attribute and of each
  level's `zarr_vectors_level` attribute, not the envelope: validate
  `attributes["zarr_vectors"]`, not the whole attributes block. Two
  representation differences are bridged in the test: `bounds` is a pair of
  lists on disk but a `BoundingBox` object in the schema, and `crs` is
  free-form.
- It covers the per-array attribute shapes discriminated by the `zv_array`
  attribute (`vertices`, `links`, `links_family`, `attribute`, …).
- The root class models the `zarr_vectors` payload; the axes live in the
  NGFF `multiscales` block and the RFC 8 `ome` node is a third root attribute —
  see [Root metadata](layout/root_metadata.md) for all three. The schema
  has no class for the `ome` node or for the parametric type registry.
- The schema's `$id` carries a nominal version number that is not tied to
  `zv_version` and is read by no code.

For details on coverage, vocabulary mappings (OME-Zarr / Schema.org),
and the build-time regeneration workflow, see
[`schema/README.md`].

[`schema/`]:                            https://github.com/AllenInstitute/zarr-vectors-py/tree/main/schema
[`schema/zarr_vectors.linkml.yaml`]:    https://github.com/AllenInstitute/zarr-vectors-py/blob/main/schema/zarr_vectors.linkml.yaml
[`schema/zarr_vectors.schema.json`]:    https://github.com/AllenInstitute/zarr-vectors-py/blob/main/schema/zarr_vectors.schema.json
[`schema/reference.md`]:                https://github.com/AllenInstitute/zarr-vectors-py/blob/main/schema/reference.md
[`schema/README.md`]:                   https://github.com/AllenInstitute/zarr-vectors-py/blob/main/schema/README.md
[`zarr_vectors/core/metadata.py`]:      ../../zarr_vectors/core/metadata.py
[LinkML]:                               https://linkml.io/
