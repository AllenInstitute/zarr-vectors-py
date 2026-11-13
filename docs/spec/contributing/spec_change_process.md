# Spec change process

This page describes how a change to the Zarr Vectors format is
proposed, what a format change has to update in this repository, and
the standard every spec page follows.

## Terms

**Format change**
: A change to what a store contains or means: a key in the root, level
  or array attributes, an array's layout or cell encoding, a geometry
  type, or a validation rule.

**RFC (request for comments)**
: A GitHub issue, titled with the prefix `[RFC]`, that proposes a
  format change before it is implemented.

**Format version**
: The `zv_version` value in the root `zarr_vectors` block, written from
  `FORMAT_VERSION` in `zarr_vectors/constants.py` (`"0.9.4"`). It has
  the form `X.Y` or `X.Y.Z`.

**Breaking change**
: A format change after which this package cannot read stores written
  before it, or reads them with a different meaning.

**Additive change**
: A format change that adds something optional, such as a new key or a
  new [format capability](../layout/root_metadata.md), and leaves
  existing stores readable with their meaning unchanged.

---

## Introduction

Format changes are discussed in an RFC before code is written, so that
design problems are found while they are cheap to fix and so that
existing stores do not change meaning without notice. Changes that do
not touch the format go straight to a pull request.

---

## Technical reference

### When an RFC is needed

An RFC is needed for:

- adding, renaming or removing a key in the root, level or array
  attributes;
- changing an array's data type, shape, cell encoding or meaning;
- adding a geometry type (see
  [Adding a geometry type](adding_geometry_types.md));
- adding or changing a validation rule, or any change that makes the
  validator give a different result on an existing store;
- changing the `multiscales` block or the `ome` node.

No RFC is needed for bug fixes that leave the format unchanged, new
ingest or export formats, new API functions, performance work that
leaves the stored bytes unchanged, or documentation.

### RFC process

1. **Open an issue** in the
   [`zarr-vectors-py` repository](https://github.com/AllenInstitute/zarr-vectors-py)
   with the title prefix `[RFC]` and these sections:

   ```markdown
   ## Motivation
   What use case needs the change.

   ## Proposed change
   What changes in the format, with example JSON or pseudocode.

   ## Compatibility
   Whether existing stores stay readable, and what happens to them if not.

   ## Alternatives considered
   Other designs and why they were not chosen.
   ```

2. **Discussion.** The issue stays open for at least 14 days for an
   additive change and 30 days for a breaking change.
3. **Acceptance.** The RFC is accepted when at least two maintainers
   approve it in the issue, and is labelled `accepted`.
4. **Implementation pull request** (see below).
5. **Merge** after approval from at least two maintainers and a passing
   test workflow.

### What an implementation pull request updates

- The code that writes and reads the change, under `zarr_vectors/`.
- `FORMAT_VERSION` in `zarr_vectors/constants.py`, and the test that
  pins it (`test_format_version_is_0_9_4` in
  `tests/test_linkml_schema.py`).
- The validator in `zarr_vectors/validate/`, when a rule changes.
- The LinkML schema `schema/zarr_vectors.linkml.yaml`, followed by
  `python schema/regen.py` to regenerate the JSON Schema and Markdown
  reference (see the [machine-readable schema](../schema.md)).
- Tests under `tests/` (see [Compliance testing](test_compliance.md)).
- The spec pages under `docs/spec/` that describe the changed part.

### Breaking changes

`RootMetadata.validate` (`zarr_vectors/core/metadata.py`) refuses a
store whose `zv_version` is older than the oldest layout this package
reads (`0.9.0`) with a `MetadataError` that names the
version. A breaking change moves that bound and must say, in the RFC,
how users rewrite their stores.

A change that only some readers can handle can instead be declared as
a format capability: a store lists it in `format_capabilities`, and in
`required_capabilities` when a reader that does not implement it must
refuse the store. See [Root metadata](../layout/root_metadata.md).

### Pull request conventions

- Link the RFC issue from the pull request.
- Keep format changes and unrelated fixes in separate pull requests.
- The test workflow runs on a pull request from an author who is not
  a repository owner, member or collaborator only after a maintainer
  adds the `ci-approved` label.

### Spec documentation standards

Every page in `docs/spec/` follows the same template:

- **Terms**: defines each term the page uses.
- **Introduction**: what the part of the format is for, written for a
  Python developer new to Zarr Vectors.
- **Technical reference**: JSON examples of the stored metadata, the
  rules, worked examples, and a **Validation** subsection that lists
  the checks that apply and their level, checked against
  `zarr_vectors/validate/`.

Link to related pages with relative Markdown links, describe the format
as it is, and run every code example against the package before
publishing it.

### Contact

To ask whether a change needs an RFC, or to discuss an idea first, open
an issue in the
[`zarr-vectors-py` repository](https://github.com/AllenInstitute/zarr-vectors-py).
