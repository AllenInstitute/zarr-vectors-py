# Spec change process

How to propose a change to the Zarr Vectors format, what the change must
update in this repository, and the template every spec page follows.

## Terms

**Format change**
: A change to what a store contains or means: a key in the root, level
  or array attributes, an array's layout or cell encoding, a geometry
  type, or a validation rule.

**RFC**
: A GitHub issue titled with the prefix `[RFC]` that proposes a format
  change before it is implemented.

**Breaking change**
: A format change after which this package cannot read existing stores,
  or reads them with a different meaning.

**Additive change**
: A format change that adds something optional, such as a key or a
  format capability, and leaves existing stores unchanged in meaning.

## Introduction

A format change is agreed in an RFC before code is written, so
existing stores do not change meaning without notice. Other changes go
straight to a pull request. If unsure, open an issue and ask.

## Technical reference

### When an RFC is needed

| RFC needed | No RFC needed |
|---|---|
| Adding, renaming or removing a root, level or array attribute key | Fixes that leave the format unchanged |
| Changing an array's data type, shape, cell encoding or meaning | New ingest or export formats |
| Adding a geometry type (see [Adding a geometry type](adding_geometry_types.md)) | New API functions |
| Adding or changing a validation rule, or anything that changes the validator's result on an existing store | Performance work that leaves the stored bytes unchanged |
| Changing the `multiscales` block or the `ome` node | Documentation |

### Process

1. **Open an issue** in the
   [`zarr-vectors-py` repository](https://github.com/AllenInstitute/zarr-vectors-py)
   titled `[RFC] ...`, with the sections **Motivation**, **Proposed
   change** (with example JSON or pseudocode), **Compatibility** (are
   existing stores still readable?) and **Alternatives considered**.
2. **Discuss.** The issue stays open for at least 14 days for an
   additive change, 30 days for a breaking change.
3. **Accept.** Two maintainers approve in the issue, and it is labelled
   `accepted`.
4. **Implement** in a pull request that links the RFC and holds no
   unrelated changes.
5. **Merge** after two maintainer approvals and a passing test
   workflow. For an author who is not a repository owner, member or
   collaborator, the workflow runs only after a maintainer adds the
   `ci-approved` label.

The implementation pull request updates:

- the code under `zarr_vectors/`;
- `FORMAT_VERSION` in `zarr_vectors/constants.py` (written to
  `zv_version` in the root attributes) and the test that pins it
  (`test_format_version_is_0_9_4` in `tests/test_linkml_schema.py`);
- the validator in `zarr_vectors/validate/`, when a rule changes;
- `schema/zarr_vectors.linkml.yaml`, then `python schema/regen.py` (see
  [Machine-readable schema](../schema.md));
- tests (see [Compliance testing](test_compliance.md));
- the affected pages under `docs/spec/`.

### Breaking changes and capabilities

`RootMetadata.validate` (`zarr_vectors/core/metadata.py`) raises
`MetadataError` for a store whose `zv_version` is older than `0.9.0`,
the oldest layout this package reads. A breaking change moves that
bound, and its RFC must say how users rewrite their stores.

A change that only some readers can handle can be a format capability
instead. The store lists it in `format_capabilities`, and also in
`required_capabilities` when a reader without it must refuse the store.
See [Root metadata](../layout/root_metadata.md).

### Spec documentation standards

Every page in `docs/spec/` has three parts, in this order:

1. **Terms**: defines each term the page uses.
2. **Introduction**: what this part of the format is for, for a Python
   developer new to Zarr Vectors.
3. **Technical reference**: JSON examples, rules and worked examples,
   ending in a **Validation** subsection that lists the checks and
   their level, checked against `zarr_vectors/validate/`.

Pages use relative links and run every code example against the
package.
