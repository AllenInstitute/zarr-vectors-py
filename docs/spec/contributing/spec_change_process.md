# Spec change process

## Terms

**RFC (Request for Comments)**
: A structured discussion issue proposing a change to the Zarr Vectors specification.
  An RFC is required before opening a PR that changes the format
  (root / level attribute schemas, array schemas, validation rules, or geometry type
  definitions). Code-only changes (bug fixes, new ingest formats, new
  export targets) do not require an RFC.

**Breaking change**
: A spec change that makes existing valid stores invalid, or that changes
  the semantics of existing keys in a backward-incompatible way. Breaking
  changes increment the format version. So far (all `0.x`) they have
  incremented the **minor** component, shipped **no compatibility shim**,
  and told users to rewrite stores from source (see the version history in
  the docstring of `FORMAT_VERSION` in `zarr_vectors/constants.py`: for
  example 0.8.x → 0.9.0).

**Additive change**
: A spec change that adds new *optional*, backward-compatible keys without
  affecting the validity or reading of existing stores. Historically these
  have incremented the **patch** component (0.9.0 → 0.9.1 added
  `attribute_specs`; 0.9.1 → 0.9.2 added the `ome` block; a missing
  block means the prior behaviour).

**Spec version**
: The value of `zv_version` in the root `zarr_vectors` attribute block,
  equal to `FORMAT_VERSION` in `zarr_vectors/constants.py`. Current
  version: `"0.9.2"`. It is `MAJOR.MINOR.PATCH`. A reader rejects stores
  older than 0.9.0 (`RootMetadata.validate`). The release workflow refuses
  to publish a release whose tag differs from `FORMAT_VERSION`.

---

## Introduction

The Zarr Vectors specification is a living document. The format will evolve as new
use cases emerge, as implementation experience reveals design issues, and
as the broader Zarr and OME-Zarr ecosystems change. This page documents
the process for proposing, discussing, and ratifying specification changes.

The process is designed to balance agility (changes should be possible without
months of committee process) with stability (existing stores should not
silently break after a spec update). The RFC step ensures that non-trivial
changes are discussed before implementation, catching design issues early.

---

## Technical reference

### When an RFC is required

**RFC required:**
- Adding, renaming, or removing a required or optional key in the root
  `zarr_vectors` / `multiscales` / `ome` blocks or the per-level
  `zarr_vectors_level` block.
- Changing the dtype, shape, or semantics of any array defined in the spec.
- Adding a new geometry type.
- Adding or modifying validation rules.
- Changing the coordinate transform encoding or multiscale metadata schema.
- Any change that would cause existing validators (L1–L5) to produce
  different results on existing stores.

**RFC not required (direct PR acceptable):**
- Bug fixes in `zarr-vectors-py` that do not change the format.
- New ingest formats (TRK, TRX, LAS variants, etc.).
- New export targets.
- Documentation improvements (including this spec).
- New CLI commands or API convenience functions.
- Performance improvements that do not change the on-disk format.

### RFC process

**Step 1 — Open a discussion issue.**

Open an issue in the `zarr-vectors-py` GitHub repository with title prefix
`[RFC]`. The issue body should include:

```markdown
## Motivation
Why is this change needed? What use case does it address?

## Proposed change
What exactly should change in the spec? Include proposed JSON examples,
schema diffs, or pseudocode.

## Backward compatibility
Is this a breaking change? If so, how will existing stores be migrated?

## Alternatives considered
What other approaches were considered and why were they rejected?
```

**Step 2 — Discussion period.**

The discussion period is a minimum of **14 days** for additive changes
and **30 days** for breaking changes. During this period, maintainers and
community members may propose modifications or raise concerns.

**Step 3 — Ratification.**

An RFC is ratified when at least two maintainers have explicitly approved
it (GitHub 👍 reaction or comment). Once ratified, the RFC is labelled
`accepted` and a PR may be opened.

**Step 4 — Implementation PR.**

The implementation PR must:
- Update the relevant spec pages in `docs/spec/`.
- Update the validator to reflect new rules (`zarr_vectors/validate/`).
- Update `schema/zarr_vectors.linkml.yaml` and re-run `python schema/regen.py`.
- Bump `FORMAT_VERSION` and add a note to its docstring if the format changes.
- Add or update reference fixtures (see
  [Compliance testing](test_compliance.md)).
- Update the write functions to produce conforming stores.
- Describe the change in the PR (the repository has no changelog file;
  the `FORMAT_VERSION` docstring is the format history).

**Step 5 — Merge.**

The PR requires approval from at least two maintainers and passing CI
(all tests, including compliance tests against reference fixtures).

### Breaking changes and migration

When a breaking change is ratified, the format version is incremented
(minor, while the format is `0.x`). The implementation PR should also:

- Make the validator or reader produce a clear, actionable error for
  stores at the old version (as `RootMetadata.validate` does for
  pre-0.9.0 stores: "no backwards-compat shim ... Rewrite from source").
- Document the migration path in the PR and in the `FORMAT_VERSION`
  docstring. For additive changes an in-place upgrade is preferable where
  possible (for example `zarr_vectors.building.stamp_ome_node` brings an
  older store up to the 0.9.2 `ome` block without rewriting data).

A store at an older version is not required to be readable by a newer
implementation without explicit migration. This repository currently has
no `zarr-vectors migrate` command and no `docs/how_to/migrate_*` pages.

### PR etiquette

- Reference the RFC issue number in the PR description.
- Keep format changes and code changes in separate commits for clarity.
- Do not mix spec changes with unrelated bug fixes in the same PR.
- Respond to review comments within a reasonable time (one week is a
  reasonable expectation for active contributors).
- If a PR is stalled (no activity for 30 days), a maintainer may close it
  with an invitation to reopen when the contributor is available.

### Spec documentation standards

All spec pages follow the template documented in the
[spec overview](../index.md): Terms → Introduction → Technical reference.
When adding a new spec page:

- Define all terms used on the page in the Terms section.
- Write the Introduction at a level accessible to a competent Python
  developer who is new to Zarr Vectors.
- Include full JSON examples, pseudocode, or worked numerical examples
  in the Technical reference.
- Add a Validation subsection listing which checks apply and at which level.
- Cross-link to related pages using relative Markdown links.

### Contacting the maintainers

For questions about whether a change requires an RFC, or to discuss a
potential contribution before writing a full RFC, open a
[GitHub Discussion](https://github.com/AllenInstitute/zarr-vectors-py/discussions)
in the "Ideas" or "Q&A" category.
