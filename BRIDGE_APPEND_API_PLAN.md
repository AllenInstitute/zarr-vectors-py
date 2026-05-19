# zarr-vectors-py: append-mode write APIs + idempotent creates + soft-fail reads

## Mission

Five small additive changes to `zarr_vectors.core.arrays` that let downstream consumers (BRIDGE in particular) delete ~250 lines of hand-roll, including three process-wide class-level buffer dicts that currently force single-process operation. None of these changes should alter behaviour for existing range-only callers; all are pure additions with default values that preserve the current API.

You can land R1–R5 as five separate PRs or one larger PR — they are independent in implementation and review. R6 is a follow-up for a downstream consumer to re-evaluate after R1 ships and is not a deliverable here.

## Why this matters (consumer context)

BRIDGE (`c:\Users\andre\BRIDGE\`) writes per-chunk concurrent across Dask workers. Per-chunk arrays (`vertices/<chunk>/`, `vertex_fragments/<chunk>/`, etc.) are safe because each worker owns disjoint chunk keys. But store-wide arrays — `object_index/manifests`, `object_attributes/<name>/data`, `cross_chunk_links/<delta>/data`, and the per-chunk `vertex_fragments` / `link_fragments` blobs once Core 1 has already written the canonical range fragment 0 — are not per-chunk: any second `write_*` call replaces the whole blob, clobbering an earlier worker's data.

Today BRIDGE works around this with three patterns:

1. **`_append_explicit_fragments` / `_append_explicit_link_fragments`** — read-modify-write of the `vertex_fragments/<chunk>/` and `link_fragments/<chunk>/` blobs to append explicit-mode entries while preserving existing ones. ~107 lines of glue.
2. **`_global_attr_buffer`** — class-level dict on the writer that accumulates per-chunk attribute values, rewriting the cumulative blob on every flush. O(N²) total bytes written; only works in single-process Dask.
3. **`_global_cross_chunk_link_buffer` + attr buffer** — same pattern for `cross_chunk_links`. ~90 lines + two class-level dicts.

Append-mode write APIs on these primitives would let BRIDGE delete all three. The class-level dicts go away, which also unblocks multi-process Dask.

## Repository orientation

| Path | Role |
|---|---|
| `zarr_vectors/core/arrays.py` | Public API surface — all five changes land here |
| `zarr_vectors/encoding/fragments.py` | `encode_fragments` + `ChunkFragmentIndex` + `decode_fragments` — already supports both range and explicit modes; no changes needed |
| `zarr_vectors/exceptions.py` | `ArrayError` (line 24) — raise from new code |
| `zarr_vectors/constants.py` | `VERTEX_FRAGMENTS`, `LINK_FRAGMENTS`, etc. — string constants for array names |
| `zarr_vectors/core/paths.py` | `links_path`, `cross_chunk_links_path`, etc. |
| `tests/test_arrays.py` | Add tests here (follow existing `TestExplicitFragments` patterns at line 572) |
| `zarr_vectors/ops/edit.py` / `ops/fragments.py` | Existing high-level edit API — **do not modify**. It's session-based and designed for human-editor workflows. The work here is on the low-level `core/arrays.py` writers that BRIDGE uses directly. |

Existing reference: the recent commit that added explicit-mode dispatch on the read side (`read_fragment`, `read_chunk_link_fragment`, `read_chunk_vertices`, `read_chunk_links`) introduced four shared private helpers — `_reshape_link_buffer`, `_slice_link_range`, `_reshape_vertex_buffer`, `_slice_vertex_range` (around `arrays.py:1816`). Mirror that style for the new append helpers.

## Shared internal helper (recommended factoring)

R1, R2, R3 all have the same shape: read-existing → concat → write-back. Factor a private helper near the existing `_read_vertex_offsets` block in `arrays.py`:

```python
def _read_modify_write_blob(
    level_group: FsGroup,
    full_name: str,
    key: str,
    *,
    decode_fn,           # raw_bytes -> existing_value
    merge_fn,            # (existing_value, new_value) -> combined_value
    encode_fn,           # combined_value -> raw_bytes
    initial_value,       # what existing_value is when the blob doesn't yet exist
) -> tuple[Any, Any]:    # (combined_value, existing_value)
    """Atomically read-modify-write a single blob. Returns the merged
    value AND the pre-merge existing value (callers need the latter to
    compute row indices for the appended portion).

    Atomicity: write_bytes is atomic per (full_name, key). The pattern
    is NOT cross-writer-safe — concurrent appends to the same blob will
    race. Callers must enforce single-writer-per-blob serialisation
    (e.g. per-chunk sharding for vertex/link fragment indices, or a
    single rendezvous worker for store-wide arrays).
    """
    try:
        raw = level_group.read_bytes(full_name, key)
        existing = decode_fn(raw)
    except (ArrayError, Exception):
        existing = initial_value
    combined = merge_fn(existing, ...)
    level_group.write_bytes(full_name, key, encode_fn(combined))
    return combined, existing
```

This is optional; each `R` can land independently with its own inline RMW logic, but the helper reduces total code added.

---

## R1 — `write_chunk_fragments` with append mode

### What to build

A new public function `write_chunk_fragments` in `zarr_vectors/core/arrays.py` that writes (or appends to) a chunk's `vertex_fragments/<chunk>/` or `link_fragments/<chunk>/` fragment-index blob. Replaces today's "read existing blob via `read_vertex_fragment_index` → decode to list → concat → re-encode via `encode_fragments` → `write_bytes`" pattern that consumers must implement themselves.

### Signature

```python
def write_chunk_fragments(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
    new_fragments: list,                # mix of (start, count) tuples + int64 ndarrays
    *,
    target: Literal["vertex", "link"] = "vertex",
    mode: Literal["replace", "append"] = "replace",
) -> list[int]:                          # fragment_index values for the new entries
    """Write fragment-index entries to a chunk's vertex_fragments/<chunk>
    or link_fragments/<chunk> blob.

    Args:
        level_group: Resolution level group.
        chunk_coords: Spatial chunk coordinates.
        new_fragments: Mix of ``(start, count)`` range tuples and
            ``np.ndarray[int64]`` explicit index arrays. Order is
            preserved in the output index.
        target: Which fragment-index to target.
        mode: ``"replace"`` writes ``new_fragments`` as the whole blob
            (equivalent to calling ``encode_fragments`` then
            ``write_bytes`` directly). ``"append"`` reads the existing
            blob, decodes its fragments, concatenates ``new_fragments``,
            re-encodes, writes back.

    Returns:
        List of fragment_index values assigned to the newly-written
        entries. In ``"replace"`` mode this is ``list(range(len(new_fragments)))``.
        In ``"append"`` mode this is
        ``list(range(n_existing, n_existing + len(new_fragments)))``.
        Existing fragment_index values are stable.

    Concurrency:
        Atomic per (chunk_coords, target) — ``write_bytes`` is atomic.
        NOT cross-writer-safe; callers must serialise appends to the
        same chunk's fragment-index. Per-chunk sharding satisfies this.
    """
```

### Implementation sketch

```python
from zarr_vectors.constants import VERTEX_FRAGMENTS, LINK_FRAGMENTS
from zarr_vectors.encoding.fragments import encode_fragments, decode_fragments

def write_chunk_fragments(
    level_group, chunk_coords, new_fragments, *,
    target="vertex", mode="replace",
):
    if target == "vertex":
        constant = VERTEX_FRAGMENTS
    elif target == "link":
        constant = LINK_FRAGMENTS
    else:
        raise ArrayError(f"target must be 'vertex' or 'link', got {target!r}")

    key = _chunk_key(chunk_coords)
    new_list = list(new_fragments)

    if mode == "replace":
        level_group.write_bytes(constant, key, encode_fragments(new_list))
        return list(range(len(new_list)))

    if mode != "append":
        raise ArrayError(f"mode must be 'replace' or 'append', got {mode!r}")

    # Append: read existing fragment-index, materialise to a list, concat.
    try:
        raw = level_group.read_bytes(constant, key)
        fi = decode_fragments(raw)
        existing = [
            fi.range(i) if fi.is_range(i) else fi.indices(i)
            for i in range(fi.num_fragments)
        ]
    except (ArrayError, Exception):
        existing = []

    n_before = len(existing)
    if not new_list:
        return []   # no-op append; don't touch the blob
    combined = existing + new_list
    level_group.write_bytes(constant, key, encode_fragments(combined))
    return list(range(n_before, n_before + len(new_list)))
```

### Edge cases

- **Empty existing blob** (chunk doesn't exist yet on `append`): treat as `existing = []`. The first append behaves like a `replace`.
- **Empty `new_fragments`**: no-op; return `[]`; don't write to the blob.
- **Mixed range + explicit in `new_fragments`**: pass through; `encode_fragments` handles it.
- **Invalid `target` or `mode`**: raise `ArrayError` with a descriptive message.

### Tests to add

In `tests/test_arrays.py`, near the existing `TestExplicitFragments` class (line 572):

1. **`test_write_chunk_fragments_vertex_replace`** — `mode="replace"` writes the input as the whole index; readback matches.
2. **`test_write_chunk_fragments_vertex_append`** — Write a range fragment via `write_chunk_vertices`, then call `write_chunk_fragments(..., mode="append")` with `[np.array([1,3,2], dtype=np.int64)]`. Verify `fi.num_fragments == 2`, the original range is at index 0, the new explicit entry is at index 1. Return value is `[1]`.
3. **`test_write_chunk_fragments_link_append`** — Same as above but with `target="link"`.
4. **`test_write_chunk_fragments_append_to_missing`** — Append to a chunk that has no fragment-index blob yet; verify it creates one (`existing` falls back to `[]`).
5. **`test_write_chunk_fragments_append_empty_noop`** — `mode="append"` with `new_fragments=[]` returns `[]` and doesn't modify the blob.
6. **`test_write_chunk_fragments_invalid_target`** — `target="invalid"` raises `ArrayError`.

Pattern these on the existing `TestExplicitFragments._write_vertex_explicit_index` helper (lines 586-595). Use `_make_level_group(tmp_path)` as the fixture.

### What BRIDGE deletes

[`bridge/utils/components.py`](c:\Users\andre\BRIDGE\bridge\utils\components.py) lines 507–615 — both `_append_explicit_fragments` and `_append_explicit_link_fragments` collapse to call-site `write_chunk_fragments(level_group, cc, arrays, target="vertex"|"link", mode="append")`. **~107 lines deleted.**

---

## R2 — `write_object_attributes` with append mode

### What to build

Extend the existing `write_object_attributes` ([arrays.py:669](zarr_vectors/core/arrays.py)) with a `mode` kwarg defaulting to `"replace"` (current behaviour). `mode="append"` reads the existing array, concatenates, writes back.

### Signature change

```python
def write_object_attributes(
    level_group: FsGroup,
    attr_name: str,
    data: npt.NDArray,
    *,
    present_mask: npt.NDArray | None = None,
    mode: Literal["replace", "append"] = "replace",
) -> None:
    """Write dense O×C object attribute data.

    ``mode="replace"`` (default, current behaviour): ``data`` is the
    full dense array. Overwrites any existing array.

    ``mode="append"``: ``data`` is the NEW rows to append. Reads the
    existing array, concatenates along axis 0, writes back. The
    existing array's dtype wins on dtype mismatch (new rows are cast
    to it). Creates the array if it doesn't yet exist (treats existing
    as empty).

    Atomicity: ``write_bytes`` is atomic per (level_group, attr_name).
    Append mode is read-modify-write; callers MUST serialise concurrent
    appends to the same attribute.

    Args (existing):
        level_group: Resolution level group.
        attr_name: Attribute name.
        data: ``(O,)`` or ``(O, C)`` array. In ``"append"`` mode,
            interpreted as new rows to append.
        present_mask: Optional ``(O,)`` byte array (``0``/``1`` per
            object). When ``mode="append"``, this is the present_mask
            for the APPENDED rows only; the helper extends the on-disk
            mask accordingly. When omitted in append mode, the helper
            preserves whatever sidecar exists (no extension).
    """
```

### Implementation sketch

```python
def write_object_attributes(
    level_group, attr_name, data, *,
    present_mask=None, mode="replace",
):
    full_name = f"{OBJECT_ATTRIBUTES}/{attr_name}"

    if mode == "replace":
        # existing body, unchanged
        ...
        return

    if mode != "append":
        raise ArrayError(f"mode must be 'replace' or 'append', got {mode!r}")

    # Append branch: read existing, concatenate, write back.
    try:
        meta = level_group.read_array_meta(full_name)
        existing_dtype = np.dtype(meta["dtype"])
        existing_shape = tuple(meta["shape"])
        raw = level_group.read_bytes(full_name, "data")
        existing = np.frombuffer(raw, dtype=existing_dtype).reshape(existing_shape)
    except (ArrayError, Exception):
        existing = np.empty((0,) + data.shape[1:], dtype=data.dtype)

    new = np.asarray(data).astype(existing.dtype, copy=False)
    combined = np.concatenate([existing, new], axis=0)
    _ensure_array_dir(level_group, full_name)
    level_group.write_bytes(full_name, "data", combined.tobytes())
    level_group.write_array_meta(full_name, {
        "zv_array": "object_attribute",
        "name": attr_name,
        "dtype": str(combined.dtype),
        "shape": list(combined.shape),
        "has_present_mask": bool(present_mask is not None) or bool(meta.get("has_present_mask", False)) if existing.size else bool(present_mask is not None),
    })

    # present_mask handling (extend if caller provided)
    if present_mask is not None:
        # Read existing mask (if any), extend, write back
        ...
```

### Edge cases

- **First append** (attribute doesn't exist): treat as `existing = np.empty((0, ...), dtype=data.dtype)`.
- **dtype mismatch**: cast `data` to `existing.dtype`. Document this. If lossy (e.g. float→int), don't try to be clever — the cast will raise via numpy.
- **Shape mismatch** (existing is `(N, C1)`, new is `(M, C2)` with `C1 != C2`): raise `ArrayError` with a clear message.
- **present_mask in append mode**: the simplest contract is "if you provide present_mask, it's the mask for the new rows only; existing mask is extended". If you don't provide one, leave the on-disk mask as-is. Document.
- **Variable-length object dtype attributes**: out of scope. Document that append mode requires fixed-dtype numeric attributes.

### Tests to add

1. **`test_write_object_attributes_append`** — Write `np.array([1, 2, 3], dtype=np.int32)` with `mode="replace"`. Then append `np.array([4, 5], dtype=np.int32)`. Read back; expect `[1, 2, 3, 4, 5]`.
2. **`test_write_object_attributes_append_to_missing`** — Append `[10, 20]` to a non-existent attribute. Read back; expect `[10, 20]`.
3. **`test_write_object_attributes_append_dtype_cast`** — Existing int32, append float32; result is int32, casts succeed.
4. **`test_write_object_attributes_append_shape_mismatch`** — Existing `(3, 2)`, append `(2, 3)`; raises `ArrayError`.
5. **`test_write_object_attributes_append_2d`** — Existing `(2, 3) float32`, append `(1, 3) float32`; result is `(3, 3)` float32 with rows concatenated correctly.

### What BRIDGE deletes

[`bridge/utils/components.py`](c:\Users\andre\BRIDGE\bridge\utils\components.py) lines 367–413 — `_append_object_attributes_batch` method and `_global_attr_buffer` class-level dict. Call sites switch from `cls._append_object_attributes_batch(level_group, attrs, n_existing=..., store_key=...)` to a loop calling `write_object_attributes(level_group, name, values, mode="append")`. **~50 lines deleted + one class-level buffer dict dropped.**

---

## R3 — `write_cross_chunk_links` and `write_cross_chunk_link_attributes` with append mode

### What to build

Extend the existing `write_cross_chunk_links` ([arrays.py:783](zarr_vectors/core/arrays.py)) and `write_cross_chunk_link_attributes` ([arrays.py:866](zarr_vectors/core/arrays.py)) with `mode` kwargs. Same shape as R2.

### Signature changes

```python
def write_cross_chunk_links(
    level_group: FsGroup,
    links: list[list[tuple[ChunkCoords, int]]] | list[CrossChunkLink],
    sid_ndim: int,
    *,
    delta: int = 0,
    link_width: int | None = None,
    mode: Literal["replace", "append"] = "replace",
) -> int:
    """... existing docstring ...

    Returns:
        In ``"replace"`` mode: 0.
        In ``"append"`` mode: the row index of the first newly-appended
        record (i.e. ``len(existing_records)`` before the append).
        Callers can stamp this row index onto downstream attribute
        tables that reference records by row.
    """


def write_cross_chunk_link_attributes(
    level_group: FsGroup,
    attr_name: str,
    attr_data: npt.NDArray,
    *,
    num_links: int,
    delta: int = 0,
    mode: Literal["replace", "append"] = "replace",
) -> None:
    """... existing docstring ...

    ``mode="append"``: concatenate ``attr_data`` onto the existing
    attribute array. After write, the array length MUST equal
    ``num_links`` (the caller-supplied row count of the post-append
    cross_chunk_links data). Raises ``ArrayError`` if the resulting
    length doesn't match — guards against desynchronisation between
    the records array and its attribute arrays.
    """
```

### Implementation sketch

Use `read_cross_chunk_links` + `read_cross_chunk_link_attributes` to fetch the existing arrays, concatenate, write back via the existing replace path. Return the pre-append count from `write_cross_chunk_links`.

For both functions in append mode: handle "no existing array" as `existing = []` (or empty ndarray for attrs).

### Tests to add

1. **`test_write_cross_chunk_links_append`** — Replace with two records. Append a third. Read all three back; expect the original two first, the new one last. Return value of the append call is `2`.
2. **`test_write_cross_chunk_links_append_to_empty`** — Append two records to a store with no existing cross_chunk_links. Return value is `0`. Read back gets both.
3. **`test_write_cross_chunk_link_attributes_append`** — Same pattern for the attribute table; verify length-sync check raises when `num_links` mismatches.
4. **`test_write_cross_chunk_link_attributes_append_misaligned`** — Append an attribute array but pass `num_links` that doesn't match the post-append length; expect `ArrayError`.

### What BRIDGE deletes

[`bridge/utils/components.py`](c:\Users\andre\BRIDGE\bridge\utils\components.py) lines 375–457 — `_append_cross_chunk_links_to_store` + the two class-level buffer dicts `_global_cross_chunk_link_buffer` and `_global_cross_chunk_link_attr_buffer`. Call sites switch to:

```python
first_new = write_cross_chunk_links(lg, records, sid_ndim=3, mode="append")
for attr_name, values in attrs.items():
    write_cross_chunk_link_attributes(
        lg, attr_name, np.asarray(values),
        num_links=first_new + len(records), mode="append",
    )
```

**~90 lines deleted + two class-level buffers dropped.**

---

## R4 — idempotent `create_*_array` helpers

### What to build

Add `exist_ok: bool = True` to every `create_*_array` function in `zarr_vectors/core/arrays.py`. When `True` (the new default), the function is a no-op if the array already exists with compatible metadata.

### Functions to update

Search `arrays.py` for `^def create_`. The targets are:

- `create_vertices_array` ([arrays.py:154](zarr_vectors/core/arrays.py))
- `create_links_array` ([arrays.py:179](zarr_vectors/core/arrays.py))
- `create_attribute_array` ([arrays.py:210](zarr_vectors/core/arrays.py))
- `create_object_index_array` ([arrays.py:251](zarr_vectors/core/arrays.py))
- `create_object_attributes_array` ([arrays.py:259](zarr_vectors/core/arrays.py))
- `create_groupings_array` ([arrays.py:283](zarr_vectors/core/arrays.py))
- `create_groupings_attributes_array` ([arrays.py:291](zarr_vectors/core/arrays.py))
- `create_cross_chunk_links_array` ([arrays.py:308](zarr_vectors/core/arrays.py))
- `create_link_attributes_array` ([arrays.py:335](zarr_vectors/core/arrays.py))
- `create_cross_chunk_link_attributes_array` ([arrays.py:354](zarr_vectors/core/arrays.py))

### Pattern

```python
def create_vertices_array(
    level_group: FsGroup,
    *,
    dtype: str = "float32",
    exist_ok: bool = True,
) -> None:
    """Create the vertices array. With ``exist_ok=True`` (default), no-op
    if the array already exists; with ``exist_ok=False``, raise
    :class:`ArrayError` on conflict."""
    full_name = VERTICES
    if exist_ok:
        try:
            existing = level_group.read_array_meta(full_name)
            # Optionally: validate dtype matches; raise on mismatch.
            return
        except (ArrayError, Exception):
            pass   # doesn't exist; fall through to create
    # ... existing create logic ...
```

### Notes

- **dtype validation on existing** is optional but recommended — if `exist_ok=True` and the existing array has a different dtype than requested, that's a real bug and should raise. Document this.
- **Don't break callers that depend on the raise** — `exist_ok=True` is the new default; callers who explicitly want the raise can pass `exist_ok=False`. Defaulting to `True` is fine because the current behaviour is dominated by "create once at store init, never recreate" use cases where the raise is just noise.

### Tests to add

1. **`test_create_vertices_array_idempotent`** — Call twice; second is a no-op, no exception.
2. **`test_create_vertices_array_strict`** — Call once. Call again with `exist_ok=False`; expect `ArrayError`.
3. **Spot-check 2-3 other create_* helpers** with the same idempotent pattern; full coverage is mechanical.

### What BRIDGE deletes

`_ensure_array` helper at [`bridge/utils/components.py:168-176`](c:\Users\andre\BRIDGE\bridge\utils\components.py) — 8 lines. Every call site of `_ensure_array(create_X, lg, ...)` unwraps to `create_X(lg, ...)`. ~20 callsite lines simplified.

---

## R5 — `default=…` soft-fail option on `read_fragment` + `read_chunk_link_fragment`

### What to build

Add an optional `default` keyword to both `read_fragment` ([arrays.py:966](zarr_vectors/core/arrays.py)) and `read_chunk_link_fragment` ([arrays.py:1078](zarr_vectors/core/arrays.py)). When supplied, return it on `ArrayError` instead of raising.

### Signatures

```python
_UNSET = object()   # module-level sentinel

def read_fragment(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
    fragment_index: int,
    dtype: np.dtype | str = np.float32,
    ndim: int = 3,
    *,
    default=_UNSET,
) -> npt.NDArray[np.floating] | Any:
    """Read a single vertex fragment from a chunk.

    ... existing docstring ...

    Args:
        ... existing args ...
        default: When supplied, returned on read failure / missing chunk /
            out-of-range fragment_index instead of raising
            :class:`ArrayError`. Use ``None`` to get the common "soft-fail
            with None" semantics.
    """
    try:
        # ... existing body unchanged ...
        return result
    except ArrayError:
        if default is _UNSET:
            raise
        return default


def read_chunk_link_fragment(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
    fragment_index: int,
    dtype: np.dtype | str = np.int64,
    link_width: int | None = None,
    *,
    default=_UNSET,
) -> npt.NDArray[np.integer] | Any: ...
```

### Edge cases

- **Don't swallow non-`ArrayError`** exceptions. Only `ArrayError` triggers the default branch. Programming errors (TypeError, ValueError on bad input) still raise.
- **Sentinel object**: use a module-level `_UNSET = object()` to distinguish "caller didn't pass `default`" from "caller passed `default=None`" (the latter should return `None` on miss, not raise).

### Tests to add

1. **`test_read_fragment_default_on_missing_chunk`** — Call `read_fragment(lg, (99, 99, 99), 0, default=None)` against a store; expect `None`.
2. **`test_read_fragment_default_on_out_of_range`** — Valid chunk, fragment_index out of range, `default=np.empty((0, 3))`; expect the empty array.
3. **`test_read_fragment_no_default_still_raises`** — Same setup but without `default`; expect `ArrayError`.
4. **Same three tests for `read_chunk_link_fragment`**.

### What BRIDGE deletes

This pass already inlined `try: read_fragment(...); except ArrayError: ...` at two call sites in `cpu_smoke.py` and `tests/test_cpu_e2e.py`. Once R5 lands, those inline try/except blocks collapse to a single one-liner each:

```python
arr = read_fragment(lg, cc, 0, dtype=np.float32, ndim=3, default=None)
if arr is None: continue
```

A handful of lines saved per call site.

---

## Acceptance checklist

For each R, before opening the PR:

- [ ] **No regression**: all existing tests in `tests/test_arrays.py` still pass (`pytest tests/test_arrays.py -v`).
- [ ] **No regression on the whole suite**: `pytest tests/` is green.
- [ ] **Range-mode output is byte-identical** to the pre-change implementation in `mode="replace"` paths. Spot-check by writing a fragment with the new function in replace mode and a fragment with the old function (via git checkout) — compare the resulting blob bytes.
- [ ] **New tests cover both modes** (the "happy path" tests in this doc).
- [ ] **Edge cases tested** (empty input, missing existing blob, dtype mismatch, etc. — listed per R).
- [ ] **Docstrings explicit about concurrency** — append modes are NOT cross-writer-safe.

## Style notes

- Follow the existing module's style — 4-space indent, double quotes for strings, type hints on every public function. Use `from __future__ import annotations` at the top of `arrays.py` if not already there.
- Public functions raise `ArrayError` on user-input violations (bad mode, bad target, dtype mismatch). Don't raise `ValueError` from public code — match the existing convention.
- Tests live in `tests/test_arrays.py`. Use the `_make_level_group(tmp_path)` fixture pattern from the top of that file (line 50). Group new tests into classes named `TestWriteChunkFragments`, `TestWriteObjectAttributesAppend`, etc. — match the existing class-per-feature convention.

## PR sequencing recommendation

1. **R1 first** — highest leverage (deletes 107 lines downstream), structurally simple, mostly mechanical given `encode_fragments` already supports the mixed input.
2. **R2 second** — independent of R1. Same shape; mostly mechanical.
3. **R3 third** — independent. Can ride a shared internal helper alongside R2.
4. **R4 fourth** — mechanical sweep across 10 functions.
5. **R5 last** — lowest impact; smallest change.

Each PR can land independently. Consider whether to factor the shared `_read_modify_write_blob` helper (proposed at the top of this doc) when R3 lands — by then the pattern has appeared three times.

## Out of scope

- **`ops/edit.py` / `ops/fragments.py`** session-based APIs — these target a different use case (interactive edits with manifest propagation). Don't touch.
- **Cross-level (delta != 0) operations** for cross_chunk_links — BRIDGE doesn't use them; keep existing behaviour.
- **Variable-length object_attributes** (string / object dtype attrs in append mode) — defer until a consumer needs it.
- **Cross-writer transactional safety** — out of scope for this lib. Document the per-blob single-writer contract loudly; downstream coordinates.
