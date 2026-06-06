"""In-place migration: v0.7 monolithic CCL blobs → v0.8 sharded kN cells.

The v0.7 layout stored every cross-chunk-link record in a single
``cross_chunk_links/<delta>/data`` int64 blob with chunk coordinates
baked into every record.  v0.8 partitions records into K-separated
sharded vlen-bytes zarr arrays (``cross_chunk_links/<delta>/kK``)
keyed by the sorted unique chunks each record touches; per-record
encoding drops from ``L * (sid_ndim + 1) * 8`` bytes to ``9 * L``
bytes.

This module ships a one-shot, idempotent migration helper that
opens a v0.7 store via raw zarr (bypassing the v0.8 metadata
guard), rewrites every CCL blob plus its parallel attribute blob,
deletes the legacy blobs, and bumps ``zv_version`` to ``"0.8.0"``.

Run once per v0.7 store; subsequent invocations are no-ops thanks
to the ``partitioned_cross_chunk_links`` capability guard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import zarr

from zarr_vectors.constants import (
    CAP_MULTISCALE_LINKS,
    CAP_PARTITIONED_CROSS_CHUNK_LINKS,
    CROSS_CHUNK_LINK_ATTRIBUTES,
    CROSS_CHUNK_LINKS,
    FORMAT_VERSION,
)
from zarr_vectors.core.arrays import (
    write_cross_chunk_link_attributes,
    write_cross_chunk_links,
)
from zarr_vectors.core.group import Group
from zarr_vectors.core.paths import (
    cross_chunk_link_attributes_path,
    cross_chunk_links_path,
    format_delta,
)
from zarr_vectors.exceptions import StoreError


def partition_legacy_cross_chunk_links(
    store_path: str | Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Rewrite v0.7 CCL blobs into v0.8 sharded ``kN`` cells in place.

    Walks every resolution level and every ``cross_chunk_links/<delta>/``
    group, decoding the legacy ``data`` blob via the pre-0.8 endpoint
    encoding (``L * (sid_ndim + 1) * 8`` bytes per record, chunk coords
    inline) and re-writing the records via :func:`write_cross_chunk_links`
    so they land in the v0.8 ``kK`` sharded vlen-bytes arrays.  Each
    parallel ``cross_chunk_link_attributes/<name>/<delta>/data`` blob is
    re-emitted into the matching ``kN`` attribute arrays in the same
    pass.

    After every level migrates cleanly, the helper stamps
    ``CAP_PARTITIONED_CROSS_CHUNK_LINKS`` (alongside
    ``CAP_MULTISCALE_LINKS``) on root ``format_capabilities`` and bumps
    ``zv_version`` to ``"0.8.0"``.  Stores already advertising
    ``CAP_PARTITIONED_CROSS_CHUNK_LINKS`` are recognised and skipped
    (idempotent guard).

    The migration is destructive: the legacy ``data`` blob is removed
    once the new layout is in place.  Back up the store first if you
    need a recoverable snapshot.

    Args:
        store_path: Path to the v0.7 store root.  Must be a local
            filesystem path — non-FS backends are not supported by
            this helper (rebuild from source via the new writers
            instead).
        dry_run: If ``True``, scan the store and report what would
            change without writing anything.

    Returns:
        Summary dict::

            {
                "store_path": "...",
                "already_v08": bool,        # nothing to do
                "dry_run": bool,
                "level_results": [
                    {
                        "level": int,
                        "deltas": [
                            {
                                "delta": int,
                                "record_count": int,
                                "cell_count": int,
                                "kK_distribution": {K: count},
                                "attributes": ["weight", ...],
                            },
                            ...
                        ],
                    },
                    ...
                ],
                "version_bumped_to": "0.8.0" | None,
            }

    Raises:
        StoreError: If the store cannot be opened at the zarr level or
            if a legacy blob fails to decode under the assumed
            ``link_width`` / ``sid_ndim``.
    """
    store_path = Path(store_path)
    if not store_path.exists():
        raise StoreError(f"store path does not exist: {store_path}")

    # Open at the bare-zarr level: the v0.8 RootMetadata.validate()
    # would refuse a v0.7 store, so we sidestep it here.
    try:
        root_zg = zarr.open_group(store=str(store_path), mode="r+")
    except Exception as e:
        raise StoreError(f"cannot open zarr group at {store_path}: {e}") from e

    root_attrs = dict(root_zg.attrs)
    zv = dict(root_attrs.get("zarr_vectors") or {})
    caps = list(zv.get("format_capabilities") or [])

    summary: dict[str, Any] = {
        "store_path": str(store_path),
        "already_v08": False,
        "dry_run": bool(dry_run),
        "level_results": [],
        "version_bumped_to": None,
    }
    if CAP_PARTITIONED_CROSS_CHUNK_LINKS in caps:
        summary["already_v08"] = True
        return summary

    # Walk levels.
    level_indices = _list_levels(root_zg)
    for li in level_indices:
        level_result = _migrate_level(
            root_zg, li, dry_run=dry_run,
        )
        if level_result is not None:
            summary["level_results"].append(level_result)

    if dry_run:
        return summary

    # Bump version + stamp capabilities.
    if CAP_MULTISCALE_LINKS not in caps:
        caps.append(CAP_MULTISCALE_LINKS)
    if CAP_PARTITIONED_CROSS_CHUNK_LINKS not in caps:
        caps.append(CAP_PARTITIONED_CROSS_CHUNK_LINKS)
    zv["format_capabilities"] = caps
    zv["zv_version"] = FORMAT_VERSION
    root_attrs["zarr_vectors"] = zv
    root_zg.attrs.update(root_attrs)
    summary["version_bumped_to"] = FORMAT_VERSION

    return summary


def _list_levels(root_zg: zarr.Group) -> list[int]:
    levels: list[int] = []
    for name in root_zg:
        try:
            levels.append(int(name))
        except ValueError:
            continue
    return sorted(levels)


def _migrate_level(
    root_zg: zarr.Group,
    level: int,
    *,
    dry_run: bool,
) -> dict[str, Any] | None:
    """Migrate every legacy CCL blob under one resolution level."""
    try:
        level_zg = root_zg[str(level)]
    except KeyError:
        return None

    if CROSS_CHUNK_LINKS not in level_zg:
        return None
    ccl_root = level_zg[CROSS_CHUNK_LINKS]

    deltas = _list_legacy_deltas(ccl_root)
    if not deltas:
        return None

    level_group = Group._from_zarr(level_zg)
    out: dict[str, Any] = {"level": level, "deltas": []}
    for delta_seg, delta_int in deltas:
        delta_result = _migrate_delta(
            root_zg, level_zg, level_group, delta_seg, delta_int,
            dry_run=dry_run,
        )
        out["deltas"].append(delta_result)
    return out


def _list_legacy_deltas(
    ccl_root: zarr.Group,
) -> list[tuple[str, int]]:
    """Enumerate delta segments under ``cross_chunk_links/`` that look legacy.

    Legacy = group contains a ``data`` blob (the monolithic int64
    record table).  v0.8-shaped groups have ``kK`` sub-arrays
    instead, never a direct ``data`` blob.
    """
    out: list[tuple[str, int]] = []
    for seg in ccl_root:
        try:
            delta_group = ccl_root[seg]
        except KeyError:
            continue
        store = delta_group.store
        # Check for the legacy ``data`` blob.  Its zarr.json sits at
        # ``<level>/cross_chunk_links/<delta>/data/zarr.json``.
        data_meta_key = f"{delta_group.path}/data/zarr.json"
        if _store_has_key(store, data_meta_key):
            try:
                delta_int = int(seg)
            except ValueError:
                # e.g. "+1" / "-1" — strip the sign-prefix
                try:
                    delta_int = int(seg.lstrip("+"))
                except ValueError:
                    continue
            out.append((seg, delta_int))
    return out


def _store_has_key(store: Any, key: str) -> bool:
    import asyncio

    try:
        return asyncio.run(store.exists(key))
    except RuntimeError:
        # Already in an event loop — fall back to a sync probe via
        # a fresh loop.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(store.exists(key))
        finally:
            loop.close()


def _migrate_delta(
    root_zg: zarr.Group,
    level_zg: zarr.Group,
    level_group: Group,
    delta_seg: str,
    delta_int: int,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    """Decode and re-emit one ``cross_chunk_links/<delta>/`` blob."""
    ccl_group = level_zg[CROSS_CHUNK_LINKS][delta_seg]
    legacy_arr = ccl_group["data"]
    meta = dict(legacy_arr.attrs)
    if meta.get("zv_array") != "cross_chunk_links":
        # Foreign blob; refuse rather than corrupt.
        raise StoreError(
            f"resolution_{level_zg.path}/cross_chunk_links/{delta_seg}: "
            f"unexpected zv_array={meta.get('zv_array')!r} under legacy "
            f"data blob"
        )
    link_width = int(meta.get("link_width", 2))
    sid_ndim = int(meta.get("sid_ndim", 0))
    if sid_ndim <= 0:
        raise StoreError(
            f"cross_chunk_links/{delta_seg}: missing sid_ndim on legacy blob"
        )

    # Decode legacy bytes.  The v0.7 writer wrote an int64 array's
    # ``tobytes()`` into a 1-D ``uint8`` chunk-array via ``write_bytes``,
    # so we re-interpret the raw bytes as int64.  Each record =
    # ``link_width * (sid_ndim + 1)`` int64s.
    raw = bytes(np.asarray(legacy_arr[:], dtype=np.uint8).tobytes())
    flat = np.frombuffer(raw, dtype=np.int64)
    per_endpoint = sid_ndim + 1
    per_record = link_width * per_endpoint
    if flat.size % per_record != 0:
        raise StoreError(
            f"cross_chunk_links/{delta_seg}: legacy blob length {flat.size} "
            f"is not a multiple of per_record {per_record}"
        )
    n_records = flat.size // per_record
    records: list[list[tuple[tuple[int, ...], int]]] = []
    for r in range(n_records):
        offset = r * per_record
        endpoints: list[tuple[tuple[int, ...], int]] = []
        for ep in range(link_width):
            base = offset + ep * per_endpoint
            chunk = tuple(int(x) for x in flat[base : base + sid_ndim])
            vi = int(flat[base + sid_ndim])
            endpoints.append((chunk, vi))
        records.append(endpoints)

    # Enumerate parallel attribute blobs at the same delta.  Read each
    # blob's rows in legacy (flat) order.
    attr_blobs = _read_legacy_attribute_blobs(
        level_zg, delta_seg, expected_rows=n_records,
    )

    kK_distribution: dict[int, int] = {}
    for rec in records:
        K = len({ep[0] for ep in rec})
        kK_distribution[K] = kK_distribution.get(K, 0) + 1

    out: dict[str, Any] = {
        "delta": delta_int,
        "record_count": n_records,
        "cell_count": None,  # computed post-write
        "kK_distribution": kK_distribution,
        "attributes": sorted(attr_blobs.keys()),
    }

    if dry_run:
        return out

    # Delete legacy ``data`` blobs FIRST so the new writers' structural
    # "is this a legacy store?" guard doesn't fire on the in-progress
    # state.  After this point the parent groups look empty until the
    # new ``kN`` arrays land.
    _delete_legacy_data_blob(level_zg, f"{CROSS_CHUNK_LINKS}/{delta_seg}/data")
    for attr_name in attr_blobs:
        _delete_legacy_data_blob(
            level_zg,
            f"{CROSS_CHUNK_LINK_ATTRIBUTES}/{attr_name}/{delta_seg}/data",
        )

    # Re-emit link records via the new writer.  This handles the
    # delta=0 L=2 ci=[0,1] canonicalization internally.
    if records:
        write_cross_chunk_links(
            level_group,
            records,
            sid_ndim=sid_ndim,
            delta=delta_int,
            link_width=link_width,
            mode="replace",
        )

    # Re-emit attribute rows.  The new attribute writer slices the
    # passed rows by per-cell record count in canonical (K, lex(chunks))
    # order — so we must reorder the legacy rows to match.
    if attr_blobs:
        permutation = _build_canonical_permutation(
            records, link_width=link_width,
            canonicalize_delta0_l2=(delta_int == 0 and link_width == 2),
        )
        for attr_name, (attr_array, attr_dtype) in attr_blobs.items():
            reordered = np.asarray(attr_array)[permutation]
            write_cross_chunk_link_attributes(
                level_group,
                attr_name,
                reordered,
                num_links=n_records,
                delta=delta_int,
                mode="replace",
            )

    return out


def _read_legacy_attribute_blobs(
    level_zg: zarr.Group,
    delta_seg: str,
    *,
    expected_rows: int,
) -> dict[str, tuple[np.ndarray, np.dtype]]:
    """Read every legacy CCL attribute blob at ``<delta_seg>``.

    Returns a dict ``{attr_name: (rows, dtype)}``.
    """
    out: dict[str, tuple[np.ndarray, np.dtype]] = {}
    if CROSS_CHUNK_LINK_ATTRIBUTES not in level_zg:
        return out
    attrs_root = level_zg[CROSS_CHUNK_LINK_ATTRIBUTES]
    for attr_name in attrs_root:
        attr_name_group = attrs_root[attr_name]
        if delta_seg not in attr_name_group:
            continue
        delta_group = attr_name_group[delta_seg]
        store = delta_group.store
        data_meta_key = f"{delta_group.path}/data/zarr.json"
        if not _store_has_key(store, data_meta_key):
            continue
        legacy_arr = delta_group["data"]
        rows = np.asarray(legacy_arr[:])
        if rows.shape[0] != expected_rows:
            raise StoreError(
                f"cross_chunk_link_attributes/{attr_name}/{delta_seg}: "
                f"row count {rows.shape[0]} != link record count "
                f"{expected_rows}"
            )
        out[attr_name] = (rows, rows.dtype)
    return out


def _build_canonical_permutation(
    records: list[list[tuple[tuple[int, ...], int]]],
    *,
    link_width: int,
    canonicalize_delta0_l2: bool,
) -> np.ndarray:
    """Build the index permutation that takes flat legacy order to canonical.

    Mirrors the per-cell walk performed by
    :func:`write_cross_chunk_link_attributes`: cells are sorted by
    ``(K, lex(sorted_chunks))`` then by within-cell write order (the
    order the writer inserts records, which matches the order this
    helper hands records to :func:`write_cross_chunk_links`).
    """
    if not records:
        return np.array([], dtype=np.int64)

    # Bucket each legacy index by its (K, sorted_chunks_tuple) cell key.
    buckets: dict[tuple[int, tuple[tuple[int, ...], ...]], list[int]] = {}
    for idx, rec in enumerate(records):
        sorted_chunks = tuple(sorted({ep[0] for ep in rec}))
        K = len(sorted_chunks)
        buckets.setdefault((K, sorted_chunks), []).append(idx)

    # Walk buckets in canonical (K, lex(sorted_chunks)) order.  Within
    # each bucket, the order is insertion order (matches what the new
    # writer will record).
    perm: list[int] = []
    for key in sorted(buckets.keys()):
        perm.extend(buckets[key])
    return np.asarray(perm, dtype=np.int64)


def _delete_legacy_data_blob(level_zg: zarr.Group, full_name: str) -> None:
    """Delete a legacy single-blob ``.../data`` zarr Array node in place."""
    parent, _, leaf = full_name.rpartition("/")
    try:
        parent_zg = level_zg[parent]
    except KeyError:
        return
    if leaf not in parent_zg:
        return
    # zarr v3 lacks a public delete; use the store's async API directly.
    arr = parent_zg[leaf]
    store = arr.store

    import asyncio

    async def _delete_all() -> None:
        prefix = arr.path + "/"
        async for key in store.list_prefix(prefix):
            await store.delete(key)
        meta_key = f"{arr.path}/zarr.json"
        try:
            await store.delete(meta_key)
        except Exception:
            pass

    try:
        asyncio.run(_delete_all())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_delete_all())
        finally:
            loop.close()
