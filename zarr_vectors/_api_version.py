"""What this build's CODE surface offers, and how to require it.

Three versions are in play and only one of them is expressible in a
dependency pin:

* the **package** version (``__version__``) — derived from the git tag by
  setuptools-scm, so it moves with every commit and should never be
  asserted against. Each *release* tag is held equal to the format
  version below, enforced by the release workflow, so the number on PyPI
  names the format it ships; a build between releases carries a ``.devN``
  of the next one, which is why a pin still cannot express anything;
* the **on-disk format** version (:data:`zarr_vectors.constants
  .FORMAT_VERSION`) — already negotiable via
  :func:`zarr_vectors.api.require_format`;
* the **API surface** version, this module — which moved when ``api`` and
  ``building`` were introduced without the package version moving at all.

So a consumer that needs ``zarr_vectors.building`` cannot say so. Against
a build without it, it fails with a bare ``ImportError`` from whichever
module imports first, and the pin that was supposed to prevent that is
satisfied. ``require_format`` already exists for exactly this argument one
axis over; this is the same helper for the code.

``FEATURES`` is deliberately small and names things a caller *branches*
on, not every function that exists — a feature flag per symbol would just
be ``__all__`` with extra steps, and the contract test already pins that.
"""

from __future__ import annotations

#: Version of the Python API surface.  Bump the minor when a supported
#: name is added, the major when one is removed or changes meaning.
#: Independent of both ``__version__`` and the on-disk format.
__api_version__ = (1, 3)

#: Capabilities a caller may branch on, each True only when usable.
FEATURES: frozenset[str] = frozenset({
    # The two-surface split exists: zarr_vectors.api and .building.
    "surfaces",
    # building exports the level-wide presence rebuild and the walk it
    # needs, so a consumer need not fork _is_per_chunk_array.
    "presence-rebuild",
    # derive_nonempty_chunks refuses a sharded array instead of silently
    # emptying its manifest.
    "sharded-presence-guard",
    # Selection carries level=None for "unset", so level 0 is requestable.
    "selection-level-optional",
    # Vertex attributes come back from Level.read() for every geometry,
    # not only point clouds.
    "vertex-attributes-on-read",
    # Query.cells() enumerates the grid cells a query touches, for
    # sharding work across processes.
    "query-cells",
    # coarsen_level/build_pyramid forward options= to a registered
    # strategy, and api.coarsen_methods() lists what is installed.
    "coarsen-strategy-options",
    # zarr_vectors.runtime_capabilities() reports what this install can
    # do, including whether device arrays are usable.
    "runtime-capabilities",
    # building.read_cells / read_neighbourhood: many cells of many arrays
    # in one prefetch, as CSR; flat readers and ReadResult take device=.
    "read-cells",
    # write_chunk_fragments(csr=(indices, offsets)): fragments as arrays,
    # and appends that concatenate sections instead of re-encoding.
    "csr-fragments",
    # write_object_manifests(chunk_coords=, fragment_idx=, ...) and
    # read_all_object_manifests_csr: manifests as arrays both ways.
    "array-manifests",
    # building.object_manifest_writer(level, at=): write_object_manifests
    # as a stream, the layout resolved once, rows written in whole shards.
    "object-manifest-writer",
    # write_object_attribute_columns: many object attribute columns in
    # one call, grown in place concurrently.
    "object-attribute-columns",
    # write_link_cells(chunks=, vids=, attributes=): link records and
    # their attributes as arrays, one read-modify-write per cell.
    "array-link-cells",
    # create_store(manifest_layout="dense") and write_object_index /
    # write_object_manifests(layout=): object indexes as fixed-width
    # integer arrays (format 0.9.4).
    "dense-manifests",
    # read_cells(device="cuda", decode=...): cells decoded on the device.
    "device-decode",
    # read_cells(device="cuda", io=...): the local-file read path chosen
    # per call, auto following cuFile's GPUDirect Storage verdict.
    "device-io",
    # CellBatch.io / io_seconds: how each array of a read was served
    # (gds, kvikio-compat, pinned-host, store, host) and where the time
    # went; runtime_capabilities(probe_device=True)["gds"].
    "read-io-report",
    # zarr_vectors.gpu.decode_zstd: zstd frames a caller fetched into
    # device memory, structure-checked before nvCOMP decodes them.
    "decode-zstd",
    # read_cells(fragments=True): each cell's vertex fragment index as
    # flat arrays (FragmentColumn), in the same prefetch.
    "read-cells-fragments",
    # ChunkFragmentIndex.gather(frags) -> (rows, lengths): many fragments'
    # vertex rows in one vectorised pass.
    "fragment-gather",
    # The shared reader and writer pools are dropped in a forked child,
    # which builds fresh ones on first use: a child of a process that had
    # used them no longer hangs on its first pooled read or write.
    "fork-safe-pools",
    # building.set_presence / end_presence_deferral: presence recorded
    # from the cells a caller says it wrote, and a deferral ended without
    # a store-wide rebuild; finalize_links leaves a deferred level alone.
    "supplied-presence",
    # set_presence(verify="objects"|"index", on_mismatch=) and
    # building.stored_objects: supplied presence checked against the
    # store's listing and shard indexes, on any store that lists; a
    # sharded array's presence derived from its shard indexes.
    "verified-presence",
    # building.commit_object_index: the object index's metadata committed
    # after manifest writes, num_present counted rather than carried.
    "commit-object-index",
    # batched_writes / open_write_session(durable=True): local writes
    # fsynced before the block returns; building.cell_objects names the
    # object (or shard) holding each cell.
    "durable-writes",
    # building.store_layout: a store's layout resolved from what it holds,
    # with min_reader, the oldest zarr-vectors that reads all of it.
    "store-layout",
    # building.shard_object_layer / shard_store(object_shard_rows=): the
    # object layer sharded along rows, by zarr's own codec (no format key);
    # writers keep a sharded layer sharded.
    "object-layer-shards",
    # zarr_vectors.concurrency_contract(), building.reserve_object_rows and
    # mode="place" on the object-layer writers: disjoint row ranges filled
    # by several processes at once.
    "concurrency-contract",
    # building.shard_transaction / shard_of: one shard of every per-chunk
    # array staged privately and published by rename on exit.
    "shard-transaction",
    # shard_transaction(io_threads=N): objects encoded, written and fsynced
    # on N lanes, renamed once all are on disk; the publish reopens no
    # array by path, and stores open without v2 metadata probes.
    "shard-transaction-io-threads",
    # shard_transaction(sweep=False): no listing of the owned shards'
    # directories for a failed attempt's partials on entry.
    "transaction-sweep",
    # No ceiling on an object index's rows: OBJECT_INDEX_MAX_ROWS bounds
    # only the empty rows write_object_index(total_objects=) declares, so
    # patch_object_manifests and the writers take indexes past 2**26.
    "object-count-unbounded",
    # Group.cached_nodes keeps direct-read specs apart from presence, so
    # batched_reads inside it on a deferred level reads its plan's cells
    # and derives no presence (no whole-shard reads).
    "spec-without-presence",
    # read_object_manifests_csr on a dense index reads ascending ids'
    # blocks as slices, with no unique or searchsorted.
    "manifests-csr-ascending",
})


def parse_version(text: str) -> tuple[int, ...]:
    """A dotted version string as a comparable tuple.

    Tolerant on purpose: anything non-numeric in a component is dropped,
    so ``"0.2.1.dev66"`` and ``"0.9.0"`` both compare.
    """
    out: list[int] = []
    for part in str(text).strip().split("."):
        digits = "".join(c for c in part if c.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out) or (0,)


# Backwards-compatible private alias.
_parse = parse_version


def satisfies(found: tuple[int, ...], spec: str) -> str | None:
    """``None`` if ``found`` satisfies every clause of ``spec``, else the
    first clause it fails.

    ``spec`` is a comma-separated list of ``>=`` / ``>`` / ``<=`` / ``<``
    / ``==`` clauses, e.g. ``">=0.9,<0.11"``.  One definition, because
    there were two -- here and in ``api.dataset.require_format`` -- and
    two parsers for one syntax is one parser too many.

    Raises:
        ValueError: If a clause cannot be parsed.
    """
    for clause in (c.strip() for c in spec.split(",") if c.strip()):
        for op in (">=", "<=", "==", ">", "<"):
            if not clause.startswith(op):
                continue
            want = parse_version(clause[len(op):])
            width = max(len(found), len(want))
            lhs = found + (0,) * (width - len(found))
            rhs = want + (0,) * (width - len(want))
            ok = {
                ">=": lhs >= rhs, "<=": lhs <= rhs, "==": lhs == rhs,
                ">": lhs > rhs, "<": lhs < rhs,
            }[op]
            if not ok:
                return clause
            break
        else:
            raise ValueError(
                f"cannot parse version clause {clause!r} in {spec!r}; "
                f"expected one of >=, >, <=, <, == followed by a version"
            )
    return None


def require_api(spec: str = "", *, features: object = ()) -> None:
    """Raise unless this build's API surface satisfies ``spec``.

    Args:
        spec: Comma-separated ``>=``/``>``/``<=``/``<``/``==`` clauses
            against :data:`__api_version__`, e.g. ``">=1.0"``.  Empty
            checks only ``features``.
        features: Feature names from :data:`FEATURES` that must all be
            present.  Prefer these to a version range: they say what the
            caller needs rather than when it happened to land.

    Raises:
        ImportError: If the requirement is not met.  ``ImportError``
            rather than a ZVError because the failure is "this build does
            not have what I import", and that is what a caller's
            dependency handling already catches.
    """
    if isinstance(features, str):
        features = (features,)
    missing = sorted(set(features) - FEATURES)
    if missing:
        raise ImportError(
            f"zarr-vectors {'.'.join(map(str, __api_version__))} does not "
            f"provide: {', '.join(missing)}. Known features: "
            f"{', '.join(sorted(FEATURES))}."
        )

    found = __api_version__
    if spec and satisfies(found, spec) is not None:
        raise ImportError(
            f"zarr-vectors API surface is "
            f"{'.'.join(map(str, found))}, which does not satisfy "
            f"{spec!r}. Note this is NOT the package version — "
            f"the two move independently, which is why a pin on "
            f"the package cannot express this."
        )
