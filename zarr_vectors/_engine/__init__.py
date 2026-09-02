"""The request / plan / execute engine behind the data-oriented API.

This package is the seam that lets a caller talk about *data* — "the
polylines in this box", "these object ids" — while the I/O underneath
stays batched.  Nothing in here is public API; it is imported by
``zarr_vectors.api`` and by the ``read_*`` functions, and by nothing
outside this package.

The shape is borrowed from what replaced Zarr's abandoned storage
transformers, not from the transformers themselves:

1. **Plural-first at the planning boundary.**  A read is expressed as one
   :class:`~zarr_vectors._engine.plan.ReadPlan` describing everything it
   wants, not as N scalar requests.  Callers may be scalar; the port is
   not.
2. **Batch in, one batch out, realign.**  A :class:`Fetcher` takes a whole
   plan and returns a whole :class:`Snapshot`; a cell that came back
   absent is recorded as a miss rather than dropped or reordered.
3. **A narrow port.**  :class:`Fetcher` has exactly one method, so a fake
   over a plain dict is trivial and the layers above genuinely cannot
   tell what the storage is.  (Zarr's ``ByteGetter`` has one method; the
   ``StorageTransformer`` it replaced had about twenty, and every one of
   them had to sustain the illusion.)
4. **Top-down.**  A plan says what the *read* needs.  It does not describe
   the store's layout bottom-up and leave a hidden layer to aggregate —
   that inversion is what forced Zarr's sharding transformer to invent
   fake chunk keys, and it is why sharding is a codec today.

The executor is a fixpoint loop: fetch what we know we want, run an
ordinary *synchronous* decoder against the snapshot, and let the decoder
tell us what else it needed.  That is the design already proven by
:func:`zarr_vectors.core.aio.read_async` — generalised here over the
fetcher and over strictness, so the sync, batched and async read paths
are one code path rather than three.
"""

from __future__ import annotations

from zarr_vectors._engine.execute import aexecute, execute
from zarr_vectors._engine.fetch import (
    AsyncFetcher,
    Capabilities,
    DictFetcher,
    Fetcher,
    GroupFetcher,
    RecordingFetcher,
)
from zarr_vectors._engine.plan import CellRequest, PlanCost, ReadPlan, RowRequest
from zarr_vectors._engine.snapshot import Snapshot

__all__ = [
    "AsyncFetcher",
    "Capabilities",
    "CellRequest",
    "DictFetcher",
    "Fetcher",
    "GroupFetcher",
    "PlanCost",
    "ReadPlan",
    "RecordingFetcher",
    "RowRequest",
    "Snapshot",
    "aexecute",
    "execute",
]
