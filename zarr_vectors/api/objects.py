"""Objects, addressed by id rather than by where their pieces landed.

An object — a streamline, a skeleton, a mesh — is stored as fragments
scattered across the chunks it passes through, plus a manifest naming
them.  Reading one therefore means decoding its manifest and gathering
what it points at.

The existing entry point decodes *every* manifest in the store.  On a
whole-brain tractography store that is twenty-one million Python objects
built to answer a question about a hundred, and it is the step that
hangs.  Downstream worked around it by importing a private block
expander, a layout sentinel and a raw coordinate selection — three
internals pinned purely to avoid an O(N) decode.

:class:`ObjectCatalog` is the question they were trying to ask.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zarr_vectors.api.level import Level
    from zarr_vectors.api.result import ReadResult

__all__ = ["ObjectCatalog"]


class ObjectCatalog:
    """The objects at one level.

    Indexing reads only what was asked for::

        level.objects[7]            # one object's geometry
        level.objects[[7, 9, 11]]   # three, in one gather
        len(level.objects)          # from metadata; reads nothing
    """

    __slots__ = ("_level",)

    def __init__(self, level: Level) -> None:
        self._level = level

    def _group(self) -> Any:
        from zarr_vectors.core.store import get_resolution_level

        return get_resolution_level(
            self._level.dataset._group, self._level.index,
        )

    def __len__(self) -> int:
        """How many object SLOTS exist, from metadata alone.

        Slots, not objects.  A sparsified pyramid level keeps a dropped
        object's id as an empty manifest so ids stay stable across levels,
        so this counts those too.  :attr:`count` is the number actually
        present; :attr:`slots` is this, named for what it is.
        """
        from zarr_vectors.core.arrays import object_count
        from zarr_vectors.core.store import get_resolution_level

        try:
            group = get_resolution_level(
                self._level.dataset._group, self._level.index,
            )
        except Exception:
            return 0
        return object_count(group)

    @property
    def slots(self) -> int:
        """How many ids this level addresses, present or not."""
        return len(self)

    @property
    def count(self) -> int:
        """How many objects are actually here.

        Reads the stamped ``num_present`` where available and decodes the
        manifests otherwise, so it is correct on stores written before
        that field existed and merely slower.
        """
        from zarr_vectors.core.arrays import object_present_count

        try:
            return object_present_count(self._group())
        except Exception:
            return len(self)

    def __getitem__(self, key: int | Sequence[int] | slice) -> ReadResult:
        """Read one object, or several, by id."""
        if isinstance(key, slice):
            ids = list(range(*key.indices(len(self))))
        elif isinstance(key, (int, np.integer)):
            ids = [int(key)]
        else:
            ids = [int(i) for i in key]
        return self._level.select(objects=ids).read()

    def __iter__(self) -> Iterator[int]:
        return iter(self.ids())

    def ids(self, *, present: bool = True) -> npt.NDArray[Any]:
        """Object ids at this level.

        Args:
            present: When True (the default), only ids that actually hold
                geometry.  When False, every addressable slot.

        This used to return ``arange(slot_count)`` unconditionally, which
        is wrong for exactly the stores this package builds: a sparsified
        level reports every dropped id as present, and reading one gives
        ``vertex_count == 0`` — indistinguishable from an empty region.
        """
        if not present:
            return np.arange(len(self), dtype=np.int64)
        try:
            return np.flatnonzero(self.present_mask()).astype(np.int64)
        except Exception:
            return np.arange(len(self), dtype=np.int64)

    def present_mask(self) -> npt.NDArray[Any]:
        """Per-slot boolean: does this id hold any geometry?

        Decodes the manifests — there is no cheaper exact answer per id.
        Use :attr:`count` when only the total is needed.
        """
        from zarr_vectors.core.arrays import object_present_mask

        return object_present_mask(self._group())

    def manifests(self, ids: Sequence[int] | None = None) -> dict[int, Any]:
        """Where each object's fragments live.

        The physical answer, offered because a tool that builds its own
        gather plan genuinely needs it — and because the alternative was
        that such a tool imported three private names to get it.
        """
        from zarr_vectors.core.arrays import read_object_manifests

        return read_object_manifests(self._group(), ids=ids)

    def __contains__(self, object_id: int) -> bool:
        """Whether this id holds geometry — not merely whether it is in range."""
        oid = int(object_id)
        if not (0 <= oid < len(self)):
            return False
        try:
            return bool(self.present_mask()[oid])
        except Exception:
            return True

    def __repr__(self) -> str:
        return (
            f"ObjectCatalog(level={self._level.index}, "
            f"count={self.count}, slots={self.slots})"
        )
