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

    def __len__(self) -> int:
        """How many objects exist, from metadata alone."""
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
    def count(self) -> int:
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

    def ids(self) -> npt.NDArray[Any]:
        """Every object id present at this level.

        Reads the manifests' *count*, not their contents — an id is
        present if it has a manifest row, and that is answerable from
        metadata.
        """
        return np.arange(len(self), dtype=np.int64)

    def manifests(self, ids: Sequence[int] | None = None) -> dict[int, Any]:
        """Where each object's fragments live.

        The physical answer, offered because a tool that builds its own
        gather plan genuinely needs it — and because the alternative was
        that such a tool imported three private names to get it.
        """
        from zarr_vectors.core.arrays import read_object_manifests
        from zarr_vectors.core.store import get_resolution_level

        group = get_resolution_level(self._level.dataset._group, self._level.index)
        return read_object_manifests(group, ids=ids)

    def __contains__(self, object_id: int) -> bool:
        return 0 <= int(object_id) < len(self)

    def __repr__(self) -> str:
        return f"ObjectCatalog(level={self._level.index}, count={len(self)})"
