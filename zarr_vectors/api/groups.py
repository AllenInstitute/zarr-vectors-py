"""Object groups, addressed by name.

A grouping is stored as a row of object ids.  Nothing on disk says what
the row *means*, so every reader has to hard-code the integer — and a
store cannot be understood without the writing application's source next
to it.  One downstream module exists solely to paper over this, and its
docstring is the clearest statement of the problem: *"nothing on disk
says that row 3 means 'the network'. Every reader therefore had to
hard-code the integer."*

Naming them is a small addition — a list of names on the groupings
array's own attributes, parallel to the rows — and it makes the store
self-describing.  Stores written before it are still readable: an
unnamed row answers to ``"group_3"``, so nothing has to be migrated to
be usable.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import GROUPS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zarr_vectors.api.level import Level
    from zarr_vectors.api.result import ReadResult

__all__ = ["GroupCatalog", "ObjectGroup"]

# Attribute on the groupings array holding one name per row.
GROUP_NAMES_ATTR = "group_names"


def _fallback_name(group_id: int) -> str:
    """What an unnamed row is called.

    Stores written before names existed still have to be addressable, and
    by something more meaningful than a bare integer at the call site.
    """
    return f"group_{group_id}"


class ObjectGroup:
    """One named set of objects."""

    __slots__ = ("_level", "_id", "_name")

    def __init__(self, level: Level, group_id: int, name: str) -> None:
        self._level = level
        self._id = int(group_id)
        self._name = name

    @property
    def id(self) -> int:
        """The row index this group occupies.  An implementation detail,
        exposed because a store written by other tools may only be
        addressable this way."""
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def members(self) -> npt.NDArray[Any]:
        """The object ids in this group."""
        from zarr_vectors.core.arrays import read_group_object_ids
        from zarr_vectors.core.store import get_resolution_level

        group = get_resolution_level(self._level.dataset._group, self._level.index)
        try:
            return np.asarray(read_group_object_ids(group, self._id), dtype=np.int64)
        except Exception:
            return np.zeros((0,), dtype=np.int64)

    def read(self, **kw: Any) -> ReadResult:
        """Read this group's geometry."""
        return self._level.select(groups=[self._id], **kw).read()

    def __len__(self) -> int:
        return int(len(self.members))

    def __repr__(self) -> str:
        return f"ObjectGroup({self._name!r}, id={self._id}, members={len(self)})"


class GroupCatalog(Mapping[str, ObjectGroup]):
    """The object groups at one level, keyed by name."""

    __slots__ = ("_level",)

    def __init__(self, level: Level) -> None:
        self._level = level

    def _names(self) -> list[str]:
        """One name per row, falling back for rows that have none."""
        from zarr_vectors.core.store import get_resolution_level

        try:
            group = get_resolution_level(
                self._level.dataset._group, self._level.index,
            )
            meta = group.read_array_meta(GROUPS)
        except Exception:
            return []
        declared = meta.get(GROUP_NAMES_ATTR)
        count = int(meta.get("num_groups", 0) or 0)
        if isinstance(declared, (list, tuple)) and declared:
            names = [str(n) for n in declared]
            count = max(count, len(names))
        else:
            names = []
        return [
            names[i] if i < len(names) and names[i] else _fallback_name(i)
            for i in range(count)
        ]

    def __getitem__(self, name: str) -> ObjectGroup:
        names = self._names()
        if name in names:
            return ObjectGroup(self._level, names.index(name), name)
        raise KeyError(
            f"no group named {name!r}; this level has {names or 'none'}"
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self._names())

    def __len__(self) -> int:
        return len(self._names())

    def names(self) -> tuple[str, ...]:
        return tuple(self._names())

    def by_id(self, group_id: int) -> ObjectGroup:
        """A group by row index, for stores that carry no names."""
        names = self._names()
        gid = int(group_id)
        name = names[gid] if 0 <= gid < len(names) else _fallback_name(gid)
        return ObjectGroup(self._level, gid, name)

    def name_rows(self, names: Sequence[str]) -> None:
        """Give the existing rows names, in row order.

        Purely additive: the rows and their members are untouched, and a
        reader that does not know about names sees exactly what it saw
        before.
        """
        from zarr_vectors.core.store import get_resolution_level

        group = get_resolution_level(self._level.dataset._group, self._level.index)
        meta = dict(group.read_array_meta(GROUPS))
        meta[GROUP_NAMES_ATTR] = [str(n) for n in names]
        group.write_array_meta(GROUPS, meta)

    def __repr__(self) -> str:
        return f"GroupCatalog({list(self.names())})"
