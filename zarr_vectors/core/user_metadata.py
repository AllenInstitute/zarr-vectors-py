"""A sanctioned place for an application's own metadata.

Consumers need to keep state beside the data it describes — which chunks
a pipeline has processed, a chunk-coordinate lookup table, a provenance
stamp — and they need it *in* the store, because a sidecar file and a
store drift apart.

With nowhere provided, they use the group's own ``attrs``.  Three
separate downstream modules do exactly that, and the failure mode is not
hypothetical: ``attrs`` is one shared JSON blob, so two writers touching
different keys still read-modify-write the same object and the second
overwrites the first's addition.  There is also nothing keeping an
application's keys from colliding with the format's own.

Namespacing fixes the collision; :meth:`Namespace.transact` and
:meth:`Namespace.compare_and_set` narrow the race.  Neither makes
concurrent writers *safe* — that needs coordination this library does not
provide, and the docstrings say so rather than implying otherwise.
"""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any

from zarr_vectors.exceptions import ZVError

__all__ = ["Metadata", "Namespace", "USER_METADATA_KEY"]

# One reserved key on the group's attributes, under which every
# namespace lives.  A single reserved name is easier to keep out of the
# format's way than a convention about prefixes.
USER_METADATA_KEY = "user_metadata"


class MetadataConflict(ZVError):
    """A compare-and-set found a value other than the one expected."""


class Namespace(MutableMapping[str, Any]):
    """One application's slice of a store's metadata.

    A ``MutableMapping``, so it behaves like the dict callers are already
    using — the migration is a change of where it lives, not of how it is
    used.
    """

    __slots__ = ("_owner", "_name")

    def __init__(self, owner: Any, name: str) -> None:
        self._owner = owner
        self._name = name

    # -- reading ------------------------------------------------------

    def _read(self) -> dict[str, Any]:
        block = self._owner.attrs.get(USER_METADATA_KEY) or {}
        value = block.get(self._name) or {}
        return dict(value)

    def __getitem__(self, key: str) -> Any:
        try:
            return self._read()[key]
        except KeyError:
            raise KeyError(f"{key!r} not in namespace {self._name!r}") from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._read())

    def __len__(self) -> int:
        return len(self._read())

    def to_dict(self) -> dict[str, Any]:
        return self._read()

    # -- writing ------------------------------------------------------

    def _write(self, value: dict[str, Any]) -> None:
        attrs = self._owner.attrs
        block = dict(attrs.get(USER_METADATA_KEY) or {})
        block[self._name] = value
        attrs[USER_METADATA_KEY] = block

    def __setitem__(self, key: str, value: Any) -> None:
        current = self._read()
        current[key] = value
        self._write(current)

    def __delitem__(self, key: str) -> None:
        current = self._read()
        del current[key]
        self._write(current)

    def update(self, other: Any = (), /, **kwargs: Any) -> None:  # type: ignore[override]
        """Merge in several keys with one read-modify-write.

        Assigning them one at a time is N round-trips and N chances to
        lose a concurrent writer's key; this is one of each.
        """
        current = self._read()
        current.update(dict(other), **kwargs)
        self._write(current)

    @contextmanager
    def transact(self) -> Iterator[dict[str, Any]]:
        """Edit the whole namespace as a dict, writing once at the end.

        Not a transaction in the durable sense — nothing is rolled back
        and no lock is held.  It narrows the window between read and
        write from N operations to one, which is worth having and is not
        the same as being safe against a concurrent writer.
        """
        working = self._read()
        yield working
        self._write(working)

    def compare_and_set(self, key: str, expect: Any, value: Any) -> bool:
        """Set ``key`` to ``value`` only if it currently equals ``expect``.

        Returns whether the write happened.  Use ``expect=None`` to mean
        "only if absent".  The read and the write are still two
        operations, so this narrows a race rather than eliminating one; it
        is enough for the common case of several workers claiming
        disjoint keys, and not enough for two workers claiming the same
        one.
        """
        current = self._read()
        if current.get(key) != expect:
            return False
        current[key] = value
        self._write(current)
        return True

    def clear(self) -> None:
        self._write({})

    def __repr__(self) -> str:
        return f"Namespace({self._name!r}, {len(self)} key(s))"


class Metadata:
    """The user-metadata namespaces on one group."""

    __slots__ = ("_owner",)

    def __init__(self, owner: Any) -> None:
        self._owner = owner

    def namespace(self, name: str) -> Namespace:
        """Get (or implicitly create) a namespace.

        Names are the caller's to choose; the convention that helps is a
        package name, so two applications sharing a store do not have to
        coordinate.
        """
        if not name or "/" in name:
            raise ValueError(
                f"namespace name {name!r} must be non-empty and contain no '/'"
            )
        return Namespace(self._owner, name)

    def __getitem__(self, name: str) -> Namespace:
        return self.namespace(name)

    def names(self) -> tuple[str, ...]:
        block = self._owner.attrs.get(USER_METADATA_KEY) or {}
        return tuple(sorted(block))

    def __contains__(self, name: str) -> bool:
        return name in self.names()

    def drop(self, name: str) -> None:
        attrs = self._owner.attrs
        block = dict(attrs.get(USER_METADATA_KEY) or {})
        block.pop(name, None)
        attrs[USER_METADATA_KEY] = block

    def __repr__(self) -> str:
        return f"Metadata({list(self.names())})"
