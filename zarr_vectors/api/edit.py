"""Editing, addressed by what a thing *is* rather than where it landed.

The editing layer underneath is physical by design, and rightly so: a
surgical edit has to name the exact fragment row it is changing, and
``VertexRef(level, chunk, fragment, local)`` says that precisely.  Its
own module docstring calls these "physical addresses".

The problem is that a caller rarely *has* one.  What a caller has is "the
third vertex of object 7", or "whatever is at this coordinate", and
turning that into four grid numbers means knowing the chunking, the
fragment layout, and where the object's pieces were scattered.

The resolution already exists — ``VertexRef.from_object`` and
``VertexRef.from_position`` — but only for vertices, and only if you know
to look.  :class:`EditPlan` makes it the normal way in, and leaves the
physical constructors available for the cases that genuinely need them.

Nothing here reimplements the edit semantics.  Manifest propagation,
object-id allocation and the flush are the existing session's, unchanged.

**Edits are copy-on-write by default, and this surprises people.**
``atomic=True`` — the underlying default — appends a new fragment and
rewrites the referring manifests under a *new* object id, leaving the
original object exactly as it was.  So after moving a vertex of object 7,
object 7 still reads back unchanged and the edited geometry lives under
some other id.  That is deliberate: a concurrent reader never sees a
half-written object.  But a caller who does not know it concludes the
edit silently failed.

:meth:`EditPlan.renamed` reports the mapping, so the surprise is at
least visible.  ``in_place=True`` asks for overwrite-in-place instead,
but it is a *request*: object-bearing geometry has its manifest rewritten
either way, and measurement shows a polyline edit reallocates under both
settings.  So do not branch on the flag — read :meth:`renamed` and act on
what actually happened.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zarr_vectors.api.dataset import Dataset

__all__ = ["EditPlan"]


class EditPlan:
    """A batch of edits, applied on exit.

    Used as a context manager::

        with dataset.editing() as edit:
            edit.move_vertex(object=7, index=3, to=[10.0, 20.0, 30.0])
            edit.set_attribute("intensity", object=7, index=3, value=0.5)

    Every method takes intent — an object id and an index within it, or a
    coordinate — and resolves it to a physical reference through the
    existing ``from_object`` / ``from_position`` constructors.
    """

    __slots__ = ("_dataset", "_session", "_level", "_report")

    def __init__(
        self, dataset: Dataset, *, level: int = 0, in_place: bool = False,
        **session_kw: Any,
    ) -> None:
        from zarr_vectors.ops.edit import EditSession

        self._dataset = dataset
        self._level = int(level)
        session_kw.setdefault("atomic", not in_place)
        self._session = EditSession(dataset._group, **session_kw)
        self._report: Any = None

    def __enter__(self) -> EditPlan:
        self._session.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._session.__exit__(exc_type, exc, tb)
        # Metadata may have moved underneath the facade's caches.
        self._dataset._meta = None
        self._dataset._levels.clear()

    @property
    def session(self) -> Any:
        """The underlying edit session.

        An escape hatch, and a deliberate one: the physical layer has
        operations this wrapper does not cover, and a caller reaching for
        it is a signal about what to add here rather than something to
        prevent.
        """
        return self._session

    def renamed(self) -> dict[int, int]:
        """Old object id to new, for edits that copied rather than
        overwrote.

        Empty under ``atomic=False``.  Non-empty means the objects you
        edited now live under different ids, and reading the old ones
        returns the *pre-edit* geometry.
        """
        return dict(getattr(self.report, "oid_remap", None) or {})

    def new_id_of(self, object_id: int) -> int:
        """Where ``object_id``'s edited geometry ended up.

        Returns the id unchanged when nothing moved, so a caller can
        apply it unconditionally.
        """
        return self.renamed().get(int(object_id), int(object_id))

    @property
    def report(self) -> Any:
        """What the last flush did.

        ``EditSession.report`` is a property, not a method -- calling it
        raises ``'EditReport' object is not callable``, which is an
        unhelpful way to learn that.
        """
        return self._report or self._session.report

    # ---------------- resolution ----------------

    def _vertex(
        self,
        *,
        object: int | None = None,
        index: int = 0,
        at: Sequence[float] | None = None,
        tolerance: float = 1e-6,
    ) -> Any:
        """Turn intent into a physical vertex reference.

        Exactly one of ``object`` or ``at`` names the vertex.  Both
        constructors already exist; what was missing was anybody being
        pointed at them.
        """
        from zarr_vectors.ops.refs import VertexRef

        if (object is None) == (at is None):
            raise ValueError(
                "name the vertex either by object= and index=, or by at= "
                "(a coordinate) -- not both and not neither."
            )
        if object is not None:
            return VertexRef.from_object(
                self._dataset._group, level=self._level,
                object_id=int(object), vertex_index=int(index),
            )
        assert at is not None  # narrowed by the check above
        return VertexRef.from_position(
            self._dataset._group, level=self._level, pos=list(at), tol=tolerance,
        )

    # ---------------- operations ----------------

    def move_vertex(
        self,
        *,
        to: npt.ArrayLike,
        object: int | None = None,
        index: int = 0,
        at: Sequence[float] | None = None,
    ) -> None:
        """Move a vertex to a new position.

        Under the default copy-on-write mode the object this vertex
        belongs to is reallocated; :meth:`new_id_of` says where it went.
        """
        self._session.edit_vertex(
            self._vertex(object=object, index=index, at=at), new_pos=to,
        )

    def remove_vertex(
        self,
        *,
        object: int | None = None,
        index: int = 0,
        at: Sequence[float] | None = None,
    ) -> None:
        """Delete a vertex."""
        self._session.remove_vertex(self._vertex(object=object, index=index, at=at))

    def set_attribute(
        self,
        name: str,
        value: npt.ArrayLike,
        *,
        object: int | None = None,
        index: int = 0,
        at: Sequence[float] | None = None,
    ) -> None:
        """Set a per-vertex attribute."""
        from zarr_vectors.ops.refs import AttributeRef

        target = self._vertex(object=object, index=index, at=at)
        self._session.edit_attribute(
            AttributeRef(scope="vertex", name=name, target=target), value,
        )

    def set_object_attribute(self, name: str, object: int, value: npt.ArrayLike) -> None:
        """Set a per-object attribute."""
        from zarr_vectors.ops.refs import AttributeRef, ObjectRef

        self._session.edit_attribute(
            AttributeRef(
                scope="object", name=name,
                target=ObjectRef(level=self._level, object_id=int(object)),
            ),
            value,
        )

    def remove_object(self, object: int) -> None:
        """Delete an object and everything belonging to it."""

        self._session.remove_object(int(object), level=self._level)

    def flush(self) -> Any:
        """Apply everything queued so far, without closing the plan."""
        self._report = self._session.flush()
        return self._report

    def __repr__(self) -> str:
        return f"EditPlan(level={self._level})"
