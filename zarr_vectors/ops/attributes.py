"""Attribute edits across the three scopes.

- **Per-vertex**: ragged, fragment-aligned, lives in
  ``vertex_attributes/<name>/<chunk>``.  The Iteration-1 ``edit_vertex``
  already supports per-vertex attribute edits via ``new_attrs=``; this
  module exposes the standalone surface (``edit_attribute(VertexRef
  ...)``) that delegates to the same path.
- **Per-object**: dense, OID-indexed, lives in
  ``object_attributes/<name>/{data,present_mask}``.  A single-OID edit
  reads the whole array, replaces row ``oid``, and writes back.
  ``add_attribute`` for a brand-new name allocates a zero-filled
  ``(num_objects, ...)`` buffer.
- **Per-link**: ragged, parallel to ``links/0/<chunk>``, lives under
  ``link_attributes/<name>/0/<chunk>``.  Read-modify-write the fragment
  group via the change-set builder.

Tombstones for ``remove_attribute``: there is no on-disk "null" for a
dense per-object attribute, so removal writes the dtype's zero
sentinel.  Per-vertex / per-link removal raises ``EditError`` —
attribute groups are positional and cannot drop a row without
shifting downstream indices (see ``remove_vertex(atomic=False)``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import OBJECT_ATTRIBUTES
from zarr_vectors.exceptions import EditError
from zarr_vectors.ops.refs import AttributeRef, LinkRef, ObjectRef, VertexRef

if TYPE_CHECKING:
    from zarr_vectors.ops.edit import EditSession


def edit_attribute_in_session(
    session: EditSession,
    ref: AttributeRef,
    value: npt.ArrayLike,
) -> None:
    """Dispatch to the right scope-specific edit path."""
    if ref.scope == "vertex":
        return _edit_vertex_attr(session, ref, value)
    if ref.scope == "object":
        return _edit_object_attr(session, ref, value)
    if ref.scope == "link":
        return _edit_link_attr(session, ref, value)
    raise EditError(f"unknown attribute scope {ref.scope!r}")


def add_attribute_in_session(
    session: EditSession,
    ref: AttributeRef,
    value: npt.ArrayLike,
) -> None:
    """Add an attribute value.

    For per-vertex and per-link scopes this is equivalent to
    ``edit_attribute`` once the array exists; if the named attribute is
    new and the scope is per-object, this allocates a zero-filled dense
    array of ``num_objects`` rows first, then writes the supplied
    value at ``ref.target.object_id``.
    """
    if ref.scope == "object":
        _ensure_object_attribute_array(
            session, ref.name,
            value=np.asarray(value),
            target=ref.target,
        )
    return edit_attribute_in_session(session, ref, value)


def remove_attribute_in_session(
    session: EditSession,
    ref: AttributeRef,
) -> None:
    """Remove one attribute value.

    Per-object: writes a dtype-zero sentinel at that OID.  Per-vertex
    / per-link: refused because positional ragged groups cannot drop a
    row without shifting indices.  Callers can write a zero sentinel
    themselves via ``edit_attribute(value=...)`` if a tombstone is
    semantically sensible.
    """
    if ref.scope == "object":
        from zarr_vectors.core.arrays import read_object_attributes
        from zarr_vectors.core.store import get_resolution_level
        assert isinstance(ref.target, ObjectRef)
        level_group = get_resolution_level(session.root, ref.target.level)
        try:
            arr = read_object_attributes(level_group, ref.name)
        except Exception as e:
            raise EditError(
                f"remove_attribute: attribute {ref.name!r} not present "
                f"at level {ref.target.level}: {e}"
            ) from None
        zero = np.zeros(arr.shape[1:] if arr.ndim > 1 else (), dtype=arr.dtype)
        return _edit_object_attr(session, ref, zero)
    raise EditError(
        f"remove_attribute is only supported for per-object scope this "
        f"iteration; got scope={ref.scope!r}.  Use edit_attribute(value=...) "
        f"with an explicit sentinel value for vertex / link scopes."
    )


# ---------------------------------------------------------------------
# Per-vertex (delegates to the existing vertex-edit path)
# ---------------------------------------------------------------------

def _edit_vertex_attr(
    session: EditSession,
    ref: AttributeRef,
    value: npt.ArrayLike,
) -> None:
    if not isinstance(ref.target, VertexRef):
        raise EditError(
            f"AttributeRef(scope='vertex') requires target to be a "
            f"VertexRef; got {type(ref.target).__name__}"
        )
    vref = ref.target
    builder = session._builder(vref.level, vref.chunk)
    builder.require_attribute(session.root, ref.name)
    attr_list = builder.attr_groups[ref.name]
    if vref.fragment >= len(attr_list):
        raise EditError(
            f"vertex_attributes/{ref.name}: fragment {vref.fragment} "
            f"missing in chunk {vref.chunk}"
        )
    group = attr_list[vref.fragment]
    if vref.local >= group.shape[0]:
        raise EditError(
            f"vertex_attributes/{ref.name}: row {vref.local} out of "
            f"range in fragment {vref.fragment}"
        )
    val = np.asarray(value, dtype=builder.attr_dtype.get(ref.name, np.float32))
    group[vref.local] = val
    builder.attrs_dirty[ref.name] = True
    session._mark_edit(vref.level)


# ---------------------------------------------------------------------
# Per-object (dense RMW with present_mask preservation)
# ---------------------------------------------------------------------

def _edit_object_attr(
    session: EditSession,
    ref: AttributeRef,
    value: npt.ArrayLike,
) -> None:
    if not isinstance(ref.target, ObjectRef):
        raise EditError(
            f"AttributeRef(scope='object') requires target to be an "
            f"ObjectRef; got {type(ref.target).__name__}"
        )
    oref = ref.target
    from zarr_vectors.core.arrays import (
        read_object_attribute_present_mask,
        read_object_attributes,
        write_object_attributes,
    )
    from zarr_vectors.core.store import get_resolution_level

    level_group = get_resolution_level(session.root, oref.level)
    try:
        arr = read_object_attributes(level_group, ref.name)
    except Exception as e:
        raise EditError(
            f"edit_attribute: per-object attribute {ref.name!r} not "
            f"present at level {oref.level}: {e}.  Use add_attribute "
            f"to allocate it first."
        ) from None

    if oref.object_id < 0 or oref.object_id >= arr.shape[0]:
        raise EditError(
            f"edit_attribute: object_id={oref.object_id} out of range "
            f"[0, {arr.shape[0]})"
        )
    val = np.asarray(value, dtype=arr.dtype)
    if arr.ndim == 1:
        if val.shape not in ((), (1,)):
            raise EditError(
                f"per-object attribute {ref.name!r} expects scalar rows; "
                f"got shape {val.shape}"
            )
        arr = arr.copy()
        arr[oref.object_id] = val.reshape(())
    else:
        if val.shape != arr.shape[1:]:
            raise EditError(
                f"per-object attribute {ref.name!r} expects row shape "
                f"{arr.shape[1:]}; got {val.shape}"
            )
        arr = arr.copy()
        arr[oref.object_id] = val

    mask = read_object_attribute_present_mask(level_group, ref.name)
    if mask is not None:
        mask = mask.copy()
        mask[oref.object_id] = 1
    write_object_attributes(level_group, ref.name, arr, present_mask=mask)
    session._mark_edit(oref.level)


def _ensure_object_attribute_array(
    session: EditSession,
    name: str,
    *,
    value: np.ndarray,
    target,
) -> None:
    """Allocate a zero-filled dense ``object_attributes/<name>/`` array
    sized for the current ``num_objects`` if it doesn't exist yet.
    """
    from zarr_vectors.core.arrays import (
        read_object_attributes,
        write_object_attributes,
    )
    from zarr_vectors.core.store import get_resolution_level

    if not isinstance(target, ObjectRef):
        raise EditError(
            "add_attribute(scope='object') requires target to be an "
            "ObjectRef"
        )
    level_group = get_resolution_level(session.root, target.level)
    try:
        read_object_attributes(level_group, name)
        return  # already exists
    except Exception:
        pass

    manifests = session._all_manifests_for(target.level)
    n_objects = len(manifests)
    if n_objects == 0:
        raise EditError(
            f"add_attribute(scope='object'): no objects at level "
            f"{target.level}; cannot size the dense buffer"
        )
    dtype = value.dtype
    shape = (n_objects,) if value.ndim == 0 else (n_objects,) + value.shape
    arr = np.zeros(shape, dtype=dtype)
    # No present_mask on a brand-new attribute; rows are zero-filled
    # for objects that haven't been written yet.
    write_object_attributes(level_group, name, arr)


# ---------------------------------------------------------------------
# Per-link
# ---------------------------------------------------------------------

def _edit_link_attr(
    session: EditSession,
    ref: AttributeRef,
    value: npt.ArrayLike,
) -> None:
    if not isinstance(ref.target, LinkRef):
        raise EditError(
            f"AttributeRef(scope='link') requires target to be a "
            f"LinkRef; got {type(ref.target).__name__}"
        )
    lref = ref.target
    if lref.delta != 0:
        raise EditError(
            f"per-link attribute edits only support intra-level "
            f"links (delta=0); got delta={lref.delta}"
        )
    from zarr_vectors.core.arrays import (
        read_chunk_link_attributes,
        write_chunk_link_attributes,
    )
    from zarr_vectors.core.store import get_resolution_level

    level_group = get_resolution_level(session.root, lref.level)
    # Determine dtype + ncols from on-disk meta when available.
    from zarr_vectors.core.paths import link_attributes_path
    full_name = link_attributes_path(ref.name, 0)
    try:
        meta = level_group.read_array_meta(full_name)
        dtype = np.dtype(meta.get("dtype", "float32"))
        shape = meta.get("shape", [])
        ncols = int(shape[-1]) if len(shape) >= 2 else 1
    except Exception:
        dtype = np.dtype(np.float32)
        ncols = 1

    try:
        groups = read_chunk_link_attributes(
            level_group, ref.name, lref.chunk,
            dtype=dtype, ncols=ncols, delta=0,
        )
    except Exception as e:
        raise EditError(
            f"edit_attribute(scope='link'): {ref.name!r} not present "
            f"in chunk {lref.chunk}: {e}"
        ) from None

    if lref.fragment < 0 or lref.fragment >= len(groups):
        raise EditError(
            f"link_attributes/{ref.name}: fragment {lref.fragment} out "
            f"of range in chunk {lref.chunk}"
        )
    group = groups[lref.fragment]
    if lref.row < 0 or lref.row >= group.shape[0]:
        raise EditError(
            f"link_attributes/{ref.name}: row {lref.row} out of range "
            f"in fragment {lref.fragment}"
        )

    val = np.asarray(value, dtype=dtype)
    if ncols == 1:
        group[lref.row] = val.reshape(())
    else:
        group[lref.row] = val
    write_chunk_link_attributes(
        level_group, ref.name, lref.chunk, groups,
        dtype=dtype, delta=0,
    )
    session._mark_edit(lref.level)


# Module-export marker — referenced by ops/__init__.py.
__all__ = [
    "edit_attribute_in_session",
    "add_attribute_in_session",
    "remove_attribute_in_session",
]


# Silence unused-import warnings for symbols referenced only via TYPE_CHECKING.
_ = OBJECT_ATTRIBUTES
