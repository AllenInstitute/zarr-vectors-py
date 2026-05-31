"""Parametric object I/O for zarr vectors stores.

Parametric objects are algebraic entities (planes, lines, spheres)
defined by coefficients rather than sampled vertices.  They live in
the ``/parametric/`` group at the store root, outside the resolution
level hierarchy — they are resolution-independent.

Each object is stored as a type tag followed by its coefficients.
A type registry in ``/parametric/.zattrs`` maps type IDs to names
and coefficient schemas so readers can parse any object without
hardcoded knowledge of every type.

Supported built-in types:

- **Plane** (type 0): ``Ax + By + Cz + D = 0`` → coefficients ``[A, B, C, D]``
- **Line** (type 1): ``P(t) = P₀ + t·d`` → coefficients ``[x0, y0, z0, dx, dy, dz]``
- **Sphere** (type 2): ``|P - C|² = r²`` → coefficients ``[cx, cy, cz, r]``
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import PARAMETRIC_GROUP
from zarr_vectors.core.metadata import (
    DEFAULT_PARAMETRIC_TYPES,
    PARAMETRIC_LINE,
    PARAMETRIC_PLANE,
    PARAMETRIC_SPHERE,
    ParametricTypeDef,
    deserialise_parametric_types,
    serialise_parametric_types,
)
from zarr_vectors.core.store import (
    FsGroup,
    create_store,
    get_parametric_group,
    open_store,
    read_root_metadata,
    write_parametric_types,
    read_parametric_types,
)
from zarr_vectors.exceptions import ArrayError, MetadataError


# ===================================================================
# Write
# ===================================================================

def write_parametric_objects(
    store_path: str,
    objects: list[dict[str, Any]],
    *,
    object_attributes: dict[str, npt.NDArray] | None = None,
    groups: dict[int, list[int]] | None = None,
    group_attributes: dict[str, npt.NDArray] | None = None,
    custom_types: list[ParametricTypeDef] | None = None,
    create_new_store: bool = False,
    store_kwargs: dict[str, Any] | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Write parametric objects to a zarr vectors store.

    Can write to an existing store (appending to ``/parametric/``) or
    create a new store if ``create_new_store=True``.

    Args:
        store_path: Path to the store.
        objects: List of dicts, each with:
            - ``"type"``: type name (``"plane"``, ``"line"``, ``"sphere"``)
              or integer type ID.
            - ``"coefficients"``: list of floats matching the type's schema.
            - Optionally ``"name"``: human-readable name for the object.
        object_attributes: Per-object attributes as ``{name: (O,) or (O,C)}``.
        groups: Group memberships ``{group_id: [object_indices]}``.
        group_attributes: Per-group attributes ``{name: (G,) or (G,C)}``.
        custom_types: Additional parametric type definitions beyond the
            built-in plane/line/sphere.
        create_new_store: If True, create a new store.  Otherwise open
            existing.
        store_kwargs: Extra kwargs passed to ``create_store`` (e.g.
            ``root_metadata`` fields).

    Returns:
        Summary dict with ``object_count``, ``type_counts``.

    Raises:
        ArrayError: If an object has an unknown type or wrong coefficient count.
    """
    # Build type registry
    all_types = list(DEFAULT_PARAMETRIC_TYPES)
    if custom_types:
        all_types.extend(custom_types)

    type_by_name = {t.name: t for t in all_types}
    type_by_id = {t.type_id: t for t in all_types}

    # Open or create store
    if create_new_store:
        kw = dict(store_kwargs or {})
        # Allow callers that still pass ``spatial_index_dims`` (the
        # in-memory RootMetadata field name) — translate to ``axes``.
        if "spatial_index_dims" in kw and "axes" not in kw:
            kw["axes"] = kw.pop("spatial_index_dims")
        kw.setdefault("axes", [
            {"name": "x", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "z", "type": "space"},
        ])
        kw.setdefault("chunk_shape", (1000.0, 1000.0, 1000.0))
        kw.setdefault("bounds", ([0, 0, 0], [1000, 1000, 1000]))
        kw.setdefault("geometry_types", ["point_cloud"])
        root = create_store(store_path, backend=backend, **kw)
    else:
        root = open_store(store_path, mode="r+", backend=backend)

    # Write type registry
    write_parametric_types(root, all_types)

    para = get_parametric_group(root)

    # Encode objects
    n_objects = len(objects)
    type_counts: dict[str, int] = {}
    encoded_rows: list[list[float]] = []
    names: list[str] = []

    for i, obj in enumerate(objects):
        # Resolve type
        obj_type = obj.get("type")
        if isinstance(obj_type, str):
            if obj_type not in type_by_name:
                raise ArrayError(
                    f"Object {i}: unknown type '{obj_type}'. "
                    f"Known types: {list(type_by_name.keys())}"
                )
            tdef = type_by_name[obj_type]
        elif isinstance(obj_type, int):
            if obj_type not in type_by_id:
                raise ArrayError(
                    f"Object {i}: unknown type ID {obj_type}. "
                    f"Known IDs: {list(type_by_id.keys())}"
                )
            tdef = type_by_id[obj_type]
        else:
            raise ArrayError(
                f"Object {i}: 'type' must be a string name or integer ID, "
                f"got {type(obj_type)}"
            )

        coeffs = obj.get("coefficients", [])
        if len(coeffs) != len(tdef.coefficients):
            raise ArrayError(
                f"Object {i} (type '{tdef.name}'): expected "
                f"{len(tdef.coefficients)} coefficients "
                f"({tdef.coefficients}), got {len(coeffs)}"
            )

        # Row: [type_id, coeff0, coeff1, ...]
        row = [float(tdef.type_id)] + [float(c) for c in coeffs]
        encoded_rows.append(row)

        type_counts[tdef.name] = type_counts.get(tdef.name, 0) + 1
        names.append(obj.get("name", f"{tdef.name}_{i}"))

    # Pad rows to same length (different types may have different coeff counts)
    max_len = max(len(r) for r in encoded_rows) if encoded_rows else 0
    for r in encoded_rows:
        while len(r) < max_len:
            r.append(float("nan"))

    # Write encoded objects as a dense chunked Zarr v3 array.
    if encoded_rows:
        data = np.array(encoded_rows, dtype=np.float64)
        para.write_array(
            "objects", data,
            attributes={
                "zv_array": "parametric_objects",
                "num_objects": n_objects,
                "max_row_length": max_len,
                "dtype": "float64",
            },
        )

    # Write names as a 1D vlen-bytes array, one element per name.
    if names:
        para.write_vlen_array(
            "names", [n.encode("utf-8") for n in names],
            attributes={
                "zv_array": "parametric_names",
                "num_objects": n_objects,
            },
        )

    # Write additional object attributes — each one a standalone
    # chunked Zarr v3 array under ``object_attributes/<name>``.
    if object_attributes:
        for attr_name, attr_data in object_attributes.items():
            arr = np.asarray(attr_data)
            full_name = f"object_attributes/{attr_name}"
            para.write_array(
                full_name, arr,
                attributes={
                    "dtype": str(arr.dtype),
                    "shape": list(arr.shape),
                },
            )

    # Write groups as a 1D vlen-bytes array (each blob = the int64
    # member list of one group).
    if groups:
        max_gid = max(groups.keys())
        blobs = [
            np.array(groups.get(gid, []), dtype=np.int64).tobytes()
            for gid in range(max_gid + 1)
        ]
        para.write_vlen_array(
            "groups", blobs,
            attributes={"num_groups": max_gid + 1},
        )

    if group_attributes:
        for attr_name, attr_data in group_attributes.items():
            arr = np.asarray(attr_data)
            full_name = f"group_attributes/{attr_name}"
            para.write_array(
                full_name, arr,
                attributes={
                    "dtype": str(arr.dtype),
                    "shape": list(arr.shape),
                },
            )

    return {
        "object_count": n_objects,
        "type_counts": type_counts,
    }


# ===================================================================
# Read
# ===================================================================

def read_parametric_objects(
    store_path: str,
    *,
    backend: str | None = None,
) -> list[dict[str, Any]]:
    """Read all parametric objects from a zarr vectors store.

    Returns:
        List of dicts, each with:
        - ``"type"``: type name string
        - ``"type_id"``: integer type ID
        - ``"coefficients"``: list of float coefficient values
        - ``"coefficient_names"``: list of coefficient name strings
        - ``"name"``: object name (if stored)
    """
    root = open_store(store_path, backend=backend)
    para = get_parametric_group(root)

    # Read type registry
    types = read_parametric_types(root)
    type_by_id = {t.type_id: t for t in types}

    # Read encoded objects from the standalone chunked array.
    if not para.standalone_array_exists("objects"):
        return []
    data = para.read_array("objects")
    attrs = para.read_array_attrs("objects")
    if "num_objects" not in attrs:
        return []
    n_objects = int(attrs["num_objects"])

    # Read names from the vlen-bytes array if present.
    if para.standalone_array_exists("names"):
        names = [b.decode("utf-8") for b in para.read_vlen_array("names")]
    else:
        names = [f"object_{i}" for i in range(n_objects)]

    # Decode objects
    result: list[dict[str, Any]] = []
    for i in range(n_objects):
        row = data[i]
        type_id = int(row[0])
        tdef = type_by_id.get(type_id)
        if tdef is None:
            result.append({
                "type": f"unknown_{type_id}",
                "type_id": type_id,
                "coefficients": row[1:].tolist(),
                "coefficient_names": [],
                "name": names[i] if i < len(names) else f"object_{i}",
            })
            continue

        n_coeffs = len(tdef.coefficients)
        coeffs = row[1 : 1 + n_coeffs].tolist()

        result.append({
            "type": tdef.name,
            "type_id": type_id,
            "coefficients": coeffs,
            "coefficient_names": tdef.coefficients,
            "name": names[i] if i < len(names) else f"object_{i}",
        })

    return result
