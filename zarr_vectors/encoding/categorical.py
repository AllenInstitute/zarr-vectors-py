"""Dictionary (categorical/enum) encoding for vertex attributes.

Zarr has no native enum dtype, so categorical string (or other
hashable) attributes are stored as an integer ``codes`` array plus a
``categories`` lookup table in the array's metadata.  The on-disk
convention mirrors Apache Arrow's Dictionary type, pandas Categorical,
and the CF ``flag_values`` / ``_FillValue`` conventions:

    .zattrs:
        encoding: "dictionary"
        categories: ["soma", "axon", "dendrite", ...]
        ordered: false
        _FillValue: -1            # optional, marks "missing"

The array data itself is a plain integer array; ``categories[code]``
recovers the original value, and any code equal to ``_FillValue``
(when set) decodes to ``None``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

DICTIONARY_ENCODING = "dictionary"


def _smallest_uint_dtype(n_categories: int, has_fill: bool) -> np.dtype:
    """Pick the smallest unsigned (or signed, when a fill code is
    needed) integer dtype that fits ``n_categories`` codes."""
    if has_fill:
        # Need a negative sentinel — use signed.
        if n_categories <= np.iinfo(np.int8).max:
            return np.dtype(np.int8)
        if n_categories <= np.iinfo(np.int16).max:
            return np.dtype(np.int16)
        return np.dtype(np.int32)
    if n_categories <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if n_categories <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def encode_categorical(
    values: npt.NDArray | list,
    *,
    ordered: bool = False,
    fill_value: int | None = None,
) -> tuple[npt.NDArray[np.integer], dict[str, Any]]:
    """Dictionary-encode an array of hashable values.

    Args:
        values: 1-D array (or sequence) of hashable values — strings,
            ints, bools, or numpy scalars.  ``None`` (or ``np.nan`` for
            float-typed inputs) is treated as a missing value and is
            encoded as ``fill_value`` rather than added to the category
            list.
        ordered: Whether the category order is semantically meaningful
            (pandas Categorical convention).  Stored as-is in metadata
            and does not affect encoding — categories are always in
            sorted order.
        fill_value: Integer code to use for missing values.  Required
            when ``values`` contains any missing entries.  By
            convention ``-1`` (and the encoder will pick a signed
            integer dtype to accommodate it).

    Returns:
        ``(codes, dict_metadata)``:

        - ``codes``: 1-D integer array of the same length as ``values``.
        - ``dict_metadata``: dict ready to merge into the array's
          ``.zattrs``, containing ``encoding``, ``categories``,
          ``ordered``, and (when set) ``_FillValue``.

    Raises:
        ValueError: If ``values`` is not 1-D, or contains missing
            entries with no ``fill_value`` set.
    """
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(
            f"encode_categorical requires a 1-D input, got shape {arr.shape}"
        )

    if arr.dtype.kind == "f":
        missing_mask = np.isnan(arr)
    elif arr.dtype.kind == "O":
        missing_mask = np.array([v is None for v in arr], dtype=bool)
    else:
        missing_mask = np.zeros(arr.shape, dtype=bool)

    has_missing = bool(missing_mask.any())
    if has_missing and fill_value is None:
        raise ValueError(
            "values contain missing entries (None/NaN) but no "
            "fill_value was provided; pass fill_value=-1 (or any "
            "integer code) to encode them."
        )

    # Pull non-missing values and dictionary-encode them.
    if arr.dtype.kind == "O":
        # Object arrays may mix str/bytes/None — normalise to str.
        non_missing_values = np.array(
            [_to_category(v) for i, v in enumerate(arr) if not missing_mask[i]]
        )
    else:
        non_missing_values = arr[~missing_mask]

    if non_missing_values.size == 0:
        unique_vals: npt.NDArray = np.array([], dtype=arr.dtype)
        inverse = np.zeros(non_missing_values.size, dtype=np.int64)
    else:
        unique_vals, inverse = np.unique(non_missing_values, return_inverse=True)
        inverse = inverse.astype(np.int64, copy=False).reshape(-1)

    code_dtype = _smallest_uint_dtype(len(unique_vals), has_fill=fill_value is not None)
    codes = np.empty(arr.size, dtype=code_dtype)
    if has_missing:
        codes[missing_mask] = fill_value  # type: ignore[assignment]
        codes[~missing_mask] = inverse.astype(code_dtype, copy=False)
    else:
        codes[:] = inverse.astype(code_dtype, copy=False)

    categories: list[Any] = [_to_category(v) for v in unique_vals]

    dict_metadata: dict[str, Any] = {
        "encoding": DICTIONARY_ENCODING,
        "categories": categories,
        "ordered": bool(ordered),
    }
    if fill_value is not None:
        dict_metadata["_FillValue"] = int(fill_value)

    return codes, dict_metadata


def decode_categorical(
    codes: npt.NDArray[np.integer],
    categories: list[Any],
    *,
    fill_value: int | None = None,
) -> npt.NDArray:
    """Inverse of :func:`encode_categorical`.

    Args:
        codes: Integer codes array.
        categories: Lookup list — ``categories[code]`` is the value.
        fill_value: If set, codes equal to this are decoded as ``None``
            in an object-dtype output (rather than indexing into
            ``categories``).

    Returns:
        Decoded array.  Dtype is ``object`` if missing values are
        present, otherwise the natural numpy dtype inferred from
        ``categories``.
    """
    codes = np.asarray(codes)
    cats_arr = np.asarray(categories)
    if fill_value is None:
        return cats_arr[codes]

    missing_mask = codes == fill_value
    if not missing_mask.any():
        return cats_arr[codes]

    # Mixed presence — return an object array so we can carry None.
    out: npt.NDArray = np.empty(codes.shape, dtype=object)
    out[missing_mask] = None
    non_missing = ~missing_mask
    out[non_missing] = cats_arr[codes[non_missing]]
    return out


def _to_category(v: Any) -> Any:
    """Coerce a numpy scalar / bytes value into a JSON-friendly
    Python value suitable for the ``categories`` list."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return v
