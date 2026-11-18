"""Tests for the ``zvf`` → ``zv`` rename.

Covers:

* New writes emit ``"zv_array"`` discriminator.
* :func:`read_zv_array_tag` finds both new (``zv_array``) and legacy
  (``zvf_array``) keys.
* ``ZVFError`` is preserved as a deprecated alias for ``ZVError`` —
  every subclass still passes ``issubclass`` against both names.
* Legacy in-memory ``.zattrs`` dicts with the old key keep parsing
  through the helper.
"""

from __future__ import annotations

import numpy as np

from zarr_vectors.core.arrays import read_zv_array_tag
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.exceptions import (
    ArrayError,
    ChunkingError,
    ConventionError,
    MetadataError,
    StoreError,
    ValidationError,
    ZVError,
    ZVFError,
)
from zarr_vectors.types.points import write_points

# ===================================================================
# Exception aliasing
# ===================================================================


def test_zvferror_is_zverror():
    """``ZVFError`` must be the same class object as ``ZVError``."""
    assert ZVFError is ZVError


def test_every_subclass_inherits_from_both_names():
    """Every concrete error class catches under both the old and new base."""
    for exc in (
        StoreError,
        MetadataError,
        ArrayError,
        ChunkingError,
        ConventionError,
        ValidationError,
    ):
        assert issubclass(exc, ZVError)
        assert issubclass(exc, ZVFError)


def test_raising_zverror_caught_as_zvferror():
    """Code that still catches the old name keeps working."""
    try:
        raise StoreError("test")
    except ZVFError as e:  # legacy catch site
        assert str(e) == "test"


# ===================================================================
# Wire-format discriminator
# ===================================================================


def test_new_writes_emit_zv_array_key(tmp_path):
    """``write_chunk_vertices`` writes the new ``zv_array`` discriminator."""
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 100, (50, 3)).astype("f4")
    store = tmp_path / "p.zv"
    write_points(str(store), pos, chunk_shape=(50.0, 50.0, 50.0))

    root = open_store(str(store))
    lvl = get_resolution_level(root, 0)
    vmeta = lvl.read_array_meta("vertices")
    assert vmeta["zv_array"] == "vertices"
    assert "zvf_array" not in vmeta


def test_read_helper_accepts_both_keys():
    """The fallback helper handles the legacy key transparently."""
    new = {"zv_array": "vertices", "dtype": "float32"}
    old = {"zvf_array": "vertices", "dtype": "float32"}
    assert read_zv_array_tag(new) == "vertices"
    assert read_zv_array_tag(old) == "vertices"
    # Newer key wins when both are present (defensive double-write).
    assert read_zv_array_tag({"zv_array": "v_new", "zvf_array": "v_old"}) == "v_new"
    # Absent both → None.
    assert read_zv_array_tag({"dtype": "f4"}) is None


def test_legacy_zattrs_dict_still_parses_via_helper():
    """A simulated legacy 0.3 store's array .zattrs is recoverable."""
    legacy_meta = {
        "zvf_array": "object_index",
        "num_objects": 100,
        "sid_ndim": 3,
    }
    tag = read_zv_array_tag(legacy_meta)
    assert tag == "object_index"
