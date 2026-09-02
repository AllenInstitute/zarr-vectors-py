"""Tests for the ``compressor=`` kwarg on public write functions.

Covers each `compressor` shape accepted by
:func:`zarr_vectors.encoding.compression.resolve_compressor`:

* ``None`` / ``"none"`` / ``False`` — no compression (``bytes`` only).
  This is the default; keeps the fast async PUT path.
* ``"zstd"`` — zarr v3's default compressor (``bytes`` + ``zstd``).
* ``"blosc"`` — Blosc(Zstd, BitShuffle, l5) shorthand.
* Caller-supplied codec list.

For each codec configuration we (a) inspect the per-chunk
``zarr.json`` and assert the on-disk ``codecs`` list matches what the
resolver returns, and (b) round-trip the written data through
``read_points`` to confirm the chunks decode correctly.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.encoding.compression import (
    ZARR_V3_DEFAULT_ZSTD_CODEC,
    resolve_compressor,
)
from zarr_vectors.types.points import read_points, write_points


CHUNK = (200.0, 200.0, 200.0)
BIN = (50.0, 50.0, 50.0)


def _new_store(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"codec_{prefix}_")) / "store.zarrvectors"


def _read_vertices_codecs(store: Path) -> list[dict]:
    """Return the ``codecs`` list of the single ``vertices`` vlen array.

    Under the single-array layout ``vertices`` is one Zarr v3 vlen-bytes
    array; its ``codecs`` are the ``vlen-bytes`` serializer followed by
    any BytesBytes compressor the writer applied.
    """
    meta = json.loads((store / "0" / "vertices" / "zarr.json").read_text())
    return meta["codecs"]


def _compressor_codecs(codecs: list[dict]) -> list[dict]:
    """Drop the ``vlen-bytes`` serializer, leaving only compressors."""
    return [c for c in codecs if c.get("name") != "vlen-bytes"]


def _sorted_positions(p: np.ndarray) -> np.ndarray:
    """Sort rows of a (N, D) array so reorder-on-read can be compared."""
    return p[np.lexsort(p.T)]


@pytest.fixture(scope="module")
def positions() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(0, 1000, (5_000, 3)).astype(np.float32)


def test_resolve_compressor_default_is_bytes_only() -> None:
    """``compressor=None`` resolves to BytesCodec-only (no compression).

    The default is the fast async PUT path; compression is opt-in.
    """
    assert resolve_compressor(None) == [{"name": "bytes"}]
    assert resolve_compressor("none") == [{"name": "bytes"}]
    assert resolve_compressor(False) == [{"name": "bytes"}]


def test_resolve_compressor_zstd_shorthand() -> None:
    """``compressor='zstd'`` resolves to zarr v3's default compressor."""
    codecs = resolve_compressor("zstd")
    assert codecs == [{"name": "bytes"}, dict(ZARR_V3_DEFAULT_ZSTD_CODEC)]


def test_resolve_compressor_blosc_shorthand() -> None:
    codecs = resolve_compressor("blosc")
    assert codecs[0] == {"name": "bytes"}
    assert codecs[1]["name"] == "blosc"
    assert codecs[1]["configuration"]["cname"] == "zstd"
    assert codecs[1]["configuration"]["shuffle"] == "bitshuffle"


def test_resolve_compressor_passes_through_list() -> None:
    custom = [{"name": "blosc", "configuration": {
        "cname": "lz4", "clevel": 3, "shuffle": "shuffle",
        "typesize": 4, "blocksize": 0,
    }}]
    out = resolve_compressor(custom)
    # BytesCodec serializer prepended automatically.
    assert out[0] == {"name": "bytes"}
    assert out[1] == custom[0]


def test_resolve_compressor_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        resolve_compressor(42)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        resolve_compressor([{"missing_name_key": True}])


def test_write_points_default_is_uncompressed(positions: np.ndarray) -> None:
    """Default ``compressor=None`` writes BytesCodec-only chunks (no
    compression).  Keeps the fast async PUT path active.
    """
    store = _new_store("default")
    write_points(store, positions, chunk_shape=CHUNK, bin_shape=BIN)
    codecs = _read_vertices_codecs(store)
    # vlen-bytes serializer, no compressor.
    assert [c["name"] for c in codecs] == ["vlen-bytes"]
    out = read_points(store)
    assert np.allclose(
        _sorted_positions(positions),
        _sorted_positions(out["positions"]),
    )


def test_write_points_zstd_round_trips(positions: np.ndarray) -> None:
    """``compressor='zstd'`` writes Zstd-compressed chunks and round-trips."""
    store = _new_store("zstd")
    write_points(
        store, positions, chunk_shape=CHUNK, bin_shape=BIN, compressor="zstd",
    )
    codecs = _read_vertices_codecs(store)
    compressors = _compressor_codecs(codecs)
    assert compressors[0]["name"] == "zstd"
    out = read_points(store)
    assert np.allclose(
        _sorted_positions(positions),
        _sorted_positions(out["positions"]),
    )


def test_write_points_blosc_shorthand_round_trips(positions: np.ndarray) -> None:
    store = _new_store("blosc")
    write_points(
        store, positions, chunk_shape=CHUNK, bin_shape=BIN, compressor="blosc",
    )
    codecs = _read_vertices_codecs(store)
    compressors = _compressor_codecs(codecs)
    assert compressors[0]["name"] == "blosc"
    assert compressors[0]["configuration"]["shuffle"] == "bitshuffle"
    out = read_points(store)
    assert np.allclose(
        _sorted_positions(positions),
        _sorted_positions(out["positions"]),
    )


def test_write_points_custom_list_passes_through(positions: np.ndarray) -> None:
    """Caller-supplied codec list lands on disk verbatim and round-trips."""
    custom = [{"name": "blosc", "configuration": {
        "cname": "lz4", "clevel": 3, "shuffle": "shuffle",
        "typesize": 4, "blocksize": 0,
    }}]
    store = _new_store("custom")
    write_points(
        store, positions, chunk_shape=CHUNK, bin_shape=BIN, compressor=custom,
    )
    codecs = _read_vertices_codecs(store)
    compressors = _compressor_codecs(codecs)
    assert compressors[0]["name"] == "blosc"
    assert compressors[0]["configuration"]["cname"] == "lz4"
    out = read_points(store)
    assert np.allclose(
        _sorted_positions(positions),
        _sorted_positions(out["positions"]),
    )


def test_compression_reduces_disk_size(positions: np.ndarray) -> None:
    """``compressor='zstd'`` produces a strictly smaller store than the
    default (BytesCodec-only) for the same input."""
    def store_bytes(store: Path) -> int:
        return sum(f.stat().st_size for f in store.rglob("*") if f.is_file())

    s_default = _new_store("size_default")
    write_points(
        s_default, positions, chunk_shape=CHUNK, bin_shape=BIN,
    )
    s_zstd = _new_store("size_zstd")
    write_points(
        s_zstd, positions, chunk_shape=CHUNK, bin_shape=BIN, compressor="zstd",
    )
    assert store_bytes(s_zstd) < store_bytes(s_default), (
        f"zstd store ({store_bytes(s_zstd)} B) was not smaller than "
        f"default uncompressed store ({store_bytes(s_default)} B)"
    )


# ---------------------------------------------------------------------------
# chunk_array_codecs — the creation half of batched_writes, on its own
# ---------------------------------------------------------------------------

def _codecs_of(store_path, array_name):
    meta = json.loads(
        (Path(store_path) / "0" / array_name / "zarr.json").read_text()
    )
    return [c.get("name") for c in meta.get("codecs", [])]


def test_chunk_array_codecs_stamps_arrays_created_inside_it():
    from zarr_vectors.core.arrays import create_vertices_array, write_chunk_vertices
    from zarr_vectors.core.store import create_store, get_resolution_level

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "s.zarrvectors"
        root = create_store(
            path, bounds=[[0, 0, 0], [8, 8, 8]], chunk_shape=(8, 8, 8),
            geometry_types=["point_cloud"],
        )
        lg = get_resolution_level(root, 0)
        # create_store warm-creates ``vertices``; drop it so this test is about
        # what the session does rather than what already existed.
        lg.delete_subtree("vertices")
        lg.delete_subtree("vertex_fragments")

        with lg.chunk_array_codecs("zstd"):
            create_vertices_array(lg, dtype="float32")
            # Writes inside the block are immediate, not deferred — that is the
            # whole reason this exists next to batched_writes.
            write_chunk_vertices(
                lg, (0, 0, 0),
                [np.zeros((4, 3), dtype=np.float32)], dtype=np.float32,
            )
            assert lg.chunk_exists("vertices", "0.0.0"), (
                "write inside chunk_array_codecs was deferred; it must not be"
            )

        assert "zstd" in _codecs_of(path, "vertices")


def test_chunk_array_codecs_restores_the_previous_selection():
    from zarr_vectors.core.arrays import create_attribute_array
    from zarr_vectors.core.store import create_store, get_resolution_level

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "s.zarrvectors"
        root = create_store(
            path, bounds=[[0, 0, 0], [8, 8, 8]], chunk_shape=(8, 8, 8),
            geometry_types=["point_cloud"],
        )
        lg = get_resolution_level(root, 0)
        with lg.chunk_array_codecs("zstd"):
            create_attribute_array(lg, "inside", dtype="float32")
        create_attribute_array(lg, "outside", dtype="float32")

        assert "zstd" in _codecs_of(path, "vertex_attributes/inside")
        assert "zstd" not in _codecs_of(path, "vertex_attributes/outside")


def test_chunk_array_codecs_refuses_to_nest_in_batched_writes():
    from zarr_vectors.core.store import create_store, get_resolution_level
    from zarr_vectors.exceptions import StoreError

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "s.zarrvectors"
        root = create_store(
            path, bounds=[[0, 0, 0], [8, 8, 8]], chunk_shape=(8, 8, 8),
            geometry_types=["point_cloud"],
        )
        lg = get_resolution_level(root, 0)
        with pytest.raises(StoreError, match="batched_writes"):
            with lg.batched_writes(compressor="zstd"):
                with lg.chunk_array_codecs("blosc"):
                    pass


def test_create_store_compressor_reaches_the_warm_created_arrays():
    """``vertices`` / ``vertex_fragments`` are allocated by create_store, and
    every later create_vertices_array short-circuits on them — so if the
    compressor does not reach here it reaches nothing that matters."""
    from zarr_vectors.core.store import create_store

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "s.zarrvectors"
        create_store(
            path, bounds=[[0, 0, 0], [8, 8, 8]], chunk_shape=(8, 8, 8),
            geometry_types=["point_cloud"], compressor="zstd",
        )
        assert "zstd" in _codecs_of(path, "vertices")
        assert "zstd" in _codecs_of(path, "vertex_fragments")
