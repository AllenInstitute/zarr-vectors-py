"""Test the v0.7 → v0.8 cross-chunk-link migration helper."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from zarr_vectors.constants import (
    CAP_PARTITIONED_CROSS_CHUNK_LINKS,
    FORMAT_VERSION,
)
from zarr_vectors.core.arrays import (
    read_cross_chunk_link_attributes,
    read_cross_chunk_links,
)
from zarr_vectors.core.group import Group
from zarr_vectors.migration import partition_legacy_cross_chunk_links


def _write_legacy_store(
    tmp_path,
    *,
    records,
    sid_ndim: int,
    link_width: int,
    delta: int,
    attribute_rows: dict[str, np.ndarray] | None = None,
) -> str:
    """Build a synthetic v0.7-shaped store on disk.

    Writes the legacy monolithic ``cross_chunk_links/<delta>/data``
    int64 blob directly, bypassing the new sharded writer.  The result
    looks indistinguishable from a real pre-0.8 store to the migration
    helper.
    """
    store_path = tmp_path / "legacy.zv"
    root = zarr.open_group(store=str(store_path), mode="w")
    root.attrs.update({
        "zarr_vectors": {
            "zv_version": "0.7.0",
            "sid_ndim": sid_ndim,
            "format_capabilities": ["multiscale_links"],
            "bounds": [[0.0] * sid_ndim, [1000.0] * sid_ndim],
            "chunk_shape": [100.0] * sid_ndim,
        },
        "multiscales": [{
            "axes": [
                {"name": "x", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "z", "type": "space"},
            ][:sid_ndim],
        }],
    })

    level = root.require_group("0")

    # Stamp a minimal vertices/<chunk> blob so the chunk-existence
    # validator passes after the migration runs.
    seen_chunks: set[tuple[int, ...]] = set()
    for rec in records:
        for chunk, _vi in rec:
            seen_chunks.add(tuple(chunk))
    vertices_group = level.require_group("vertices")
    for ck in seen_chunks:
        key = ".".join(str(c) for c in ck)
        arr = vertices_group.create_array(
            name=key, shape=(1,), chunks=(1,), dtype="uint8",
            overwrite=True,
        )
        arr.attrs.update({
            "zv_array": "vertices",
            "num_vertices": 16,
            "sid_ndim": sid_ndim,
            "dtype": "float32",
        })
        arr[:] = 0

    delta_seg = f"{delta:+d}" if delta != 0 else "0"
    ccl_group = level.require_group("cross_chunk_links").require_group(delta_seg)

    flat: list[int] = []
    for rec in records:
        for chunk, vi in rec:
            flat.extend(int(c) for c in chunk)
            flat.append(int(vi))
    blob = np.asarray(flat, dtype=np.int64).tobytes()
    data_arr = ccl_group.create_array(
        name="data", shape=(len(blob),), chunks=(len(blob),),
        dtype="uint8", overwrite=True,
    )
    data_arr.attrs.update({
        "zv_array": "cross_chunk_links",
        "num_links": len(records),
        "sid_ndim": sid_ndim,
        "level_delta": int(delta),
        "link_width": int(link_width),
    })
    data_arr[:] = np.frombuffer(blob, dtype=np.uint8)

    if attribute_rows:
        attrs_root = level.require_group("cross_chunk_link_attributes")
        for name, rows in attribute_rows.items():
            ag = attrs_root.require_group(name).require_group(delta_seg)
            arr = ag.create_array(
                name="data",
                shape=rows.shape,
                chunks=rows.shape,
                dtype=str(rows.dtype),
                overwrite=True,
            )
            arr[:] = rows
            arr.attrs.update({
                "zv_array": "cross_chunk_link_attribute",
                "name": name,
                "num_links": int(rows.shape[0]),
                "level_delta": int(delta),
                "dtype": str(rows.dtype),
                "shape": list(rows.shape[1:]) or None,
            })

    return str(store_path)


def test_migration_round_trips_simple_l2_delta0(tmp_path):
    records = [
        [((0, 0, 0), 5), ((1, 0, 0), 2)],
        [((0, 0, 0), 7), ((2, 0, 0), 3)],
        [((1, 0, 0), 4), ((2, 0, 0), 6)],
    ]
    path = _write_legacy_store(
        tmp_path, records=records, sid_ndim=3, link_width=2, delta=0,
    )

    summary = partition_legacy_cross_chunk_links(path)

    assert not summary["already_v08"]
    assert not summary["dry_run"]
    assert summary["version_bumped_to"] == FORMAT_VERSION
    [level_result] = summary["level_results"]
    assert level_result["level"] == 0
    [delta_result] = level_result["deltas"]
    assert delta_result["delta"] == 0
    assert delta_result["record_count"] == 3

    # Read back via the new reader.
    root_zg = zarr.open_group(store=path, mode="r+")
    lg = Group._from_zarr(root_zg["0"])
    out = read_cross_chunk_links(lg, delta=0)
    out_sets = {frozenset(((tuple(c), int(vi)) for c, vi in rec)) for rec in out}
    in_sets = {frozenset(((tuple(c), int(vi)) for c, vi in rec)) for rec in records}
    assert out_sets == in_sets

    # Capability + version stamped.
    zv = dict(root_zg.attrs.get("zarr_vectors") or {})
    assert CAP_PARTITIONED_CROSS_CHUNK_LINKS in zv["format_capabilities"]
    assert zv["zv_version"] == FORMAT_VERSION

    # Legacy data blob removed.
    delta_group = root_zg["0"]["cross_chunk_links"]["0"]
    assert "data" not in [k for k in delta_group]


def test_migration_is_idempotent(tmp_path):
    records = [[((0, 0, 0), 5), ((1, 0, 0), 2)]]
    path = _write_legacy_store(
        tmp_path, records=records, sid_ndim=3, link_width=2, delta=0,
    )
    partition_legacy_cross_chunk_links(path)
    # Second invocation should detect the capability and no-op.
    summary = partition_legacy_cross_chunk_links(path)
    assert summary["already_v08"] is True
    assert summary["version_bumped_to"] is None


def test_migration_dry_run_reports_without_writing(tmp_path):
    records = [
        [((0, 0, 0), 5), ((1, 0, 0), 2)],
        [((0, 0, 0), 7), ((1, 0, 0), 3)],
    ]
    path = _write_legacy_store(
        tmp_path, records=records, sid_ndim=3, link_width=2, delta=0,
    )

    summary = partition_legacy_cross_chunk_links(path, dry_run=True)
    assert summary["dry_run"] is True
    assert summary["version_bumped_to"] is None
    [level_result] = summary["level_results"]
    [delta_result] = level_result["deltas"]
    assert delta_result["record_count"] == 2

    # Legacy blob still present.
    root_zg = zarr.open_group(store=path, mode="r")
    delta_group = root_zg["0"]["cross_chunk_links"]["0"]
    assert "data" in [k for k in delta_group]


def test_migration_preserves_attribute_alignment(tmp_path):
    records = [
        [((0, 0, 0), 5), ((1, 0, 0), 2)],
        [((0, 0, 0), 7), ((2, 0, 0), 3)],
        [((1, 0, 0), 4), ((2, 0, 0), 6)],
    ]
    weights = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    path = _write_legacy_store(
        tmp_path, records=records, sid_ndim=3, link_width=2, delta=0,
        attribute_rows={"weight": weights},
    )

    partition_legacy_cross_chunk_links(path)

    root_zg = zarr.open_group(store=path, mode="r+")
    lg = Group._from_zarr(root_zg["0"])
    out_records = read_cross_chunk_links(lg, delta=0)
    out_weights = read_cross_chunk_link_attributes(lg, "weight", delta=0)

    # Build a lookup from the original {sorted_chunks: weight} to verify.
    legacy_lookup: dict[frozenset, float] = {}
    for i, rec in enumerate(records):
        key = frozenset(((tuple(c), int(vi)) for c, vi in rec))
        legacy_lookup[key] = float(weights[i])

    assert out_weights.shape == (3,)
    for rec, w in zip(out_records, out_weights):
        key = frozenset(((tuple(c), int(vi)) for c, vi in rec))
        assert legacy_lookup[key] == pytest.approx(float(w))


def test_migration_handles_l2_delta_plus1(tmp_path):
    # Cross-pyramid-level edge: endpoint 0 at owning level, endpoint 1
    # at owning + delta.
    records = [
        [((0, 0, 0), 5), ((1, 0, 0), 2)],
    ]
    path = _write_legacy_store(
        tmp_path, records=records, sid_ndim=3, link_width=2, delta=1,
    )

    partition_legacy_cross_chunk_links(path)

    root_zg = zarr.open_group(store=path, mode="r+")
    lg = Group._from_zarr(root_zg["0"])
    out = read_cross_chunk_links(lg, delta=1)
    assert len(out) == 1
    # delta != 0 does NOT canonicalize ci, so the endpoint identity is
    # preserved.
    out_rec = out[0]
    assert (tuple(out_rec[0][0]), out_rec[0][1]) == ((0, 0, 0), 5)
    assert (tuple(out_rec[1][0]), out_rec[1][1]) == ((1, 0, 0), 2)


def test_migration_no_op_on_empty_store(tmp_path):
    store_path = tmp_path / "empty.zv"
    root = zarr.open_group(store=str(store_path), mode="w")
    root.attrs.update({
        "zarr_vectors": {
            "zv_version": "0.7.0",
            "sid_ndim": 3,
            "format_capabilities": [],
        },
    })
    summary = partition_legacy_cross_chunk_links(str(store_path))
    # No CCL groups → no level results.
    assert summary["level_results"] == []
    # Version still bumped (the helper unconditionally moves to 0.8.0
    # once it confirms no legacy blobs exist).
    assert summary["version_bumped_to"] == FORMAT_VERSION
