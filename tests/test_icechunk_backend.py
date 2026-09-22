"""Tests for the icechunk Zarr-Store-level backend.

These tests round-trip a tiny ZV store through an icechunk
local-filesystem repository and exercise the commit semantics
exposed by ``zarr_vectors.core.store.commit`` /
``discard_changes`` / ``session_for``.

The whole module is skipped when icechunk is not installed.

Note on the high-level ``write_*`` API: the per-type writers
(``write_points``, ``write_graph``, ...) call ``create_store``
internally and return a plain dict — they do not surface the
``Group`` and so cannot commit on their own.  To use icechunk with
the high-level writers, which auto-commit on success via
:func:`zarr_vectors.core.store._finalize_write` once all chunks have
been written.  Callers that need finer-grained snapshots can still
open the store with ``mode="r+"`` and call :func:`commit` directly
between batches.
"""

from __future__ import annotations

from pathlib import Path

import pytest

icechunk = pytest.importorskip("icechunk")

from zarr_vectors.core.store import (
    commit,
    create_store,
    discard_changes,
    open_store,
    session_for,
)
from zarr_vectors.exceptions import StoreError


# ===================================================================
# Fixtures
# ===================================================================


def _minimal_root_kwargs() -> dict:
    return dict(
        axes=[
            {"name": "x", "type": "space", "unit": "um"},
            {"name": "y", "type": "space", "unit": "um"},
            {"name": "z", "type": "space", "unit": "um"},
        ],
        chunk_shape=(50.0, 50.0, 50.0),
        bounds=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
        geometry_types=["point_cloud"],
    )


@pytest.fixture
def ic_repo_path(tmp_path: Path) -> str:
    return str(tmp_path / "ic_store")


# ===================================================================
# Create + commit + reopen round-trip
# ===================================================================


def test_create_and_reopen(ic_repo_path: str) -> None:
    """create_store writes root metadata; commit + reopen reads it back."""
    root = create_store(ic_repo_path, **_minimal_root_kwargs(), backend="icechunk")

    # Session is attached and usable for commit.
    assert session_for(root) is not None
    snapshot = commit(root, "initial setup")
    assert isinstance(snapshot, str) and len(snapshot) > 0

    # Reopen in another context.
    re = open_store(ic_repo_path, backend="icechunk")
    assert re.attrs.to_dict()["zarr_vectors"]["geometry_types"] == ["point_cloud"]
    assert session_for(re) is not None


def test_create_rejects_existing(ic_repo_path: str) -> None:
    """create_store on an existing icechunk repo must raise."""
    root = create_store(ic_repo_path, **_minimal_root_kwargs(), backend="icechunk")
    commit(root, "initial setup")
    with pytest.raises(StoreError, match="already exists"):
        create_store(ic_repo_path, **_minimal_root_kwargs(), backend="icechunk")


def test_open_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(StoreError, match="not found"):
        open_store(str(tmp_path / "does_not_exist"), backend="icechunk")


# ===================================================================
# Commit / discard semantics
# ===================================================================


def test_subgroup_can_commit(ic_repo_path: str) -> None:
    """Sub-groups (created via root.create_group) share the same session
    and can be passed to commit() too."""
    root = create_store(ic_repo_path, **_minimal_root_kwargs(), backend="icechunk")
    # 0/ was auto-created by create_store
    res0 = root["0"]
    assert session_for(res0) is session_for(root)
    snap = commit(res0, "via sub-group")
    assert isinstance(snap, str)


def test_discard_drops_uncommitted_writes(ic_repo_path: str) -> None:
    """Pending changes can be rolled back via discard_changes."""
    root = create_store(ic_repo_path, **_minimal_root_kwargs(), backend="icechunk")
    commit(root, "snapshot 1")

    # Make an uncommitted attribute mutation, then discard it.
    re = open_store(ic_repo_path, backend="icechunk", mode="r+")
    attrs = re.attrs.to_dict()
    attrs.setdefault("scratch", {})["dirty"] = True
    re.attrs.update(attrs)
    discard_changes(re)

    # Re-reopen and confirm the dirty key never landed.
    re2 = open_store(ic_repo_path, backend="icechunk", mode="r")
    assert "scratch" not in re2.attrs.to_dict()


def test_uncommitted_writes_are_not_durable(ic_repo_path: str) -> None:
    """A writable session that's never committed loses its work — this
    is the icechunk contract that callers need to be aware of when
    driving the high-level ``write_*`` functions."""
    root = create_store(ic_repo_path, **_minimal_root_kwargs(), backend="icechunk")
    commit(root, "snapshot 1")  # baseline

    # Mutate without committing, then drop the handle (simulating the
    # high-level write_*() functions returning).
    re = open_store(ic_repo_path, backend="icechunk", mode="r+")
    attrs = re.attrs.to_dict()
    attrs["scratch"] = {"never": "committed"}
    re.attrs.update(attrs)
    del re  # session GC; uncommitted state vanishes

    # Re-reopen and confirm the uncommitted key is gone.
    re2 = open_store(ic_repo_path, backend="icechunk", mode="r")
    assert "scratch" not in re2.attrs.to_dict()


# ===================================================================
# Backend layer wiring
# ===================================================================


def test_session_for_returns_none_on_local(tmp_path: Path) -> None:
    """Non-transactional backends have no session — helpers return None."""
    root = create_store(str(tmp_path / "local_store"), **_minimal_root_kwargs())
    assert session_for(root) is None
    assert commit(root, "no-op") is None
    discard_changes(root)  # no-op, must not raise


def test_unknown_scheme_in_icechunk_raises() -> None:
    with pytest.raises(StoreError, match="unsupported URL scheme"):
        create_store("ftp://example.com/x", **_minimal_root_kwargs(), backend="icechunk")


def test_memory_storage_round_trip() -> None:
    """``memory://`` URLs route to icechunk's in-memory storage."""
    url = "memory://test"
    root = create_store(url, **_minimal_root_kwargs(), backend="icechunk")
    snap = commit(root, "init")
    assert isinstance(snap, str)
    # Note: in-memory icechunk repos are per-Repository — reopening
    # ``memory://test`` makes a fresh empty store, not the one above.
    # The round-trip check is just "create + commit succeeds".


# ===================================================================
# Snapshot / branch reads
# ===================================================================


def test_readonly_at_snapshot(ic_repo_path: str) -> None:
    """Snapshot-pinned readonly sessions see the world at that snapshot."""
    root = create_store(ic_repo_path, **_minimal_root_kwargs(), backend="icechunk")
    snap1 = commit(root, "v1")

    # Mutate + commit a second snapshot.
    re = open_store(ic_repo_path, backend="icechunk", mode="r+")
    attrs = re.attrs.to_dict()
    attrs.setdefault("scratch", {})["v"] = 2
    re.attrs.update(attrs)
    snap2 = commit(re, "v2")
    assert snap1 != snap2

    # Reopen pinned to snap1 — the v2 mutation should not be visible.
    pinned = open_store(
        ic_repo_path, backend="icechunk", mode="r", snapshot_id=snap1,
    )
    assert "scratch" not in pinned.attrs.to_dict()

    # Reopen at HEAD of main — v2 should be visible.
    head = open_store(ic_repo_path, backend="icechunk", mode="r")
    assert head.attrs.to_dict().get("scratch", {}).get("v") == 2


# ===================================================================
# Auto-commit on high-level write_*()
# ===================================================================


def test_write_points_auto_commits(ic_repo_path: str) -> None:
    """``write_points(..., backend="icechunk")`` must persist data without
    requiring a follow-up explicit commit."""
    import numpy as np
    from zarr_vectors.types.points import read_points, write_points

    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 100, (32, 3)).astype(np.float32)

    write_points(
        ic_repo_path, positions,
        chunk_shape=(50.0, 50.0, 50.0),
        backend="icechunk",
    )

    # No explicit commit by the caller — auto-commit by _finalize_write
    # should have flushed the session before write_points returned.
    out = read_points(ic_repo_path, backend="icechunk")
    assert out["vertex_count"] == 32


def test_rebuild_presence_lists_cells_on_icechunk(ic_repo_path: str) -> None:
    """icechunk refuses a listing prefix that is not a node, and
    ``<array>/c/`` is not one; the rebuild lists the array and filters."""
    import numpy as np
    from zarr_vectors.building import get_resolution_level, rebuild_presence
    from zarr_vectors.types.points import write_points

    rng = np.random.default_rng(1)
    write_points(
        ic_repo_path, rng.uniform(0, 100, (64, 3)).astype(np.float32),
        chunk_shape=(50.0, 50.0, 50.0), backend="icechunk",
    )
    root = open_store(ic_repo_path, backend="icechunk", mode="r+")
    level = get_resolution_level(root, 0)
    recorded = level.list_chunks("vertices")
    assert recorded

    assert rebuild_presence(level, "vertices") == recorded
