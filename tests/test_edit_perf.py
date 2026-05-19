"""Surgical-update invariant for the fragment→OID inverted index.

The audit (Iteration 4) introduced a lazy index keyed by
``(level, chunk_coords, fragment_idx)`` that maps to the OIDs whose
manifests reference that fragment.  The index is built on first
lookup and patched on every ``_stage_manifest`` call via a diff.

This test asserts the invariant: **after a sequence of atomic + non-
atomic edits, the in-session index matches a from-scratch rebuild
based on the same manifest state**.  Any divergence is a bug in the
diff logic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.core.store import open_store
from zarr_vectors.ops import EditSession, VertexRef
from zarr_vectors.types.points import write_points


def _build_index_from_scratch(
    session,
    level: int,
) -> dict[tuple, list[int]]:
    """Reference implementation: scan every manifest from disk + pending
    ops and return the ground-truth fragment-owners index for ``level``.
    """
    from zarr_vectors.ops.edit import _LEVEL_MARKER_CHUNK  # noqa: F401

    disk_manifests = session._all_manifests_for(level)
    # Apply pending non-atomic ops over the disk state to get the
    # per-OID "current" manifest.
    per_oid: dict[int, list[tuple[tuple, int]]] = {
        oid: list(m) for oid, m in enumerate(disk_manifests)
    }
    for (lvl, oid), op in session._manifest_ops.items():
        if lvl != level:
            continue
        if op.new_manifest is None:
            continue
        per_oid[int(oid)] = [
            (tuple(int(c) for c in cc), int(fi))
            for cc, fi in op.new_manifest
        ]

    out: dict[tuple, list[int]] = {}
    for oid, manifest in per_oid.items():
        for cc, fi in manifest:
            key = (level, tuple(int(c) for c in cc), int(fi))
            out.setdefault(key, []).append(int(oid))
    return out


def _live_index_for_level(
    session,
    level: int,
) -> dict[tuple, list[int]]:
    """Return only the entries of ``session._fragment_owners`` that
    belong to ``level`` (and skip the sentinel marker).
    """
    from zarr_vectors.ops.edit import _LEVEL_MARKER_CHUNK
    idx = session._fragment_owners or {}
    return {
        k: sorted(v) for k, v in idx.items()
        if k[0] == level and not (k[1] == _LEVEL_MARKER_CHUNK and k[2] == -1)
    }


@pytest.fixture
def three_object_store(tmp_path: Path) -> str:
    """3 points in distinct chunks — each becomes its own fragment +
    its own OID — giving us a non-trivial manifest table to index.
    """
    path = tmp_path / "store.zv"
    positions = np.array(
        [
            [10.0, 10.0, 10.0],
            [60.0, 10.0, 10.0],
            [10.0, 60.0, 10.0],
        ],
        dtype=np.float32,
    )
    write_points(
        str(path), positions,
        chunk_shape=(50.0, 50.0, 50.0),
        bounds=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
        object_ids=np.arange(3, dtype=np.int64),
    )
    return str(path)


class TestFragmentOwnersSurgicalUpdate:

    def test_index_built_lazily_on_first_lookup(
        self, three_object_store: str,
    ) -> None:
        root = open_store(three_object_store, mode="r+")
        with EditSession(root, atomic=True) as ed:
            # Index hasn't been built yet.
            assert ed._fragment_owners is None
            # Trigger via a lookup.
            ed._oids_referencing(0, (0, 0, 0), 0)
            assert ed._fragment_owners is not None
            # Live index for level 0 matches a from-scratch rebuild.
            assert (
                _live_index_for_level(ed, 0)
                == {k: sorted(v) for k, v in _build_index_from_scratch(ed, 0).items()}
            )

    def test_atomic_edit_preserves_invariant(
        self, three_object_store: str,
    ) -> None:
        root = open_store(three_object_store, mode="r+")
        with EditSession(root, atomic=True) as ed:
            # Warm the index.
            ed._oids_referencing(0, (0, 0, 0), 0)
            # Atomic edit on OID 0: allocates a new OID with the
            # rewritten manifest; index should pick up the new OID.
            ref = VertexRef.from_object(
                root, level=0, object_id=0, vertex_index=0,
            )
            ed.edit_vertex(ref, new_pos=[5.0, 5.0, 5.0], atomic=True)
            assert (
                _live_index_for_level(ed, 0)
                == {k: sorted(v) for k, v in _build_index_from_scratch(ed, 0).items()}
            )

    def test_nonatomic_edit_preserves_invariant(
        self, three_object_store: str,
    ) -> None:
        root = open_store(three_object_store, mode="r+")
        with EditSession(root, atomic=False) as ed:
            ed._oids_referencing(0, (0, 0, 0), 0)
            ref = VertexRef.from_object(
                root, level=0, object_id=0, vertex_index=0,
            )
            ed.edit_vertex(ref, new_pos=[5.0, 5.0, 5.0], atomic=False)
            # atomic=False in-chunk edit doesn't change manifests, so
            # the index should be unchanged.
            assert (
                _live_index_for_level(ed, 0)
                == {k: sorted(v) for k, v in _build_index_from_scratch(ed, 0).items()}
            )

    def test_mixed_session_preserves_invariant(
        self, three_object_store: str,
    ) -> None:
        """Sequence of atomic + non-atomic + atomic edits over different
        OIDs.  After each, the index must equal a from-scratch rebuild.
        """
        root = open_store(three_object_store, mode="r+")
        with EditSession(root, atomic=True) as ed:
            ed._oids_referencing(0, (0, 0, 0), 0)
            # 1. Atomic edit on OID 0 (new fragment in chunk (0,0,0)).
            ed.edit_vertex(
                VertexRef.from_object(root, level=0, object_id=0, vertex_index=0),
                new_pos=[5.0, 5.0, 5.0], atomic=True,
            )
            # 2. Atomic remove on OID 1 (empty manifest at a new OID).
            ed.remove_vertex(
                VertexRef.from_object(root, level=0, object_id=1, vertex_index=0),
                atomic=True,
            )
            # 3. Non-atomic add_vertex for OID 2 (existing manifest grows).
            ed.add_vertex(
                level=0, pos=[55.0, 55.0, 55.0], object_id=2,
            )
            # Invariant holds after every staged op.
            expected = {
                k: sorted(v)
                for k, v in _build_index_from_scratch(ed, 0).items()
            }
            assert _live_index_for_level(ed, 0) == expected
