"""The supported surface should not promise what it cannot do.

Each of these pins one place where ``zarr_vectors.api`` advertised
something the layer underneath did not deliver -- a keyword that raised, a
parameter the writer accepted but the facade did not offer, a fallback
that answered from an exception, or a default that meant its opposite.
"""

from __future__ import annotations

import numpy as np
import pytest

import zarr_vectors as zv
from zarr_vectors.exceptions import ArrayError, ZVError

BOUNDS = ([0.0, 0.0, 0.0], [400.0, 400.0, 400.0])
CHUNK = (200.0, 200.0, 200.0)
BIN = (50.0, 50.0, 50.0)


def _verts(n=60, seed=2):
    return np.random.default_rng(seed).uniform(0, 400, (n, 3)).astype(np.float32)


@pytest.fixture
def mesh_store(tmp_path):
    from zarr_vectors.types.meshes import write_mesh

    rng = np.random.default_rng(2)
    path = tmp_path / "m.zarrvectors"
    write_mesh(
        path, _verts(), rng.integers(0, 60, (30, 3)),
        chunk_shape=CHUNK, bin_shape=BIN,
    )
    return path


class TestUnsupportedTermsSaySo:
    """``read_mesh`` / ``read_graph`` raise for ``object_ids=``.

    That is right -- silently returning the whole level is the worst
    outcome, and neither result carries per-vertex object ids for a
    post-filter to use -- but the message named a parameter of a function
    the caller never called.
    """

    def test_selecting_objects_on_a_mesh_explains_itself(self, mesh_store):
        level = zv.open(mesh_store).level(0)
        with pytest.raises(ZVError, match=r"select\(objects=\.\.\.\)"):
            level.select(objects=[0]).read()

    def test_it_does_not_silently_return_the_whole_level(self, mesh_store):
        """The failure mode the raise exists to prevent."""
        level = zv.open(mesh_store).level(0)
        with pytest.raises(ZVError):
            level.select(objects=[0]).read()
        # ...while an unnarrowed read is unaffected.
        assert level.read().vertex_count == 60

    def test_groups_too(self, mesh_store):
        level = zv.open(mesh_store).level(0)
        with pytest.raises(ZVError, match=r"select\(groups=\.\.\.\)"):
            level.select(groups=[0]).read()


class TestWriterParametersReachTheFacade:
    """``add_*`` omitted parameters the underlying writers accept."""

    def _dataset(self, tmp_path, kind):
        return zv.create(tmp_path / f"{kind}.zarrvectors", schema=zv.Schema(
            bounds=BOUNDS, kind=kind, layout=zv.Layout(cell_size=list(CHUNK)),
        ))

    def test_add_mesh_takes_object_attributes(self, tmp_path):
        ds = self._dataset(tmp_path, "mesh")
        ds.add_mesh(
            _verts(), np.random.default_rng(1).integers(0, 60, (30, 3)),
            object_attributes={"label": np.array([3.0], dtype=np.float32)},
        )
        assert ds.level(0).attribute_names("object") == ("label",)

    def test_add_graph_takes_object_attributes(self, tmp_path):
        ds = self._dataset(tmp_path, "graph")
        ds.add_graph(
            _verts(), np.column_stack([np.arange(1, 60), np.arange(59)]),
            object_attributes={"label": np.array([3.0], dtype=np.float32)},
        )
        assert ds.level(0).attribute_names("object") == ("label",)


class TestRejectionsHappenBeforeTheStoreExists:
    """Four writers refuse ``out_of_bounds="ignore"`` -- after creating."""

    @pytest.mark.parametrize("kind", ["lines", "polylines", "meshes", "graphs"])
    def test_no_store_is_left_behind(self, tmp_path, kind):
        from zarr_vectors.types import graphs, lines, meshes, polylines

        store = tmp_path / f"{kind}.zarrvectors"
        call = {
            "lines": (lines.write_lines, (np.zeros((2, 2, 3), np.float32),)),
            "polylines": (
                polylines.write_polylines, ([np.zeros((3, 3), np.float32)],),
            ),
            "meshes": (
                meshes.write_mesh,
                (np.zeros((4, 3), np.float32), np.zeros((1, 3), np.int64)),
            ),
            "graphs": (
                graphs.write_graph,
                (np.zeros((3, 3), np.float32), np.array([[1, 0]])),
            ),
        }[kind]
        fn, args = call
        with pytest.raises(ArrayError, match="out_of_bounds"):
            fn(str(store), *args, chunk_shape=CHUNK, out_of_bounds="ignore")
        assert not store.exists(), (
            "a call that could never succeed left a store behind"
        )


class TestEmptyResultsKeepTheirWidth:
    def test_a_two_dimensional_store_stays_two_dimensional(self, tmp_path):
        """``ndim`` was hard-coded to 3 for an empty polyline result."""
        from zarr_vectors.types.polylines import write_polylines

        path = tmp_path / "p2.zarrvectors"
        write_polylines(
            path, [np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)],
            chunk_shape=(10.0, 10.0),
        )
        level = zv.open(path).level(0)
        assert level.read().ndim == 2
        empty = level.select(bbox=([900.0, 900.0], [999.0, 999.0])).read()
        assert empty.vertex_count == 0
        assert empty.ndim == 2


class TestOneVersionParser:
    """``require_api`` and ``require_format`` parsed the same syntax twice."""

    def test_both_accept_the_same_clauses(self, tmp_path):
        from zarr_vectors._api_version import satisfies

        assert satisfies((0, 9, 1), ">=0.9,<0.11") is None
        assert satisfies((0, 9, 1), ">=99.0") == ">=99.0"

    def test_both_reject_the_same_nonsense(self, tmp_path):
        ds = zv.create(tmp_path / "v.zarrvectors", schema=zv.Schema(
            bounds=BOUNDS, layout=zv.Layout(cell_size=list(CHUNK)),
        ))
        with pytest.raises(ValueError, match="cannot parse"):
            zv.require_format(ds, "~0.9")
        with pytest.raises(ValueError, match="cannot parse"):
            zv.require_api("~1.0")
