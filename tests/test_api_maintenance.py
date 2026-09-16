"""The maintenance verbs, against stores that are not local directories.

``Dataset.build_pyramid`` and ``Dataset.validate`` re-opened the store
from ``self.url``.  ``Group.url`` is ``repr(store)`` for anything that is
not a ``LocalStore``, so both were unreachable on a memory- or
object-backed dataset -- and ``validate`` was broken on local ones too,
because ``url`` is a ``file://`` URI and level 1 did a raw ``Path()`` on
it.

Neither method had a single test, and nothing in the suite drove the
``Dataset`` surface against a non-local store.  That is why this file
exists rather than additions to an existing one.
"""

from __future__ import annotations

import numpy as np
import pytest

import zarr_vectors as zv

BOUNDS = ([0.0, 0.0, 0.0], [400.0, 400.0, 400.0])


def _schema():
    return zv.Schema(
        bounds=BOUNDS, kind="point_cloud",
        layout=zv.Layout(cell_size=[200.0, 200.0, 200.0]),
    )


def _points(n=2000, seed=7):
    return np.random.default_rng(seed).uniform(0, 400, (n, 3)).astype(np.float32)


@pytest.fixture
def memory_dataset():
    """A Dataset over a pre-built store with no URL at all.

    ``zv.create`` passes its target straight to ``create_store``, which
    accepts a ``zarr.abc.store.Store`` -- so this needs no special
    support, and it is the sharpest case: there is no string anything
    could have re-opened.
    """
    from zarr.storage import MemoryStore

    ds = zv.create(MemoryStore(), schema=_schema())
    ds.add_points(_points())
    return ds


@pytest.fixture
def local_path(tmp_path):
    path = tmp_path / "m.zarrvectors"
    ds = zv.create(path, schema=_schema())
    ds.add_points(_points())
    return path


class TestValidate:
    def test_runs_on_a_local_dataset(self, local_path):
        """``Dataset.validate`` failed for *every* backend, local included.

        Level 1 did ``Path(store_path)`` on a ``file://`` URI, found
        nothing, and ``validate()`` returns early when level 1 fails.
        """
        result = zv.open(local_path).validate(level=3)
        assert result.ok, result.errors

    def test_runs_on_a_memory_backed_dataset(self, memory_dataset):
        result = memory_dataset.validate(level=3)
        assert result.ok, result.errors

    def test_accepts_a_path_a_url_and_a_handle(self, local_path):
        """All three reach the same verdict.

        The Group arm is what ``Dataset.validate`` now uses; the
        ``file://`` arm is the spelling the docs called broken.
        """
        from zarr_vectors.validate import validate

        verdicts = [
            validate(str(local_path), level=3),
            validate(local_path.as_uri(), level=3),
            validate(zv.open(local_path).store, level=3),
        ]
        assert all(v.ok for v in verdicts)
        # Same checks run, whichever way in.
        assert len({len(v.passed) for v in verdicts}) == 1

    def test_reports_a_missing_root_marker_rather_than_raising(self, tmp_path):
        """Level 1's job is to describe this, not to be stopped by it.

        ``open_store`` raises on a root with no ``zarr_vectors`` block, so
        the validator opens with ``require_zv=False`` to keep its own
        diagnostic reachable.
        """
        import zarr

        from zarr_vectors.validate.structure import validate_structure

        bare = tmp_path / "bare.zarr"
        zarr.open_group(str(bare), mode="w")
        result = validate_structure(str(bare))
        assert not result.ok
        assert any("No root metadata found" in e for e in result.errors)


class TestBuildPyramid:
    def test_runs_on_a_memory_backed_dataset(self, memory_dataset):
        """The reported failure: StoreError: Store not found at MemoryStore(...)."""
        memory_dataset.build_pyramid(factors=[(2.0, 1.0)])
        assert memory_dataset.levels == (0, 1)

    def test_runs_on_a_local_dataset(self, local_path):
        ds = zv.open(local_path, mode="r+")
        ds.build_pyramid(factors=[(2.0, 1.0)])
        assert ds.levels == (0, 1)

    def test_a_read_only_dataset_cannot_write_a_pyramid(self, local_path):
        """Deliberate behaviour change.

        ``zv.open`` defaults to ``mode="r"``, and the old path re-opened
        the URL as ``"r+"`` -- so a read-only dataset silently escalated
        and wrote through a handle the caller had not asked to be
        writable.  Threading the handle removes the escalation.
        """
        with pytest.raises(Exception, match="read-only|read only"):
            zv.open(local_path).build_pyramid(factors=[(2.0, 1.0)])

    def test_refreshing_a_pyramid_works_on_a_memory_backed_dataset(
        self, memory_dataset,
    ):
        """``ops.refresh`` reached ``coarsen_level`` through ``root.url`` too."""
        memory_dataset.build_pyramid(factors=[(2.0, 1.0)])
        from zarr_vectors.ops.refresh import rebuild_pyramid_from_level

        summaries = rebuild_pyramid_from_level(memory_dataset.store, 0)
        assert len(summaries) == 1
        assert memory_dataset.levels == (0, 1)


class TestShardingTakesAHandle:
    def test_get_shard_info_accepts_a_group(self, local_path):
        from zarr_vectors.sharding.io import get_shard_info, is_sharded

        handle = zv.open(local_path).store
        assert get_shard_info(handle)["sharded"] is False
        assert is_sharded(handle) is False

    def test_shard_store_accepts_a_group(self, memory_dataset):
        """Previously unreachable: Path(repr(store)) then open_store on it."""
        from zarr_vectors.sharding.io import get_shard_info, shard_store

        shard_store(memory_dataset.store, shard_shape=2)
        assert get_shard_info(memory_dataset.store)["sharded"] is True
        # ...and the data still reads back.
        assert memory_dataset.level(0).read().vertex_count == 2000
