"""The executor: one loop, three read modes.

What is being tested is the fixpoint, not any particular reader.  A
decoder is an ordinary synchronous function that reads through a Group;
the executor supplies its I/O in batches, lets it say what else it
needed, and goes round again.  If that holds for the toy decoders here it
holds for ``read_points``, because the executor knows nothing about
either.

Async is driven with a local ``asyncio.run``, matching
``tests/test_async_reads.py``.
"""

from __future__ import annotations

import asyncio

import pytest
import zarr
from zarr.storage import MemoryStore

from zarr_vectors._engine.execute import aexecute, execute
from zarr_vectors._engine.fetch import (
    Capabilities,
    DictFetcher,
    GroupFetcher,
    RecordingFetcher,
)
from zarr_vectors._engine.plan import ReadPlan
from zarr_vectors.core.group import Group
from zarr_vectors.exceptions import StoreError


def _run(coro):
    return asyncio.run(coro)


def _memory_root() -> Group:
    return Group._from_zarr(zarr.open_group(store=MemoryStore(), mode="a"))


def _live_store(cells: dict[str, bytes]) -> tuple[Group, Group]:
    """A root plus a ``0/`` level holding ``vertices`` with ``cells``."""
    root = _memory_root()
    level = root.create_group("0")
    level.create_sharded_chunk_array(
        "vertices", grid_shape=(2, 2, 2), shard_shape=None,
    )
    for key, payload in cells.items():
        level.write_bytes("vertices", key, payload)
    return root, level


def _read_one(array: str, key: str):
    """A decoder that reads a single cell."""
    def decode(group: Group) -> bytes:
        return group.read_bytes(array, key)
    decode.__name__ = f"read_{key}"
    return decode


class TestConvergence:
    def test_empty_plan_converges_by_discovery(self):
        # The decoder is not modelled anywhere: it asks, the miss is
        # recorded, the next round fetches it. This is what lets one
        # executor serve every reader, present and future.
        f = RecordingFetcher(DictFetcher(cells={("0/vertices", "0.0.0"): b"hello"}))
        out = execute(_memory_root(), _read_one("0/vertices", "0.0.0"), fetcher=f)
        assert out == b"hello"
        assert f.rounds == 1  # round 1 fetched nothing; the discovery round did

    def test_a_good_plan_costs_one_round(self):
        f = RecordingFetcher(DictFetcher(cells={("0/vertices", "0.0.0"): b"hello"}))
        out = execute(
            _memory_root(),
            _read_one("0/vertices", "0.0.0"),
            fetcher=f,
            plan=ReadPlan.of(cells=[("0/vertices", "0.0.0")]),
        )
        assert out == b"hello"
        assert f.rounds == 1
        assert len(f.plans[0].cells) == 1

    def test_a_plan_is_not_refetched_round_on_round(self):
        # minus() is what keeps the loop from re-requesting what it
        # already holds -- and re-requesting is also what would mask a
        # stall, since "asked again, learned nothing" is the termination
        # signal.
        state = {"n": 0}

        def decode(group: Group) -> bytes:
            state["n"] += 1
            first = group.read_bytes("0/vertices", "0.0.0")
            second = group.read_bytes("0/vertices", "1.0.0")
            return first + second

        f = RecordingFetcher(DictFetcher(cells={
            ("0/vertices", "0.0.0"): b"a", ("0/vertices", "1.0.0"): b"b",
        }))
        assert execute(_memory_root(), decode, fetcher=f) == b"ab"
        fetched = [c for plan in f.plans for c in plan.cells]
        assert len(fetched) == len(set(fetched)), "a cell was fetched twice"

    def test_returns_falsy_results_intact(self):
        # A decoder legitimately returning b"" / None must not be
        # mistaken for an incomplete pass.
        f = DictFetcher(cells={("a", "0"): b""})
        assert execute(_memory_root(), _read_one("a", "0"), fetcher=f) == b""


class TestFailure:
    def test_genuinely_missing_cell_stalls_fast(self):
        f = RecordingFetcher(DictFetcher())
        with pytest.raises(StoreError, match="stalled"):
            execute(_memory_root(), _read_one("0/vertices", "0.0.0"), fetcher=f)
        # Two rounds, not a hundred: the cell was asked for, came back
        # absent, and asking again cannot change that.
        assert f.rounds <= 2

    def test_stall_message_names_the_decoder_and_the_want(self):
        with pytest.raises(StoreError) as excinfo:
            execute(
                _memory_root(), _read_one("0/vertices", "0.0.0"),
                fetcher=DictFetcher(),
            )
        assert "read_0.0.0" in str(excinfo.value)
        assert "0/vertices" in str(excinfo.value)

    def test_stall_points_at_listing_when_the_store_cannot_list(self):
        # The one dimension a browser fetch-backed Store cannot serve.
        # The error has to say so, or it reads as a corrupt store.
        def decode(group: Group) -> list[str]:
            names = group.children()
            if not names:
                raise StoreError("no children")
            return names

        with pytest.raises(StoreError, match="requires listing the store"):
            execute(
                _memory_root(), decode,
                fetcher=DictFetcher(capabilities=Capabilities(can_list=False)),
            )

    def test_a_real_error_with_no_misses_propagates(self):
        # Not every StoreError is an incomplete snapshot. One raised by
        # the decoder's own logic must not be swallowed as "go round
        # again".
        def decode(group: Group) -> None:
            raise StoreError("something is actually wrong")

        with pytest.raises(StoreError, match="something is actually wrong"):
            execute(_memory_root(), decode, fetcher=DictFetcher())

    def test_non_convergence_is_bounded(self):
        # A decoder that wants something new every round would otherwise
        # hang. The round cap turns that into a diagnosable error.
        state = {"n": 0}

        def decode(group: Group) -> bytes:
            state["n"] += 1
            return group.read_bytes("0/vertices", f"0.0.{state['n']}")

        with pytest.raises(StoreError, match="did not converge within 4 rounds"):
            execute(_memory_root(), decode, fetcher=DictFetcher(), max_rounds=4)

    def test_error_wording_is_caller_supplied(self):
        # aio.read_async says "Async read"; a sync caller must not.
        with pytest.raises(StoreError, match="Async read stalled"):
            execute(
                _memory_root(), _read_one("a", "0"),
                fetcher=DictFetcher(), what="Async read",
            )


class TestAgainstALiveStore:
    def test_strict_read_through_a_level_group(self):
        # The fetcher is built on the root (plans are root-relative) while
        # the decoder reads through the level group. Keeping those two
        # separate is what lets a level-scoped reader be served by a
        # root-scoped plan.
        root, level = _live_store({"0.0.0": b"aaa"})
        out = execute(level, _read_one("vertices", "0.0.0"), fetcher=GroupFetcher(root))
        assert out == b"aaa"

    def test_batched_mode_rebases_the_prefetch_onto_the_level_group(self):
        # Group._prefetch_cache is keyed relative to the handle it lives
        # on, so a root-relative snapshot must be rebased or every lookup
        # misses and silently falls through to a serial read.
        root, level = _live_store({"0.0.0": b"aaa", "1.0.0": b"bbb"})
        seen: list[list[tuple[str, list[str]]]] = []
        real = Group.batched_reads

        def spy(self, plan):
            seen.append(plan)
            return real(self, plan)

        Group.batched_reads = spy
        try:
            out = execute(
                level,
                _read_one("vertices", "0.0.0"),
                fetcher=GroupFetcher(root),
                plan=ReadPlan.for_cells("0/vertices", ["0.0.0", "1.0.0"]),
                strict=False,
            )
        finally:
            Group.batched_reads = real
        assert out == b"aaa"
        assert seen == [[("vertices", ["0.0.0", "1.0.0"])]]

    def test_batched_mode_tolerates_an_underspecified_plan(self):
        # Falling through to a sync read is the graceful-degradation
        # contract: under-specifying costs round-trips, never correctness.
        root, level = _live_store({"0.0.0": b"aaa", "1.0.0": b"bbb"})
        out = execute(
            level,
            _read_one("vertices", "1.0.0"),
            fetcher=GroupFetcher(root),
            plan=ReadPlan.for_cells("0/vertices", ["0.0.0"]),
            strict=False,
        )
        assert out == b"bbb"

    def test_discovery_does_not_cost_one_round_per_cell(self):
        # The regression this pins: a reader walking two parallel chunk
        # arrays used to reveal a single cell per round, because a cell
        # miss never asked for its array's node and so never triggered
        # the nonempty_chunks fan-out. read_points over 64 chunks took 66
        # rounds; it now takes 3.
        root = _memory_root()
        level = root.create_group("0")
        keys = [f"{i}.0.0" for i in range(2)]
        for name in ("vertices", "vertex_fragments"):
            level.create_sharded_chunk_array(
                name, grid_shape=(2, 2, 2), shard_shape=None,
            )
            for i, key in enumerate(keys):
                level.write_bytes(name, key, f"{name}-{i}".encode())

        def decode(group: Group) -> int:
            total = 0
            for key in keys:
                total += len(group.read_bytes("vertices", key))
                total += len(group.read_bytes("vertex_fragments", key))
            return total

        f = RecordingFetcher(GroupFetcher(root))
        assert execute(level, decode, fetcher=f) > 0
        # One round per array, not one per cell.
        assert f.rounds <= 3, [p.cost() for p in f.plans]

    def test_batched_mode_makes_exactly_one_fetch_round(self):
        root, level = _live_store({"0.0.0": b"aaa"})
        f = RecordingFetcher(GroupFetcher(root))
        execute(
            level, _read_one("vertices", "0.0.0"), fetcher=f,
            plan=ReadPlan.for_cells("0/vertices", ["0.0.0"]),
            strict=False,
        )
        assert f.rounds == 1


class TestAsync:
    def test_aexecute_converges(self):
        f = DictFetcher(cells={("0/vertices", "0.0.0"): b"hello"})
        out = _run(aexecute(_memory_root(), _read_one("0/vertices", "0.0.0"), fetcher=f))
        assert out == b"hello"

    def test_aexecute_stalls_with_the_async_wording(self):
        with pytest.raises(StoreError, match="Async read stalled"):
            _run(aexecute(
                _memory_root(), _read_one("a", "0"), fetcher=DictFetcher(),
            ))

    def test_sync_and_async_agree(self):
        cells = {("0/vertices", "0.0.0"): b"same"}
        decode = _read_one("0/vertices", "0.0.0")
        assert (
            execute(_memory_root(), decode, fetcher=DictFetcher(cells=cells))
            == _run(aexecute(_memory_root(), decode, fetcher=DictFetcher(cells=cells)))
        )
