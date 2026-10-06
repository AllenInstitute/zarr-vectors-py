"""A forked child gets fresh reader and writer pools.

The shared pools (``_batch_reader._read_pool``, ``_batch_writer._write_pool``)
are module-level ``ThreadPoolExecutor`` objects. A child forked from a process
that has used one inherits the executor but none of its threads, and an idle
parent thread leaves the executor believing a worker is free: the child's
first submit queues work that nothing takes, and waits for ever. Each module
now drops its pool (and the lock guarding it) in the child.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys

import pytest

import zarr_vectors as zv
from zarr_vectors.core import _batch_reader, _batch_writer

pytestmark = pytest.mark.skipif(
    not hasattr(os, "register_at_fork") or sys.platform == "win32",
    reason="needs fork",
)


def _use(pool_of) -> int:
    return pool_of().submit(lambda: 41).result() + 1


def _child(name: str, out) -> None:
    pool_of = {"read": _batch_reader._read_pool, "write": _batch_writer._write_pool}[name]
    out.put(_use(pool_of))


@pytest.mark.parametrize("name", ["read", "write"])
def test_a_forked_child_uses_a_fresh_pool(name):
    pool_of = {"read": _batch_reader._read_pool, "write": _batch_writer._write_pool}[name]
    assert _use(pool_of) == 42                      # the parent's pool has a live, idle thread
    parent_pool = pool_of()

    ctx = mp.get_context("fork")
    out = ctx.Queue()
    child = ctx.Process(target=_child, args=(name, out))
    child.start()
    child.join(timeout=60)
    hung = child.is_alive()
    if hung:
        child.kill()
        child.join()
    assert not hung, f"the forked child's first pooled {name} never ran"
    assert child.exitcode == 0
    assert out.get(timeout=5) == 42
    assert pool_of() is parent_pool                 # the parent's pool is untouched


def test_the_fix_is_named():
    assert "fork-safe-pools" in zv.FEATURES
