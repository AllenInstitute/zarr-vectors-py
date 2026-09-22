"""Tests for ``Group.collect_presence`` / ``apply_presence`` and the
``observe_presence_writes`` instrument.

The third presence mode.  ``nonempty_chunks`` is ONE attribute shared by
every cell of an array, so stamping it inline is a read-modify-write of
state outside the cell being written: two workers writing *disjoint*
cells still race and the loser's key vanishes while its payload sits on
disk.  Deferring with ``record_presence=False`` removes the race but
leaves the cell invisible to ``list_chunks`` until a coordinator rebuild
runs, and consumers legitimately run before that.  Collecting gives both:
payloads land inline and unlocked, stamps land together under whatever
lock the caller holds.

Covers:

* Visibility — invisible inside the block, every cell visible after apply.
* Coalescing — N cells on one array cost exactly one attribute write.
* Ordering — present-then-absent within a block leaves the key absent.
* Two workers on disjoint cells both keep their keys.
* ``record_presence=False`` is still an opt-out inside the block.
* Nesting, in both directions, against ``batched_writes``.
* A raising body leaves ``pending`` applicable.
* The instrument fires synchronously, on the writing thread, and also
  covers the ``batched_writes`` flush that bypasses
  ``_record_nonempty_chunk``.
"""

from __future__ import annotations

import threading

import pytest

from zarr_vectors.building import observe_presence_writes
from zarr_vectors.core.store import create_store
from zarr_vectors.exceptions import StoreError


def _cell_array(root, name="presence_test", grid_shape=(3, 4, 5)):
    """Allocate a per-chunk vlen array to write cells into."""
    root.create_sharded_chunk_array(name, grid_shape)
    return name


# --- visibility -------------------------------------------------------


def test_collect_presence_defers_the_stamp_but_not_the_payload(tmp_store_path):
    """The whole point: bytes land immediately, the manifest does not."""
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)

    with root.collect_presence() as pending:
        root.write_bytes(name, "0.0.0", b"payload")
        # The payload is on disk NOW -- this is not batched_writes.
        assert root.read_bytes(name, "0.0.0") == b"payload"
        # ...but nothing has been stamped.
        assert root.list_chunks(name) == []

    # Leaving the block restores normal stamping and applies NOTHING.
    assert root.list_chunks(name) == []
    assert pending == [(name, "0.0.0", True)]


def test_apply_presence_makes_every_collected_cell_visible(tmp_store_path):
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)
    keys = ["0.0.0", "1.2.3", "2.3.4"]

    with root.collect_presence() as pending:
        for k in keys:
            root.write_bytes(name, k, b"x")
        assert root.list_chunks(name) == []

    assert root.apply_presence(pending) == 1
    assert root.list_chunks(name) == sorted(keys)
    for k in keys:
        assert root.chunk_exists(name, k)


def test_apply_presence_reports_one_entry_per_array(tmp_store_path):
    root = create_store(str(tmp_store_path))
    a = _cell_array(root, "array_a")
    b = _cell_array(root, "array_b")

    with root.collect_presence() as pending:
        root.write_bytes(a, "0.0.0", b"x")
        root.write_bytes(b, "1.1.1", b"y")

    assert root.apply_presence(pending) == 2
    assert root.list_chunks(a) == ["0.0.0"]
    assert root.list_chunks(b) == ["1.1.1"]


def test_apply_presence_of_an_empty_pending_is_a_noop(tmp_store_path):
    """A caller should not have to test before calling."""
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)
    with root.collect_presence() as pending:
        pass
    assert root.apply_presence(pending) == 0
    assert root.list_chunks(name) == []


# --- coalescing -------------------------------------------------------


def test_apply_presence_coalesces_to_one_attribute_write_per_array(
    tmp_store_path, monkeypatch,
):
    """N cells, one ``nonempty_chunks`` rewrite.

    This is not only an optimisation: the caller holds a lock across the
    apply, and a write per cell would lengthen that section by a factor
    of N for a manifest that is one attribute either way.
    """
    import zarr.core.attributes as zattrs

    writes: list[tuple[str, object]] = []
    original = zattrs.Attributes.__setitem__

    def counting_setitem(self, key, value):
        writes.append((key, value))
        return original(self, key, value)

    monkeypatch.setattr(zattrs.Attributes, "__setitem__", counting_setitem)

    root = create_store(str(tmp_store_path))
    name = _cell_array(root)

    with root.collect_presence() as pending:
        for i in range(10):
            root.write_bytes(name, f"{i % 3}.0.0", b"x")
    writes.clear()
    root.apply_presence(pending)

    manifest_writes = [w for w in writes if w[0] == "nonempty_chunks"]
    assert len(manifest_writes) == 1, (
        f"expected one coalesced manifest write, got {len(manifest_writes)}"
    )


# --- ordering ---------------------------------------------------------


def test_present_then_absent_within_a_block_leaves_the_key_absent(
    tmp_store_path,
):
    """Last write wins, matching an inline stamp's ``present=bool(data)``."""
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)

    with root.collect_presence() as pending:
        root.write_bytes(name, "0.0.0", b"payload")
        root.write_bytes(name, "0.0.0", b"")  # emptied

    root.apply_presence(pending)
    assert root.list_chunks(name) == []


def test_absent_then_present_within_a_block_leaves_the_key_present(
    tmp_store_path,
):
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)

    with root.collect_presence() as pending:
        root.write_bytes(name, "0.0.0", b"")
        root.write_bytes(name, "0.0.0", b"payload")

    root.apply_presence(pending)
    assert root.list_chunks(name) == ["0.0.0"]


def test_apply_presence_discards_a_key_stamped_present_earlier(
    tmp_store_path,
):
    """An emptied cell must be removed, not merely not-added."""
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)
    root.write_bytes(name, "0.0.0", b"payload")   # inline, stamped
    assert root.list_chunks(name) == ["0.0.0"]

    with root.collect_presence() as pending:
        root.write_bytes(name, "0.0.0", b"")

    root.apply_presence(pending)
    assert root.list_chunks(name) == []


# --- the race this fixes ----------------------------------------------


def test_disjoint_workers_each_keep_their_key(tmp_store_path):
    """Two workers, interleaved payloads, serialised applies.

    The interleaving is sequenced rather than raced -- a threaded version
    would be flaky and would prove no more.  What matters is that the
    applies happen one after another, which is what the caller's lock
    guarantees, and that neither worker's key is lost.
    """
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)
    worker_a = root.require_group("")  # separate handles, one store
    worker_b = root.require_group("")

    with worker_a.collect_presence() as pending_a:
        worker_a.write_bytes(name, "0.0.0", b"a")
        # B's payload lands between A's write and A's apply -- the exact
        # window in which inline stamping loses a key.
        with worker_b.collect_presence() as pending_b:
            worker_b.write_bytes(name, "1.1.1", b"b")

    worker_a.apply_presence(pending_a)
    worker_b.apply_presence(pending_b)

    assert root.list_chunks(name) == ["0.0.0", "1.1.1"]


# --- interaction with record_presence=False ---------------------------


def test_record_presence_false_is_still_an_opt_out_inside_the_block(
    tmp_store_path,
):
    """Collecting must not resurrect a stamp the caller declined.

    ``record_presence=False`` says "a coordinator will rebuild this",
    not "stamp it later".
    """
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)

    with root.collect_presence() as pending:
        root.write_bytes(name, "0.0.0", b"x", record_presence=False)
        root.write_bytes(name, "1.1.1", b"y")

    assert pending == [(name, "1.1.1", True)]
    root.apply_presence(pending)
    assert root.list_chunks(name) == ["1.1.1"]


def test_write_cells_collects_like_write_bytes(tmp_store_path):
    """The links writers funnel through ``write_cells``, not ``write_bytes``."""
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)

    with root.collect_presence() as pending:
        n = root.write_cells(name, [("0.0.0", b"a"), ("1.1.1", b"b")])
        assert n == 2
        assert root.list_chunks(name) == []

    root.apply_presence(pending)
    assert root.list_chunks(name) == ["0.0.0", "1.1.1"]


def test_write_cells_honours_record_presence_false_while_collecting(
    tmp_store_path,
):
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)

    with root.collect_presence() as pending:
        root.write_cells(name, [("0.0.0", b"a")], record_presence=False)

    assert pending == []
    root.apply_presence(pending)
    assert root.list_chunks(name) == []


# --- nesting ----------------------------------------------------------


def test_collect_presence_nesting_rejected(tmp_store_path):
    root = create_store(str(tmp_store_path))
    with root.collect_presence():
        with pytest.raises(StoreError, match="does not support nesting"):
            with root.collect_presence():
                pass


def test_collect_presence_inside_batched_writes_rejected(tmp_store_path):
    """That block defers payloads and stamps its own manifest."""
    root = create_store(str(tmp_store_path))
    with root.batched_writes():
        with pytest.raises(StoreError, match="cannot run inside"):
            with root.collect_presence():
                pass


def test_batched_writes_inside_collect_presence_rejected(tmp_store_path):
    root = create_store(str(tmp_store_path))
    with root.collect_presence():
        with pytest.raises(StoreError, match="cannot run inside"):
            with root.batched_writes():
                pass


def test_collect_presence_is_reusable_after_a_rejected_nesting(
    tmp_store_path,
):
    """A refused inner block must not poison the outer one."""
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)
    with root.collect_presence() as pending:
        with pytest.raises(StoreError):
            with root.collect_presence():
                pass
        root.write_bytes(name, "0.0.0", b"x")
    root.apply_presence(pending)
    assert root.list_chunks(name) == ["0.0.0"]


# --- failure inside the block -----------------------------------------


def test_pending_survives_an_exception_in_the_block(tmp_store_path):
    """Payloads are on disk, so the stamps must still be applicable.

    The caller does not catch around the block, so the alternative -- a
    discarded token -- would leave written cells permanently invisible
    with nothing to rebuild them.
    """
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)
    pending = None

    with pytest.raises(RuntimeError):
        with root.collect_presence() as collected:
            pending = collected
            root.write_bytes(name, "0.0.0", b"x")
            raise RuntimeError("flush blew up")

    assert pending == [(name, "0.0.0", True)]
    root.apply_presence(pending)
    assert root.list_chunks(name) == ["0.0.0"]


def test_stamping_resumes_after_a_failed_block(tmp_store_path):
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)
    with pytest.raises(RuntimeError):
        with root.collect_presence():
            raise RuntimeError("boom")

    root.write_bytes(name, "0.0.0", b"x")
    assert root.list_chunks(name) == ["0.0.0"]


# --- apply_presence against a vanished array --------------------------


def test_apply_presence_names_an_array_that_disappeared(tmp_store_path):
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)
    with root.collect_presence() as pending:
        root.write_bytes(name, "0.0.0", b"x")
    root.delete_subtree(name)

    with pytest.raises(StoreError, match=name):
        root.apply_presence(pending)


# --- the instrument ---------------------------------------------------


def test_observe_presence_writes_sees_an_inline_stamp(tmp_store_path):
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)

    seen = []
    with observe_presence_writes(seen.append):
        root.write_bytes(name, "0.0.0", b"x")

    assert [(e.array_name, e.chunk_key, e.present) for e in seen] == [
        (name, "0.0.0", True),
    ]


def test_observe_presence_writes_is_silent_after_the_block(tmp_store_path):
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)

    seen = []
    with observe_presence_writes(seen.append):
        root.write_bytes(name, "0.0.0", b"x")
    root.write_bytes(name, "1.1.1", b"y")

    assert len(seen) == 1


def test_observe_presence_writes_fires_only_at_apply_when_collecting(
    tmp_store_path,
):
    """The instrument must report where the stamp LANDED, not where it
    was recorded -- that is the whole question it answers."""
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)

    seen = []
    with observe_presence_writes(seen.append):
        with root.collect_presence() as pending:
            root.write_bytes(name, "0.0.0", b"x")
            root.write_bytes(name, "1.1.1", b"y")
            assert seen == []
        assert seen == []
        root.apply_presence(pending)

    assert sorted(e.chunk_key for e in seen) == ["0.0.0", "1.1.1"]


def test_observe_presence_writes_fires_synchronously_on_the_writing_thread(
    tmp_store_path,
):
    """BRIDGE checks its lock depth inside the callback, which only means
    anything while the write is happening on the same thread."""
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)

    threads: list[int] = []
    with observe_presence_writes(lambda e: threads.append(threading.get_ident())):
        root.write_bytes(name, "0.0.0", b"x")

    assert threads == [threading.get_ident()]


def test_observe_presence_writes_sees_the_batched_flush(tmp_store_path):
    """The batch paths stamp the manifest directly rather than through
    ``_record_nonempty_chunk``, so hooking only that would make the
    instrument silent on exactly the writes that bypass it."""
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)

    seen = []
    with observe_presence_writes(seen.append):
        with root.batched_writes():
            root.write_bytes(name, "0.0.0", b"x")
            root.write_bytes(name, "1.1.1", b"y")

    assert sorted(e.chunk_key for e in seen) == ["0.0.0", "1.1.1"]
    assert root.list_chunks(name) == ["0.0.0", "1.1.1"]


def test_observe_presence_writes_skips_opted_out_cells_in_a_batch(
    tmp_store_path,
):
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)

    seen = []
    with observe_presence_writes(seen.append):
        with root.batched_writes():
            root.write_bytes(name, "0.0.0", b"x", record_presence=False)
            root.write_bytes(name, "1.1.1", b"y")

    assert [e.chunk_key for e in seen] == ["1.1.1"]


def test_observe_presence_writes_nests(tmp_store_path):
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)

    outer, inner = [], []
    with observe_presence_writes(outer.append):
        root.write_bytes(name, "0.0.0", b"x")
        with observe_presence_writes(inner.append):
            root.write_bytes(name, "1.1.1", b"y")

    assert [e.chunk_key for e in outer] == ["0.0.0", "1.1.1"]
    assert [e.chunk_key for e in inner] == ["1.1.1"]


def test_derive_nonempty_chunks_is_not_reported(tmp_store_path):
    """A rebuild derives the whole manifest from disk -- idempotent, no
    per-cell semantics, nothing for an observer to police."""
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)
    root.write_bytes(name, "0.0.0", b"x", record_presence=False)

    seen = []
    with observe_presence_writes(seen.append):
        assert root.derive_nonempty_chunks(name) == ["0.0.0"]

    assert seen == []
