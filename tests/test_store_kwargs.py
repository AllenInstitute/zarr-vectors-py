"""Loose keyword arguments that the chosen backend cannot use.

`create_store(path, shard_shape=8)` was accepted and silently dropped for
the whole life of that feature. There was no such parameter, so the value
was collected by `**backend_kwargs`, carried as far as the backend
dispatch, and discarded -- `LocalStore` takes a path and nothing else. A
one-line TypeError would have caught it at the call site instead of
leaving every store unsharded and every downstream guard disarmed.

Only the branches that discard options entirely reject them: a local
store, or one the caller built themselves. The remote backends take
genuinely open-ended options, so there is nothing honest to check against.

Explicit `storage_options=` is never rejected. A caller passing one
options dict across several backends means it; a loose keyword that fits
nothing is a misspelling. That asymmetry is the point, so it is pinned.
"""

from __future__ import annotations

import pytest
from zarr.storage import MemoryStore

from zarr_vectors.core.store import (
    create_store,
    open_store,
    read_root_metadata,
)


def test_a_loose_kwarg_the_local_backend_cannot_use_raises(tmp_store_path):
    with pytest.raises(TypeError, match="shrad_shape"):
        create_store(str(tmp_store_path), shrad_shape=2)


def test_the_error_says_where_to_look(tmp_store_path):
    with pytest.raises(TypeError, match="storage_options"):
        create_store(str(tmp_store_path), region="us-east-1")


def test_open_store_rejects_one_too(tmp_store_path):
    create_store(str(tmp_store_path))
    with pytest.raises(TypeError, match="nonsense"):
        open_store(str(tmp_store_path), mode="r", nonsense=1)


def test_a_prebuilt_store_rejects_options_it_cannot_apply():
    """Nothing can act on them -- the store is already built."""
    with pytest.raises(TypeError, match="anon"):
        create_store(MemoryStore(), anon=True)


def test_storage_options_on_a_local_store_is_still_accepted(tmp_store_path):
    """The deliberate asymmetry: structured options are never a typo."""
    create_store(str(tmp_store_path), storage_options={"anon": True})


def test_a_real_parameter_is_not_mistaken_for_a_loose_kwarg(tmp_store_path):
    create_store(str(tmp_store_path), shard_shape=2)
    assert read_root_metadata(
        open_store(str(tmp_store_path), mode="r")
    ).shard_shape == 2


def test_no_kwargs_at_all_is_fine(tmp_store_path):
    create_store(str(tmp_store_path))
    assert open_store(str(tmp_store_path), mode="r") is not None
