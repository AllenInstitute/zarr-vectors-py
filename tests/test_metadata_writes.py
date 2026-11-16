"""How many times an attribute change rewrites ``zarr.json``.

zarr 3's ``attrs.update(d)`` is ``MutableMapping.update``: one
``__setitem__``, and so one whole-document rewrite, per key.  On a shared
filesystem those rewrites are the expensive part of a small write, and on
a chunk array each is also a read-modify-write of the document that holds
``nonempty_chunks``.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors.core.store import create_store


@pytest.fixture
def metadata_writes(monkeypatch) -> list[str]:
    from zarr.storage import LocalStore

    keys: list[str] = []
    original = LocalStore.set

    async def counting(self, key, value, *a, **kw):
        if key.endswith("zarr.json"):
            keys.append(key)
        return await original(self, key, value, *a, **kw)

    monkeypatch.setattr(LocalStore, "set", counting)
    return keys


def test_extend_array_resizes_and_restamps_in_one_write(
    tmp_store_path, metadata_writes,
):
    root = create_store(str(tmp_store_path))
    root.write_array("col", np.arange(4, dtype=np.int64), attributes={"shape": [4]})
    metadata_writes.clear()

    assert root.extend_array("col", np.arange(3, dtype=np.int64)) == 7

    assert metadata_writes == ["col/zarr.json"]
    assert root.read_array_meta("col")["shape"] == [7]
    np.testing.assert_array_equal(
        root.read_array("col"), [0, 1, 2, 3, 0, 1, 2],
    )


def test_extend_array_with_attributes_is_still_one_write(
    tmp_store_path, metadata_writes,
):
    root = create_store(str(tmp_store_path))
    root.write_array("col", np.arange(2, dtype=np.int64), attributes={"shape": [2]})
    metadata_writes.clear()

    root.extend_array("col", np.arange(2, dtype=np.int64), attributes={"a": 1, "b": 2})

    assert metadata_writes == ["col/zarr.json"]
    meta = root.read_array_meta("col")
    assert (meta["shape"], meta["a"], meta["b"]) == ([4], 1, 2)


def test_group_attrs_update_is_one_write(tmp_store_path, metadata_writes):
    root = create_store(str(tmp_store_path))
    metadata_writes.clear()
    root.attrs.update({"a": 1, "b": 2, "c": 3})
    assert metadata_writes == ["zarr.json"]
    assert {k: root.attrs[k] for k in "abc"} == {"a": 1, "b": 2, "c": 3}
