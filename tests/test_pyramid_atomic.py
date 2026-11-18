"""``build_pyramid`` is all-or-nothing, and a rollback restores what it replaced.

Each ``coarsen_level`` step already removed its own partial level, but a
failure at level k kept levels 1..k-1 and skipped the finalize pass, so a
retry met a half-built pyramid. And a ``links/+1`` family the source level
already had (a half-removed pyramid leaves one) was rewritten by the step's
``write_links(mode="replace")`` and could not be put back.

The checks compare every file of the store, so anything the failed call
changed -- data, link families, level or root metadata -- shows up.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.constants import XLEVEL_EXPLICIT
from zarr_vectors.core.store import list_resolution_levels, open_store
from zarr_vectors.multiresolution import coarsen
from zarr_vectors.multiresolution.coarsen import build_pyramid, coarsen_level
from zarr_vectors.types.polylines import write_polylines
from zarr_vectors.validate import validate


def _store(tmp_path: Path) -> str:
    rng = np.random.default_rng(1)
    polys = [rng.uniform(0, 100, (30, 3)).astype("f4") for _ in range(12)]
    path = str(tmp_path / "p.zarrvectors")
    write_polylines(
        path, polys, chunk_shape=(50.0,) * 3, bounds=([0.0] * 3, [100.0] * 3),
        object_attributes={"score": np.arange(12, dtype=np.float32)},
    )
    return path


def _files(path: str) -> dict[str, object]:
    """Every file under the store; ``zarr.json`` parsed, so key order and
    whitespace do not count as a change."""
    out: dict[str, object] = {}
    for f in sorted(Path(path).rglob("*")):
        if f.is_file():
            rel = f.relative_to(path).as_posix()
            out[rel] = (json.loads(f.read_text()) if f.name == "zarr.json"
                        else f.read_bytes())
    return out


def _fail_on_call(monkeypatch, n: int) -> None:
    """Make the ``n``-th ``write_cross_level_links`` call (1-based) raise."""
    real = coarsen.write_cross_level_links
    calls = {"n": 0}

    def _maybe_fail(*args, **kw):
        calls["n"] += 1
        if calls["n"] == n:
            raise RuntimeError("injected failure")
        return real(*args, **kw)

    monkeypatch.setattr(coarsen, "write_cross_level_links", _maybe_fail)


_BUILD = dict(factors=[(2.0, 1.0)] * 3, cross_level_depth=1,
              cross_level_storage=XLEVEL_EXPLICIT)


def test_a_failure_at_level_2_removes_level_1_too(tmp_path, monkeypatch):
    path = _store(tmp_path)
    before = _files(path)
    # Explicit storage writes +1 at the fine level and -1 at the coarse one:
    # calls 1-2 build level 1's links, call 3 is level 2's +1.
    _fail_on_call(monkeypatch, 3)

    with pytest.raises(RuntimeError, match="injected failure"):
        build_pyramid(path, **_BUILD)

    assert list_resolution_levels(open_store(path)) == [0]
    assert _files(path) == before

    monkeypatch.undo()
    build_pyramid(path, **_BUILD)
    assert list_resolution_levels(open_store(path)) == [0, 1, 2, 3]
    assert validate(path, level=5).ok


def test_a_failure_in_the_finalize_pass_removes_every_level(tmp_path, monkeypatch):
    """Depth 2: the inline steps make 6 calls; the 7th is the finalize
    pass's +2, after every level and the root stamps have landed."""
    path = _store(tmp_path)
    before = _files(path)
    _fail_on_call(monkeypatch, 7)

    with pytest.raises(RuntimeError, match="injected failure"):
        build_pyramid(path, **{**_BUILD, "cross_level_depth": 2})

    assert list_resolution_levels(open_store(path)) == [0]
    assert _files(path) == before


def test_a_failing_registered_strategy_leaves_nothing(tmp_path, monkeypatch):
    """A registered method gets the store-level rollback per step."""
    import zarr_vectors.multiresolution.registry as reg
    from zarr_vectors.building import register_coarsen_strategy

    path = _store(tmp_path)
    before = _files(path)

    def _half_then_fail(**kw):
        coarsen._per_object_coarsen(**kw)
        raise RuntimeError("strategy failed after writing")

    monkeypatch.setitem(reg._COARSEN_STRATEGIES, "half_then_fail", None)
    register_coarsen_strategy("half_then_fail", _half_then_fail)
    with pytest.raises(RuntimeError, match="after writing"):
        coarsen_level(path, 0, 1, coarsen_factor=2.0, method="half_then_fail",
                      cross_level_storage=XLEVEL_EXPLICIT)
    assert _files(path) == before


# --- a stale links/+1 on the source ----------------------------------------


def _half_removed_pyramid(tmp_path: Path) -> str:
    """Level 0 keeps a ``links/+1`` family whose level 1 is gone."""
    path = _store(tmp_path)
    build_pyramid(path, factors=[(4.0, 1.0)], cross_level_depth=1,
                  cross_level_storage=XLEVEL_EXPLICIT)
    open_store(path, mode="r+").delete_subtree("1")
    assert (Path(path) / "0" / "links" / "+1").is_dir()
    return path


def test_a_stale_plus_one_is_put_back_by_a_failed_step(tmp_path, monkeypatch):
    path = _half_removed_pyramid(tmp_path)
    before = _files(path)
    # Call 1 replaces the stale +1; call 2 (the -1 mirror) fails.
    _fail_on_call(monkeypatch, 2)

    with pytest.raises(RuntimeError, match="injected failure"):
        coarsen_level(path, 0, 1, coarsen_factor=2.0,
                      cross_level_storage=XLEVEL_EXPLICIT)

    assert _files(path) == before


def test_a_stale_plus_one_is_put_back_by_a_failed_pyramid(tmp_path, monkeypatch):
    path = _half_removed_pyramid(tmp_path)
    before = _files(path)
    _fail_on_call(monkeypatch, 3)

    with pytest.raises(RuntimeError, match="injected failure"):
        build_pyramid(path, **_BUILD)

    assert _files(path) == before
