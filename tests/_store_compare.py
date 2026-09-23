"""Byte-level comparison of two stores on disk.

The array-form writers promise the store they produce is the one the
per-object writers produce from the same input. This is the check: every
chunk file byte for byte, every ``zarr.json`` as parsed JSON (key order
and whitespace are not content).
"""

from __future__ import annotations

import json
from pathlib import Path


def _files(root: Path) -> dict[str, Path]:
    return {
        str(p.relative_to(root)): p
        for p in root.rglob("*")
        if p.is_file()
    }


def assert_stores_identical(a, b, *, ignore=()) -> None:
    """Fail naming the first differences between stores ``a`` and ``b``.

    ``ignore`` holds relative-path prefixes to skip, for a difference a
    test declares deliberately.
    """
    a, b = Path(a), Path(b)
    fa = {k: v for k, v in _files(a).items() if not k.startswith(tuple(ignore))}
    fb = {k: v for k, v in _files(b).items() if not k.startswith(tuple(ignore))}
    only_a = sorted(set(fa) - set(fb))
    only_b = sorted(set(fb) - set(fa))
    assert not only_a and not only_b, (
        f"file sets differ: only in {a.name}: {only_a[:10]}; "
        f"only in {b.name}: {only_b[:10]}"
    )
    differ = []
    for rel in sorted(fa):
        pa, pb = fa[rel], fb[rel]
        if pa.name == "zarr.json":
            if json.loads(pa.read_text()) != json.loads(pb.read_text()):
                differ.append(rel)
        elif pa.read_bytes() != pb.read_bytes():
            differ.append(rel)
    assert not differ, f"{len(differ)} files differ, e.g. {differ[:10]}"
