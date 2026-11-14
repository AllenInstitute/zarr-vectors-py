"""Read times with and without each opt-in read option, on the same stores.

    python compare_options.py STORES_DIR OUT.json [--reps-local 5] [--reps-remote 3]

``STORES_DIR`` holds ``{points,mesh,network}-{indexed,octree,octree_chunks}.zv``
(from ``bench_kinds.build``).  Copies with presence in shard indexes
(``*-presence.zv``) are made next to them when missing.  Every
configuration reads the same queries per store, local (median of
``--reps-local`` cold reads) and behind a 20 ms ``LatencyStore``
(median of ``--reps-remote``), and must return the same answer as the
baseline.  Whole-level reads with the default ``auto`` reader on a
``sharding_indexed`` store are skipped: they go through zarr's sharded
reader one cell at a time (minutes at 200k vertices).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
from zarr.storage import LocalStore

from latency_store import LatencyStore

ALL_READ = "level_handle,prime_nodes,gather_indexes,batch_cells,octree_read_once"
CONFIGS = {  # name: (shard reader, experiments, store suffix)
    "default (auto)": ("auto", "", ""),
    "optimised reader": ("optimised", "", ""),
    "+ fewer rounds": ("optimised", "level_handle,prime_nodes,gather_indexes", ""),
    "+ batch_cells": ("optimised", "batch_cells", ""),
    "+ octree_read_once": ("optimised", "octree_read_once", ""),
    "+ presence in shards": ("optimised", "", "-presence"),
    "all": ("optimised", ALL_READ, "-presence"),
    "all but presence": ("optimised", ALL_READ, ""),
}
LATENCY = 0.02


def presence_copy(src: Path) -> Path | None:
    """A copy of ``src`` with presence in shard indexes, or None when the
    conversion refuses the store."""
    dst = src.with_name(src.name.replace(".zv", "-presence.zv"))
    if not dst.exists():
        import zarr_vectors as zv
        from zarr_vectors.building import store_presence_in_shards

        shutil.copytree(src, dst)
        try:
            ds = zv.open(str(dst), mode="r+")
            for lv in ds.levels:
                store_presence_in_shards(ds.level(lv).store, on=True)
        except Exception as exc:  # noqa: BLE001 - reported, configuration skipped
            print(f"{src.name}: no presence copy ({type(exc).__name__}: {exc})", flush=True)
            shutil.rmtree(dst)
            return None
    return dst


def queries(path: Path) -> list[tuple[str, dict]]:
    import zarr_vectors as zv

    rng = np.random.default_rng(3)
    ds = zv.open(str(path))
    pos = np.asarray(ds.select().read().positions)
    present = np.sort(np.asarray(ds.select().object_ids()))
    out: list[tuple[str, dict]] = []
    for size in (125.0, 31.25):
        c = pos[rng.integers(len(pos))]
        lo = np.clip(c - size / 2, 0, 1000 - size)
        out.append((f"box {size:g}", {"bbox": (lo.tolist(), (lo + size).tolist())}))
    if len(present):
        out.append(("1 object", {"objects": [int(present[len(present) // 2])]}))
        out.append(("10 objects", {"objects": [int(o) for o in rng.choice(present, min(10, len(present)), replace=False)]}))
    out.append(("whole level", {}))
    return out


def answer(r) -> tuple:
    p = np.asarray(r.positions)
    extra = tuple(len(getattr(r, n)) for n in ("faces", "edges") if getattr(r, n, None) is not None)
    return (len(p), round(float(np.sort(p, axis=0).astype(np.float64).sum()), 3), *extra)


def read_once(path: Path, sel: dict, latency: float):
    import zarr_vectors as zv

    if latency:
        store = LatencyStore(LocalStore(str(path), read_only=True), latency, 100e6, 128)
        ds = zv.open(store)
        store.reset()
    else:
        store, ds = None, zv.open(str(path))
    t0 = time.perf_counter()
    r = ds.select(**sel).read()
    dt = time.perf_counter() - t0
    stats = (store.gets, store.bytes, store.rounds) if store else (None, None, None)
    return dt, stats, answer(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stores", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--reps-local", type=int, default=5)
    ap.add_argument("--reps-remote", type=int, default=3)
    ap.add_argument("--only", nargs="*", help="store names (e.g. points-indexed)")
    ap.add_argument("--configs", nargs="*", help="configuration names (default: all)")
    args = ap.parse_args()
    import zarr

    zarr.config.set({"async.concurrency": 4096})
    rows = []
    names = sorted(p.name[:-3] for p in args.stores.glob("*.zv") if "-presence" not in p.name)
    for name in names:
        if args.only and name not in args.only:
            continue
        base = args.stores / f"{name}.zv"
        layout = name.split("-", 1)[1]
        os.environ.pop("ZARR_VECTORS_EXPERIMENTS", None)
        os.environ["ZARR_VECTORS_SHARD_READER"] = "optimised"
        qs = queries(base)
        pres = presence_copy(base)
        ref: dict[str, tuple] = {}
        for cname, (reader, exps, suffix) in CONFIGS.items():
            if args.configs and cname not in args.configs and cname != "optimised reader":
                continue
            if cname == "+ octree_read_once" and layout != "octree_chunks":
                continue
            if suffix and pres is None:
                continue
            path = pres if suffix else base
            os.environ["ZARR_VECTORS_SHARD_READER"] = reader
            os.environ["ZARR_VECTORS_EXPERIMENTS"] = exps
            for qname, sel in qs:
                if reader == "auto" and qname == "whole level" and layout != "octree":
                    rows.append({"store": name, "config": cname, "query": qname, "skipped": True})
                    continue
                local = [read_once(path, sel, 0.0) for _ in range(args.reps_local)]
                remote = [read_once(path, sel, LATENCY) for _ in range(args.reps_remote)]
                ans = {a for _, _, a in local + remote}
                assert len(ans) == 1, (name, cname, qname, ans)
                a = ans.pop()
                ref.setdefault(qname, a)
                assert a == ref[qname], (name, cname, qname, a, ref[qname])
                row = {
                    "store": name, "config": cname, "query": qname, "vertices": a[0],
                    "local_s": statistics.median(t for t, _, _ in local),
                    "remote_s": statistics.median(t for t, _, _ in remote),
                    "requests": remote[0][1][0], "bytes": remote[0][1][1], "rounds": remote[0][1][2],
                }
                rows.append(row)
                print(f"{name:22s} {cname:22s} {qname:12s} {a[0]:7d} v  local {row['local_s'] * 1e3:8.1f} ms  "
                      f"20ms {row['remote_s'] * 1e3:8.1f} ms  {row['rounds']:3d} rounds {row['requests']:5d} req "
                      f"{row['bytes'] / 1e6:7.2f} MB", flush=True)
        args.out.write_text(json.dumps(rows, indent=1))
    args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
