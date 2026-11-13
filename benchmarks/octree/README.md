# Spatial read benchmark: point clouds, meshes, networks

This benchmark times spatial reads from a sharded zarr-vectors store, for
each shard layout and shard reader, on a local store and on one with
emulated object-storage latency. Use it to choose a layout and reader, or
to check a change to the read path.

## Settings under test

| Setting | Values |
|---|---|
| Shard layout | `sharding_indexed` (default), or zarr-octree's `zarr_octree_sharding`, chosen with `zv.Layout(octree=...)` at creation or `zarr_vectors.sharding.set_octree` afterwards |
| `ZARR_VECTORS_SHARD_READER` | `auto`, `standard`, `optimised` or `zarr` |

Both are described in [Sharding](../../docs/spec/chunking/sharding.md).

## Requirements

- zarr-vectors from this repository.
- For the `octree` and `runs` layouts, zarr-octree (not on PyPI):
  `pip install git+https://github.com/Andrew-Keenlyside/zarr-octree`.
  `--layouts indexed` runs without it.
- For the figures, `matplotlib`.

## Running

Run from this directory:

```bash
python bench_kinds.py --quick --kinds points --layouts indexed \
    --vertices 20000 --latencies 0 --skip-zarr-all --out results/quick.json   # a short check
python bench_kinds.py                     # the full run: 4 M vertices of each kind
python figures_kinds.py
```

The short check takes a few minutes and leaves the committed results
alone. The full run takes hours.

| Option | Default | Meaning |
|---|---|---|
| `--kinds` | `points mesh network` | Kinds of data |
| `--layouts` | `indexed octree` | Stores to build: `indexed`, `octree` (codec defaults), `runs` (the codec's runs layout, 160–640 kB) |
| `--vertices` | `4000000` | Vertices of each kind |
| `--latencies` | `0 0.02 0.08` (`0 0.02` with `--quick`) | Seconds added to each request |
| `--quick` | off | Fewer queries and box sizes |
| `--skip-zarr-all` | off | Skip whole-level reads through zarr's API, which take minutes each at 4 M vertices |
| `--skip-build` | off | Reuse the stores already in `--stores` |

| Output | Default path |
|---|---|
| Stores | `_stores/` (`--stores`) |
| Results | `results/bench_kinds.json` (`--out`); a new run keeps the kinds it did not rerun |
| Figures | `results/figures/zv_kinds_<latency>.png` (query time) and `zv_kinds_bytes.png` (bytes read per query at 20 ms, and store sizes) |

## What it does

- **Data** (`kinds_data.py`, synthetic): point clouds, closed-surface
  meshes and branching-tree networks, with similar vertex counts and the
  same uneven density in a 1000-unit cube. Each uses 64³ cells, 16³ cells
  per shard, and zstd.
- **Queries**, each through `select(...).read()`: `all` (the whole level);
  `vertex` (a 0.5-unit box); `object` (one small, median or large object,
  or 10 or 100 objects); `bbox` (boxes 31.25 to 250 units wide).
- **Latency**: each query runs on the local store and behind
  `LatencyStore` (`latency_store.py`), which adds 20 or 80 ms and
  100 MB/s per request and counts requests and bytes by array. Every read
  starts from a cold cache. The script checks that every store and reader
  returns the same data.

## Results

Committed results: `results/bench_kinds.json`, 4 M vertices, `indexed`
and `octree` stores, run with `--skip-zarr-all`. Times are the geometric
mean, over a query's sizes and placements, of the median time.

| | Point cloud | Mesh | Network |
|---|---|---|---|
| Store size, indexed / octree | 87 / 82 MB | 342 / 103 MB | 156 / 96 MB |
| All, local, indexed: standard / optimised | 641 / 10.9 s | 653 / 48 s | 962 / 44 s |
| Objects, 20 ms, indexed: zarr / optimised | 6.5 / 1.5 s | 4.5 / 2.4 s | not in these results |
| Box, 20 ms, indexed: zarr / optimised | 2.0 / 1.0 s | 3.6 / 2.1 s | 3.9 / 1.5 s |
| One vertex, 20 ms, any store or reader | 0.70–0.73 s | 0.49–0.52 s | 0.66–0.72 s |

- **The reader, not the layout, sets the time.** Octree and
  `sharding_indexed` stores are within about 5 % at 20 and 80 ms. On the
  local store, octree is up to about 17 % slower.
- **Octree shrinks stores with many link arrays.** A `sharding_indexed`
  shard index holds 16 bytes per cell of the shard, occupied or not
  (65 kB at 16³ cells). The mesh has 61 link arrays, so its links take
  283 MB indexed and 51 MB octree. Octree indexes only occupied cells.
- **One vertex costs 0.5–0.7 s at 20 ms whatever the store or reader.**
  It reads 9–30 MB, nearly all array metadata (`nonempty_chunks`, which
  a cold read fetches for every array it touches) and the object index.
- **Mesh and network reads touch many arrays**, because links are split
  into one array per neighbour offset (61 for the mesh, 15 for the
  network), each with its own metadata and shards.
- **Through zarr's array API, a read re-reads a shard's index for every
  cell**, so it can read more bytes than the store holds.
