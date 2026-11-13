# Spatial read benchmark: point clouds, meshes, networks

This benchmark measures how long spatial reads take from a sharded
zarr-vectors store, for each shard layout and each shard reader, on a local
store and on one that emulates object-storage latency. Use it to choose a
layout and a reader, or to check a change to the read path.

## Settings under test

| Setting | Values |
|---|---|
| `zv.Layout(octree=...)` at creation | `False` (default): zarr's `sharding_indexed` codec. `True`, or the codec's configuration such as `{"target": 100000}` (its default): zarr-octree's `zarr_octree_sharding` codec, which also marks the store octree (root `octree_sharding` and capability `octree_sharding`). Needs zarr-octree and a sharded store. |
| `zarr_vectors.sharding.set_octree(store, octree)` | Converts an existing sharded store to the octree layout, or back with `False`. |
| `ZARR_VECTORS_SHARD_READER` | `auto` (default): per array, `optimised` for a `zarr_octree_sharding` array and `standard` for any other. `standard`: a local store's shards read from the files, each shard index once per batch and one byte range per cell; any other store through zarr's array API. `optimised`: each shard index once per batch and neighbouring cells' byte ranges coalesced, on any store. `zarr`: zarr's array API only. |

## Requirements

- zarr-vectors from this repository.
- For the `octree` and `runs` layouts, zarr-octree, which is not on PyPI:
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

The short check takes a few minutes; it writes to `results/quick.json` so
that the committed results stay as they are.

| Option | Default | Meaning |
|---|---|---|
| `--kinds` | `points mesh network` | Kinds of data |
| `--layouts` | `indexed octree` | Stores to build: `indexed`, `octree` (codec defaults), `runs` (the codec's runs layout, 160–640 kB) |
| `--vertices` | `4000000` | Vertices of each kind |
| `--latencies` | `0 0.02 0.08` (`0 0.02` with `--quick`) | Seconds added to each request |
| `--quick` | off | Fewer queries and box sizes |
| `--skip-zarr-all` | off | Skip whole-level reads through zarr's API, which take minutes each at 4 M vertices |
| `--skip-build` | off | Reuse the stores already in `--stores` |

The full run takes hours: in the committed results, each whole-level read
with the `standard` reader on the local store takes 10–17 minutes.

Outputs:

- stores in `_stores/` (`--stores`);
- results in `results/bench_kinds.json` (`--out`); a new run keeps the
  kinds it did not rerun;
- figures in `results/figures/`: `zv_kinds_<latency>.png` (query time per
  kind, query, store and reader) and `zv_kinds_bytes.png` (bytes read per
  query at 20 ms, by array, and each store's size on disk).

## What it does

**Data** (`kinds_data.py`, synthetic). Point clouds, meshes (closed
surfaces) and networks (branching trees), each with about the same vertex
count and the same uneven density in a 1000-unit cube. Each is written
with 64³ cells, 16³ cells per shard, and zstd.

**Queries**, each through `select(...).read()`:

- `all`: the whole level;
- `vertex`: a box 0.5 units wide on an existing vertex, or placed uniformly;
- `object`: one small, median or large object, and 10 or 100 objects;
- `bbox`: boxes 31.25 to 250 units wide.

**Latency.** Each query runs on the local store, and behind `LatencyStore`
(`latency_store.py`), which adds 20 or 80 ms and 100 MB/s to every request
and counts requests and bytes by array. Every read starts from a cold
cache. The script checks that every store and reader returns the same
vertices and faces or edges.

## The structure being read

Each level is a group of zarr arrays chunked on the same spatial grid of
cells:

| Array | Holds | Point cloud | Mesh | Network |
|---|---|---|---|---|
| `vertices` | Positions, one ragged block of rows per cell | yes | yes | yes |
| `vertex_fragments` | Per cell, which rows belong to which fragment | yes | yes | yes |
| `links/0/<offsets>` | Faces or edges, one array per neighbour offset between their vertices' cells | – | 61 arrays | 15 arrays |
| `link_fragments` | Fragment index over the links | – | yes | yes |
| `object_index/manifests` | Per object, its (cell, fragment) pairs | yes | yes | yes |

A read goes from an object's manifest or a box to cells, then fragment
indexes, vertex rows, and the links in those cells whose endpoints were
all read.

Properties of this structure that the results show:

- Each array's attributes list every non-empty cell (`nonempty_chunks`). A
  cold read fetches them for every array it touches, so they dominate small
  reads.
- Links are split into one array per neighbour offset, each with its own
  metadata and shards, so a mesh or network read touches many arrays.
- Through zarr's array API, a read re-reads a shard's index for every cell,
  so it can read more bytes than the store holds.
- `select(groups=...)` is refused for meshes, graphs and skeletons.

## Results

Committed results: `results/bench_kinds.json`, 4 M vertices, `indexed` and
`octree` stores, run with `--skip-zarr-all`. Times are the geometric mean
over a query's sizes and placements of the median time, as in the figures.

| | Point cloud | Mesh | Network |
|---|---|---|---|
| Store size, indexed / octree | 87 / 82 MB | 342 / 103 MB | 156 / 96 MB |
| All, local, indexed: standard / optimised | 641 / 10.9 s | 653 / 48 s | 962 / 44 s |
| Objects, 20 ms, indexed: zarr / optimised | 6.5 / 1.5 s | 4.5 / 2.4 s | not in these results |
| Box, 20 ms, indexed: zarr / optimised | 2.0 / 1.0 s | 3.6 / 2.1 s | 3.9 / 1.5 s |
| One vertex, 20 ms, any store or reader | 0.70–0.73 s | 0.49–0.52 s | 0.66–0.72 s |

How to read them:

- **The reader, not the layout, sets the time.** Octree and
  `sharding_indexed` stores are within about 5 % of each other at 20 and
  80 ms. On the local store, octree is up to about 17 % slower.
- **Octree shrinks stores with many link arrays.** A `sharding_indexed`
  shard index holds 16 bytes for every cell of the shard, occupied or not
  (65 kB at 16³ cells), in each of the mesh's 61 link arrays: its links take
  283 MB indexed and 51 MB octree. Octree indexes only occupied cells.
  Through zarr's API it also roughly halves the bytes a query reads, but at
  these latencies the time follows the request count, not the bytes.
- **One vertex costs 0.5–0.7 s at 20 ms whatever the store or reader.** It
  reads 9–30 MB, nearly all array metadata (`nonempty_chunks`) and the
  object index.
