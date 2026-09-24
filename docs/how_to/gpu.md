# GPU arrays and array-form I/O

zarr-vectors can take device (cupy) arrays in and hand them back, and it
has array-form versions of the per-object writers a pipeline calls in its
inner loop. Neither needs a GPU: the core is numpy-only, every call works
with numpy arrays, and the GPU support is an optional extra.

## What happens on the device, and what does not

Cells are stored as variable-length bytes, uncompressed or zstd, and
optionally sharded.

- **Reading** with `read_cells(..., device="cuda")`: the cells' stored
  bytes are read in one pooled pass and reach the GPU in one copy (or
  through kvikio, below). Kernels then strip each cell's framing and
  gather the rows into one array. The result is the host result,
  element for element; see [Decoding on the device](#decoding-on-the-device).
- **Writing** from device arrays: the fragment, manifest and link-partition
  encoders run on the device, and only the encoded form is copied to the
  host, often much less than the input (a run of consecutive indices
  becomes one 16-byte range row).
- **Compression and the write itself** always happen on the host, with
  libzstd. That is what keeps a store written from device arrays
  byte-identical to one written from numpy, and readable anywhere.

The other readers that take `device=` (`read_chunk_vertex_buffer`,
`read_link_arrays`, `ReadResult.to_device` and the rest, below) decode on
the host and copy each returned array to the device once.

## Installing

```bash
pip install "zarr-vectors[gpu]"          # cupy-cuda12x (Linux)
pip install "zarr-vectors[gpu-codecs]"   # + nvCOMP, for zstd cells on the device
pip install "zarr-vectors[gpu-io]"       # + kvikio, for GPUDirect Storage (pip envs only)
```

In a conda environment, install cupy from conda-forge rather than using
the `gpu` extra, which would add a second, pip-built cupy. The
`gpu-codecs` extra is safe there: its wheel carries only nvCOMP. kvikio's
wheel pins its own cupy, so in conda install kvikio from the `rapidsai`
channel at the version matching your RAPIDS stack.

Asking for `device="cuda"` without cupy raises an error naming the extra;
it never returns host arrays in its place. None of the extras is part of
`[all]`.

## Probing what an install can do

```python
import zarr_vectors as zv

caps = zv.runtime_capabilities()                   # never touches CUDA
caps = zv.runtime_capabilities(probe_device=True)  # also checks for a device
```

Probe instead of parsing version strings: a shared checkout can change
branch under a running process. `probe_device=True` initialises the CUDA
driver, so a process that forks workers should probe in the workers.

| Key | True when |
|---|---|
| `device_arrays` | the GPU extension imports (and, with `probe_device`, a device is visible) |
| `device_decode` | `read_cells(device="cuda")` can decode uncompressed cells on the device |
| `gpu_codecs` | nvCOMP imports too, so `decode="device"` can decompress zstd cells there |
| `gpu_io` | kvikio imports too, so local files can be read straight into device memory |
| `gpu_encode` | writers handed device arrays encode on the device |
| `read_cells`, `read_neighbourhood` | batched multi-cell reads are available |
| `batched_link_reads` | `read_link_arrays` / `read_link_attributes` prefetch every cell at once |
| `csr_fragments` | `write_chunk_fragments(csr=...)` is available |
| `array_manifests`, `manifests_csr_read` | manifests can be written and read as arrays |
| `dense_manifests` | object indexes can use the dense layout (below) |
| `object_attribute_columns` | `write_object_attribute_columns` is available |
| `array_link_cells` | `write_link_cells(chunks=, vids=, attributes=)` is available |
| `defer_presence`, `append_safe_sharding` | presence can be deferred; appends into shared shards are safe |

## Reading

```python
from zarr_vectors import building as zb

lg = zb.get_resolution_level(zb.open_store(path), 0)

# A chunk and its present neighbours, every array in one prefetch.
batch = zb.read_neighbourhood(lg, (3, 1, 2), ["vertices", "vertex_attributes/intensity"],
                              device="cuda")
col = batch["vertices"]
col.data      # (rows, 3) on the device: every cell's rows, concatenated
col.offsets   # (cells + 1,) CSR offsets over batch.chunk_coords
col.cell      # the cell index of every row

# Any cells, any per-chunk arrays.
batch = zb.read_cells(lg, cells, ["vertices", "links/0/0.0.0_1.0.0"])
```

### Decoding on the device

`read_cells` (and so `read_neighbourhood`) takes `decode=`:

| `decode=` | Uncompressed arrays | zstd arrays | Anything else (blosc, a store it cannot read raw) |
|---|---|---|---|
| `"auto"` (default) | device | host, then one upload | host, then one upload |
| `"device"` | device | device, through nvCOMP | raises |
| `"host"` | host | host | host |

zstd is decoded on the device only when asked for outright, because
nvCOMP trusts its input. Every frame's structure is checked on the
device before nvCOMP sees it, which catches truncation and damaged block
headers. It cannot check inside a block, though. Tested on 40 frames,
each corrupted by one byte inside a block:
- **libzstd** raised on 32 and returned wrong bytes on 8;
- **nvCOMP** returned wrong bytes on 9, hung on 10, and hit an illegal
  memory access on 21. That last failure kills the process's CUDA
  context.

Use `decode="device"` on stores you trust.

Local files are read into pinned host memory and copied up once, or
straight into device memory with kvikio. By default kvikio is used only
when GPUDirect Storage is switched on (`KVIKIO_COMPAT_MODE=OFF`); in
compatibility mode it reads through a bounce buffer, one copy per cell.
`ZARR_VECTORS_GPU_IO=kvikio` or `=host` forces the choice. Any other
zarr store (fsspec, obstore, icechunk) is read through its own byte-range
`get`, then copied up once.

Measured on 4,097 cells (5 million points, vertices plus one attribute,
warm cache, RTX A2000):

| Store | Host decode + upload | Device decode |
|---|---:|---:|
| flat, uncompressed | 380–720 ms | 170–310 ms |
| flat, zstd | 480–780 ms | 390–420 ms |
| sharded, uncompressed | 220 ms | 125–165 ms |
| sharded, zstd | 285 ms | 230 ms |

The spreads are run-to-run variation on this machine. Sharded stores
read a shard's index once on both paths.

### Other readers

`device=` also works on the flat readers (`read_chunk_vertex_buffer`,
`read_chunk_vertex_rows`, `read_chunk_attribute_rows`,
`read_chunk_fragment_attributes`, `read_link_arrays`,
`read_link_attributes`), on `read_all_object_manifests_csr`, and on a
query result through `ReadResult.to_device("cuda")`.

## Writing from arrays

Each array-form writer leaves the store its per-object counterpart leaves,
byte for byte, and accepts numpy or device arrays. Device arrays (cupy,
or anything exporting DLPack, such as a torch tensor) are encoded on the
device; set `ZARR_VECTORS_GPU_ENCODE=0` to encode them on the host
instead, which writes the same bytes. At a million items, encoding
cupy input on the device against on the host:

| Encoder | Host | Device |
|---|---:|---:|
| fragments | 116 ms | 54 ms |
| manifests | 122 ms | 83 ms |
| link partition | 1,181 ms | 416 ms |



```python
# Fragments as CSR: fragment f is indices[offsets[f]:offsets[f + 1]].
zb.write_chunk_fragments(lg, cc, csr=(indices, offsets), mode="append")

# Manifests: object o owns blocks manifest_offsets[o]:manifest_offsets[o + 1].
zb.write_object_manifests(lg, chunk_coords=cc_blocks, fragment_idx=frag_blocks,
                          manifest_offsets=offsets, mode="append", at=first_id)

# Several object attribute columns in one call.
zb.write_object_attribute_columns(lg, {"length": lengths, "alignment": align},
                                  at=first_id)

# Link records and their attributes: one read-modify-write per cell.
zb.write_link_cells(lg, chunks=seam_chunks, vids=seam_vids,
                    attributes={"weight": w}, allocate=False)
zb.finalize_links(lg, delta=0)   # coordinator: rebuild presence and counts
```

## Dense object indexes

By default an object index stores one variable-length blob per object,
so a million objects are a million Python objects to write and to decode.
A store created with `manifest_layout="dense"` keeps its object indexes as
fixed-width integer arrays instead: `manifest_spans` (start and count per
object) into `manifest_blocks` (chunk coordinates and fragment index, one
row per fragment). The array-form writers then write your arrays as they
are, and `read_all_object_manifests_csr` is a gather:

```python
root = zb.create_store(path, bounds=..., chunk_shape=..., manifest_layout="dense")
lg = zb.get_resolution_level(root, 0)
zb.write_object_manifests(lg, chunk_coords=cc, fragment_idx=frags,
                          mode="append", at=first_id)   # no blob per object
```

Every manifest reader and writer handles both layouts, and an index keeps
the layout it was created with. The layout is format 0.9.4: a 0.9.3 build
refuses such an index with an error rather than misreading it. See the
[object manifest spec](../spec/object_model/object_manifest.md).
