# GPU arrays and array-form I/O

This page shows how to read cells onto a GPU as cupy arrays, and how to write
from device arrays with the array-form writers. GPU support is an optional
extra: every call also works with numpy arrays and no GPU.

## What you need

- A CUDA GPU and one of the extras:

  ```bash
  pip install "zarr-vectors[gpu]"          # cupy-cuda12x (Linux)
  pip install "zarr-vectors[gpu-codecs]"   # + nvCOMP, for zstd cells on the device
  pip install "zarr-vectors[gpu-io]"       # + kvikio, for GPUDirect Storage (pip envs only)
  ```

  In a conda environment, install cupy from conda-forge instead of the `gpu`
  extra, and kvikio from the `rapidsai` channel. The `gpu-codecs` extra is
  safe in conda.
- A store opened with the builder functions, as `lg` below.

`device="cuda"` without cupy raises `ZVError`; it never returns host arrays
instead.

## Check what the install can do

```python
import zarr_vectors as zv

caps = zv.runtime_capabilities()                   # never touches CUDA
caps = zv.runtime_capabilities(probe_device=True)  # also checks for a device
```

Every value is a bool. `probe_device=True` initialises CUDA, so a process that
forks workers should probe in the workers. The main keys:

| Key | True when |
|---|---|
| `device_arrays` | the GPU extension imports (and, with `probe_device`, a device is visible) |
| `device_decode` | `read_cells` can decode uncompressed cells on the device |
| `gpu_codecs` | nvCOMP imports, so `decode="device"` can decompress zstd cells |
| `gpu_io` | kvikio imports, so local files can be read straight into device memory |
| `gpu_encode` | writers handed device arrays encode on the device |
| `dense_manifests` | object indexes can use the dense layout |

## Read cells onto the device

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
```

`zb.read_cells(lg, cells, arrays, device="cuda")` reads any list of cells the
same way. Pass `fragments=True` to get each cell's fragment index too, as
`batch.fragments`. The result equals the host result, element for element.

`decode=` chooses where cells are decoded:

| `decode=` | Uncompressed | zstd | Anything else |
|---|---|---|---|
| `"auto"` (default) | device | host, then one upload | host, then one upload |
| `"device"` | device | device, through nvCOMP | raises |
| `"host"` | host | host | host |

Use `decode="device"` only on stores you trust. nvCOMP does not check the
inside of a compressed block, and damaged data can kill the CUDA context.

Local files are read into pinned host memory and copied up once. When cuFile
reports GPUDirect Storage, kvikio reads them straight into device memory.
`io="kvikio"` or `io="host"` forces the choice for one call, and
`ZARR_VECTORS_GPU_IO` for every call. `batch.io` records which path each
array took.

Other readers take `device=` too, such as `read_chunk_vertex_buffer` and
`read_link_arrays`. They decode on the host and copy each array up once, as
does `ReadResult.to_device("cuda")`. To decode zstd frames you fetched
yourself, use `zarr_vectors.gpu.decode_zstd`.

## Write from arrays

Each array-form writer leaves the same bytes as its per-object counterpart.
It takes numpy or device arrays (cupy, or anything exporting DLPack, such as
a torch tensor). Device arrays are encoded on the device; compression and the
write happen on the host. Set `ZARR_VECTORS_GPU_ENCODE=0` to encode on the
host instead.

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

For several processes writing at once, see [HPC pipelines](hpc_pipelines.md).

## Dense object indexes

By default an object index stores one variable-length blob per object. A store
created with `manifest_layout="dense"` stores fixed-width integer arrays
instead, so the array-form writers write your arrays as they are and reads
are a gather:

```python
root = zb.create_store(path, bounds=..., chunk_shape=..., manifest_layout="dense")
lg = zb.get_resolution_level(root, 0)
zb.write_object_manifests(lg, chunk_coords=cc, fragment_idx=frags,
                          mode="append", at=first_id)   # no blob per object
```

Every manifest reader and writer handles both layouts. See the
[object manifest spec](../spec/object_model/object_manifest.md).
