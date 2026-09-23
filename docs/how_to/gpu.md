# GPU arrays and array-form I/O

zarr-vectors can take device (cupy) arrays in and hand them back, and it
has array-form versions of the per-object writers a pipeline calls in its
inner loop. Neither needs a GPU: the core is numpy-only, every call works
with numpy arrays, and the GPU support is an optional extra.

## What happens on the device, and what does not

Cells are stored as compressed variable-length bytes. They are fetched and
decompressed on the host, and encoded there when written. So:

- a **reader** asked for `device="cuda"` decodes on the host and copies
  each array it returns to the device **once**;
- a **writer** given device arrays copies each argument to the host
  **once**, then encodes exactly as it would from numpy.

A store written from device arrays is therefore byte-identical to one
written from the same numpy arrays, and readable anywhere. Decoding on the
device and reading straight into device memory (GPUDirect Storage) are
not implemented yet; `runtime_capabilities()` reports them as
`gpu_encode`, `gpu_io` and `gpu_codecs`.

## Installing

```bash
pip install "zarr-vectors[gpu]"          # pip: adds cupy-cuda12x (Linux)
conda install -c conda-forge cupy        # conda: use conda-forge's cupy instead
```

Asking for `device="cuda"` without cupy raises an error naming the extra;
it never returns host arrays in its place.

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
| `read_cells`, `read_neighbourhood` | batched multi-cell reads are available |
| `batched_link_reads` | `read_link_arrays` / `read_link_attributes` prefetch every cell at once |
| `csr_fragments` | `write_chunk_fragments(csr=...)` is available |
| `array_manifests`, `manifests_csr_read` | manifests can be written and read as arrays |
| `object_attribute_columns` | `write_object_attribute_columns` is available |
| `array_link_cells` | `write_link_cells(chunks=, vids=, attributes=)` is available |
| `defer_presence`, `append_safe_sharding` | presence can be deferred; appends into shared shards are safe |
| `dense_manifests`, `gpu_encode`, `gpu_io`, `gpu_codecs` | not yet |

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

`device=` also works on the flat readers (`read_chunk_vertex_buffer`,
`read_chunk_vertex_rows`, `read_chunk_attribute_rows`,
`read_chunk_fragment_attributes`, `read_link_arrays`,
`read_link_attributes`), on `read_all_object_manifests_csr`, and on a
query result through `ReadResult.to_device("cuda")`.

## Writing from arrays

Each array-form writer leaves the store its per-object counterpart leaves,
byte for byte, and accepts numpy or device arrays:

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
