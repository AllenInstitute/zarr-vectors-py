# Chunk vs bin

A store has two nested grids, set separately.

| | [`chunk_shape`](chunk_shape.md) | [`bin_shape`](bin_shape.md) |
|---|---|---|
| Controls | Files, requests, the read unit | Selection inside a chunk; resolution of coarse levels |
| Stored in | Root `zarr_vectors.chunk_shape`; optional per-level value | Root `zarr_vectors.base_bin_shape`; per-level NGFF transforms |
| Default | Required | Equal to `chunk_shape` |
| Across levels | Root value, or an integer multiple of it per level | Grows with `bin_ratio` |
| Changed after writing by | `rechunk_spatial` (new store) | `rebin_level` (point clouds, in place) |

[Bin shape](bin_shape.md) shows how a read uses both grids.
[Choose chunk and bin sizes](../../how_to/choose_chunk_and_bin.md) covers
picking values.
