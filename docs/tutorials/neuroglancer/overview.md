# Neuroglancer integration overview

[Neuroglancer](https://github.com/google/neuroglancer) is a WebGL-based
viewer for petascale volumetric data, widely used in connectomics, brain
imaging, and synchrotron science. It natively understands several data
formats (OME-Zarr image volumes, Neuroglancer precomputed, N5), but does
not have native support for the Zarr Vectors.

[`zv-ngtools`](https://github.com/BRIDGE-Neuroscience/zv-ngtools) bridges
this gap. It is a fork of
[`ngtools`](https://github.com/neuroscales/ngtools) — a collection of
Neuroglancer utilities — extended with a `zarr_vectors` layer type that
translates Zarr Vectors stores into Neuroglancer layers on the fly.

---

## Why Neuroglancer cannot read Zarr Vectors natively

Neuroglancer's data source plugins expect data in specific binary formats
(precomputed, N5, OME-Zarr). Zarr Vectors stores vertex data as spatially chunked
Zarr arrays — `vertices/` cells addressed by chunk coordinate, with a
`vertex_fragments/` index that slices each cell into per-bin or per-object
runs — a structure Neuroglancer does not understand without a mediating
translation layer.

`zv-ngtools` provides that layer: it runs a local HTTP file server that
intercepts Neuroglancer's chunk requests, reads the requested spatial
region from a Zarr Vectors store using `zarr-vectors-py`, and returns the data in
a format Neuroglancer expects.

---

## Two integration paths

### Path A — Local serving via `LocalNeuroglancer`

A Python process (`zv-ngtools`) runs a local Tornado HTTP server alongside
a Neuroglancer browser tab. The server:

1. Receives chunk requests from Neuroglancer (HTTP range requests).
2. Uses `zarr-vectors-py` to read the requested fragment slices.
3. Translates the Zarr Vectors data to the Neuroglancer layer protocol.
4. Returns the response to the browser.

LOD is driven by the Zarr Vectors resolution pyramid: as the user zooms out in
Neuroglancer, the server switches to coarser levels automatically, using
each level's bin shape to select the appropriate resolution. That number is
`ds.level(i).resolution` on the supported api; see
[LOD and resolution level control](local_viewer.md#lod-and-resolution-level-control).

```
Python process                         Browser
┌───────────────────────────────┐     ┌─────────────────────────────┐
│  LocalNeuroglancer             │     │  Neuroglancer (WebGL)        │
│  ├─ Tornado HTTP server        │◄────│  chunk requests              │
│  │   (port 9123)               │────►│  rendered layers             │
│  └─ zarr-vectors reader        │     │                              │
│      ├─ scan.zarrvectors       │     └─────────────────────────────┘
│      └─ tracts.zarrvectors     │
└───────────────────────────────┘
```

This path is best for local analysis and collaborative review sessions
where the data lives on local disk or a mounted network share.

### Path B — Precomputed export for static hosting

Convert a Zarr Vectors store to the Neuroglancer precomputed format and upload to
a public HTTP server (S3, GCS, nginx). Neuroglancer fetches the data
directly without any intermediary Python process.

This path is best for sharing published datasets with collaborators who
will not install `zv-ngtools`, or for embedding Neuroglancer links in
publications and websites.

See [Precomputed export](precomputed_export.md) for the conversion
workflow.

---

## What `zv-ngtools` adds over upstream `ngtools`

`zv-ngtools` is a fork of `ngtools` (neuroscales/ngtools). All upstream
`ngtools` functionality is preserved; the fork adds:

| Feature | Upstream `ngtools` | `zv-ngtools` |
|---------|-------------------|-------------|
| Load OME-Zarr image volumes | ✓ | ✓ |
| Load TRK / TCK tractography | ✓ | ✓ |
| Load NIfTI, MGH, TIFF | ✓ | ✓ |
| Load `.zarrvectors` stores | ✗ | ✓ |
| Zarr Vectors-aware LOD selection | ✗ | ✓ |
| `zarr_vectors` layer type | ✗ | ✓ |
| Precomputed export from Zarr Vectors | ✗ | ✓ |
| Point cloud layer rendering | ✗ | ✓ |
| Streamline layer rendering | ✗ (only TRK/TCK) | ✓ (from Zarr Vectors) |

The `zarr://` URL scheme (for OME-Zarr stores) works identically in both
forks. Only the `zarr_vectors://` scheme is new.

---

## Layer type mapping

| Zarr Vectors geometry type | Neuroglancer layer type | Notes |
|------------------|------------------------|-------|
| `point_cloud` | `annotation` (point) | Rendered as 3-D points; size and colour from attributes |
| `line` | `annotation` (line) | Rendered as line segments |
| `polyline` / `streamline` | `annotation` (line) | Rendered as connected line sequences |
| `graph` / `skeleton` | `segmentation` (mesh-less) or custom skeleton layer | SWC-style rendering |
| `mesh` | `segmentation` (mesh) or `annotation` (surface) | Draco-compressed meshes supported |

For types without a perfect Neuroglancer native equivalent (e.g. general
graphs), `zv-ngtools` uses the closest Neuroglancer layer type and adds a
server-side translation step.

---

## Coordinate system alignment

Zarr Vectors stores and Neuroglancer image volumes can be displayed together when
they share a coordinate system. Neuroglancer uses a global coordinate space
for all layers; each layer specifies its own coordinate transform.

A Zarr Vectors store is a Zarr **v3** group, so its root document is `zarr.json`.
There is no `.zattrs` — that is the Zarr v2 spelling — and no `metadata.json`.
Everything a viewer needs sits in that one file, under two attribute keys:

- `attributes.zarr_vectors` — the store-level fields: `zv_version`, `bounds`,
  `chunk_shape`, `base_bin_shape`, `geometry_types`, the convention flags, and
  an optional `crs` naming the coordinate system. `crs` is written only when
  one was declared, so a reader must treat it as absent by default.
- `attributes.multiscales` — an OME-NGFF 0.4 block. `axes` names and types each
  axis (plus a `unit` when the schema declared one), and
  **`datasets[i].coordinateTransformations` is the key that carries the `scale`
  and `translation` a viewer needs for the layer transform.**

```python
import json, pathlib

root = json.loads(pathlib.Path("scan.zarrvectors/zarr.json").read_text())
multiscales = root["attributes"]["multiscales"][0]

print([axis["name"] for axis in multiscales["axes"]])
for dataset in multiscales["datasets"]:
    print(dataset["path"], dataset["coordinateTransformations"])
```

```
['x', 'y', 'z']
0 [{'type': 'scale', 'scale': [1.0, 1.0, 1.0]}, {'type': 'translation', 'translation': [15.625, 15.625, 15.625]}]
1 [{'type': 'scale', 'scale': [2.0, 2.0, 2.0]}, {'type': 'translation', 'translation': [31.25, 31.25, 31.25]}]
```

`scale` is a multiplier on the root's `base_bin_shape`, not an absolute size:
level 1's `[2, 2, 2]` against a `base_bin_shape` of `31.25` is a bin shape of
`62.5` world units, and `translation` is half of that — the offset of the bin
centre. The supported api assembles the same number, so a tool reading through
`zarr-vectors-py` never has to multiply the two out itself:

```python
import zarr_vectors as zv

ds = zv.open("scan.zarrvectors")
print(ds.axes)
for index in ds.levels:
    print(index, ds.level(index).resolution)
```

```
({'name': 'x', 'type': 'space'}, {'name': 'y', 'type': 'space'}, {'name': 'z', 'type': 'space'})
0 (31.25, 31.25, 31.25)
1 (62.5, 62.5, 62.5)
```

Stores that declare a `crs` of `RAS` (Right-Anterior-Superior) are automatically
aligned with Neuroglancer's default RAS coordinate system.

For stores in voxel space, pass an explicit affine when loading:

```python
viewer.add(
    "scan.zarrvectors",
    transform=np.array([
        [0.004, 0,     0,     0],     # 4 µm per voxel
        [0,     0.004, 0,     0],
        [0,     0,     0.025, 0],     # 25 µm z-step
        [0,     0,     0,     1],
    ]),
)
```

---

## Quick-start checklist

Before working through the detailed tutorials:

- [ ] `pip install git+https://github.com/BRIDGE-Neuroscience/zv-ngtools.git`
- [ ] A web browser that supports WebGL 2 (Chrome, Firefox, Edge)
- [ ] A `.zarrvectors` store with at least a base level (level 0)

Optionally, for best performance:

- [ ] A multi-level pyramid (run `ds.build_pyramid(factors=[(2.0, 1.0)])` first)
- [ ] Consolidated metadata (`zarr.consolidate_metadata`) — this works on a Zarr Vectors
      store, but zarr-python warns as it writes that consolidated metadata is not
      part of the Zarr v3 specification and other implementations may ignore it
