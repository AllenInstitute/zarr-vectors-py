> [!NOTE]
> This package is under active development.

<img src="assets/zarr-vectors.png" alt="zarr-vectors" width="60%" />

**Tools for Zarr Vectors Data**

`zarr-vectors-py` reads, writes and manages large vector geometry in the Zarr Vectors format: a chunked, cloud-ready format built on Zarr v3 for multiscale points, lines, streamlines, graphs, skeletons and meshes. It adds separate chunk and bin sizes, per-level object sparsity and OME-Zarr multiscale metadata.

The package implements the [Zarr Vectors specification](https://github.com/AllenInstitute/zarr_vectors) by Forrest Collman, Allen Institute for Brain Science.

Documentation: https://zarr-vectors-py.readthedocs.io/en/latest

---

## Install

```bash
pip install zarr-vectors
```

Requires Python 3.11 or later. Optional extras add cloud backends, Draco mesh compression and GPU arrays; see [Installation](https://zarr-vectors-py.readthedocs.io/en/latest/getting_started/installation.html).

---

## Quick start

```python
import numpy as np
import zarr_vectors as zv

# A Schema says what the data is; a Layout says how it is cut up.
schema = zv.Schema(
    bounds=([0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]),
    kind="point_cloud",
    layout=zv.Layout(cells=8),  # eight chunks per axis
)

ds = zv.create("demo.zarrvectors", schema=schema)

rng = np.random.default_rng(0)
positions = rng.uniform(0, 1000, size=(10_000, 3)).astype("float32")
ds.add_points(
    positions,
    attributes={"intensity": rng.random(10_000).astype("float32")},
    object_ids=np.repeat(np.arange(100), 100),
)
ds.build_pyramid(factors=[(2.0, 1.0)])  # add a coarser level

# Read back only what falls inside a box.
result = zv.open("demo.zarrvectors").select(bbox=([0, 0, 0], [250, 250, 250])).read()

print(result.kind, result.positions.shape, result.attributes.names())
print(result.attributes["intensity"][:3])
```

```
point_cloud (122, 3) ('intensity',)
[0.58416754 0.17340323 0.28906345]
```

Every read returns a `ReadResult`, whatever the geometry: `positions` is `(N, D)`, `parts` slices it into polylines or segments, and `edges`, `faces` and `attributes` are filled in when the store has them. A whole-store read of a point cloud leaves per-vertex `object_ids` as `None` even when ids were written; use `level.objects[...]` or `select(objects=...)` to read by id.

The example uses `zarr_vectors.api`, re-exported from the top-level package, which is for reading and writing data. `zarr_vectors.building` is for tools that construct stores. These two, with `constants`, `exceptions`, `typing` and `headers`, are the supported modules; everything else may change between releases. `zv.stability("zarr_vectors.<module>")` reports a module's tier.

The [Quickstart](https://zarr-vectors-py.readthedocs.io/en/latest/getting_started/quickstart.html) walks through writing, querying and pyramids step by step.

---

## Store layout

A store is a Zarr v3 group, by convention named with a `.zarrvectors` extension. Its root `zarr.json` holds the format's metadata under `attributes.zarr_vectors` and the resolution pyramid under `attributes.multiscales`. Each resolution level is a numbered subgroup (`0/`, `1/`, …) holding `vertices/`, `vertex_fragments/` and, depending on the geometry and on what was written, links, attributes, an object index and groups. [Core concepts](https://zarr-vectors-py.readthedocs.io/en/latest/getting_started/concepts.html) explains the tree, and [Directory structure](https://zarr-vectors-py.readthedocs.io/en/latest/spec/layout/directory_structure.html) gives it in full for each geometry.
