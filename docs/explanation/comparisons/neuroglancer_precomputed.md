# Zarr Vectors and Neuroglancer precomputed

How Zarr Vectors differs from the Neuroglancer precomputed formats, and
when to use each.

## What precomputed is for

[Neuroglancer precomputed](https://github.com/google/neuroglancer/tree/master/src/datasource/precomputed)
is a family of formats that the Neuroglancer viewer fetches directly
over HTTP. Each layer is a directory with an `info` JSON file. There
are formats for annotations, skeletons (`neuroglancer_skeletons`),
meshes (`neuroglancer_legacy_mesh`, `neuroglancer_multilod_draco`) and
segment properties. A pipeline writes a layer once, for viewing.

A Zarr Vectors store is a Zarr v3 hierarchy that programs read and
update through a library. One chunk layout and object model serve every
geometry type.

## Comparison

| | Zarr Vectors | Precomputed |
|---|---|---|
| Geometry | Points, lines, polylines, streamlines, graphs, skeletons, meshes | Point, line, box and ellipsoid annotations; skeletons; meshes |
| Unit of storage | One cell per spatial chunk, holding every object's vertices there | Skeletons and meshes: one object per file or shard entry; annotations: spatially chunked |
| Objects that cross chunks | Fragments per chunk, joined by the [object manifest](../../spec/object_model/object_manifest.md) and [links](../../spec/object_model/links.md) | Skeletons stored whole; legacy meshes split into fragments named by a manifest |
| Resolution levels | Coarser levels in the same store | Depends on the format |
| Attributes | Per-vertex and per-object arrays of any NumPy dtype | Skeleton vertex attributes, segment properties, annotation properties |
| Metadata | Attributes in `zarr.json` | `info` JSON per layer |
| Viewer | ngplatform, a Neuroglancer build with a `zarr-vectors` data source | Stock Neuroglancer |

## When to use which

Use Zarr Vectors as the working store: for analysis, spatial queries,
updates, and geometry types precomputed lacks, such as streamlines.

Use precomputed to show skeletons or meshes in stock Neuroglancer.
Export them with `zarr-vectors-tools`
(`zvtools convert STORE OUTPUT --format precomputed`). Skeleton and
graph stores become a `neuroglancer_skeletons` layer, mesh stores a
single-resolution `neuroglancer_legacy_mesh` layer, and object
attributes become segment properties. Other geometry types and
Draco-encoded meshes are not exported. Before exporting, run
`zarr_vectors.validate.validate(store, level=4)`.

See [View in Neuroglancer](../../how_to/view_in_neuroglancer.md) and the
zarr-vectors-tools
[Visualise](https://zarr-vectors-tools.readthedocs.io/en/latest/visualise.html)
page.
