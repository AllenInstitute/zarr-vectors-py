# View stores in Neuroglancer with ngplatform

This page shows how to open a Zarr Vectors store in
[ngplatform](https://github.com/AllenInstitute/ngplatform), what each
geometry type looks like there, and how it uses a store's levels.

ngplatform is a browser workspace around an unchanged
[Neuroglancer](https://github.com/google/neuroglancer), extended by
plugins. Its Zarr Vectors plugin, on by default, reads a store into a
Neuroglancer segmentation layer with one segment per object. Stock
Neuroglancer cannot read a store; for it, export to the precomputed format
([Precomputed export](view_in_neuroglancer.md)).

## Start ngplatform

ngplatform runs in the browser with nothing to install: open
<https://alleninstitute.github.io/ngplatform/>.

To run it on your own computer instead, build it with its Neuroglancer
branch, then start the development server:

```bash
git clone -b ngplatform-upstream https://github.com/Andrew-Keenlyside/neuroglancer ../neuroglancer
(cd ../neuroglancer && npm ci && node build_tools/postpack.ts && rm -rf lib)
git clone https://github.com/AllenInstitute/ngplatform && cd ngplatform
npm install
npm run dev        # http://localhost:5173
```

## Open a store

**A store on your computer.** Use **Open → Open from device** or
**Add → Local**, or drop the `.zarrvectors` folder on the window. It is read
in place in the browser; nothing is uploaded.

**A store at a URL.** Use **Add → URL**, or paste the source anywhere on
the window outside a text field. The source is the store's URL followed by
`/|zarr-vectors:`:

```text
https://example.org/tracts.zarrvectors/|zarr-vectors:
gs://bucket/tracts.zarrvectors/|zarr-vectors:
s3://bucket/tracts.zarrvectors/|zarr-vectors:
https://example.org/tracts.zarrvectors/|zarr-vectors:#attributes=fa,z
https://example.org/tracts.zarrvectors/|zarr-vectors:#attributes=fa,obj:kind
```

`#attributes=` chooses which vertex attributes to load; names with an `obj:`
prefix are object attribute columns, for use in shaders. `gs://` and `s3://`
buckets are read directly. A private Google Cloud bucket needs a sign-in
from the **Auth** tab.

The server must allow cross-origin requests and answer byte-range requests,
and should list directories: without a listing, edges that cross chunk
boundaries are found by probing the 26 neighbouring chunks, and faces that
cross them are missing. zarr-vectors-tools'
[Visualise](https://zarr-vectors-tools.readthedocs.io/en/latest/visualise.html)
page covers serving stores and preparing them for the viewer, and
`zvtools virtual serve` serves tractograms, meshes and point tables as
virtual stores without converting them.

A store opens with level 0's `vertices` and `vertex_fragments`. With an
object index it also gets the object subsources and segment properties
described below.

## What each geometry type looks like

| Geometry type | Drawn as | Default colour |
|---|---|---|
| `point_cloud` | Points | Per object |
| `line`, `streamline`, `polyline` | Lines | Direction (absolute unit tangent as RGB) |
| `graph` | Lines along the stored edges | Direction |
| `skeleton` | Lines, including branches | Per object |
| `mesh` | Triangles: filled and lit in 3-D, the outline where faces cross a section in 2-D | Per object |

The layer's shader reads a vertex attribute `<name>` as `prop_<name>()`, an
object attribute column `<column>` as `prop_obj_<column>()`, and the tangent
as `prop_tangent()`. Object attributes appear as segment properties in the
layer's **Segments** tab, where they can be searched (for example
`length>50`), and groups appear as tags. Selected objects are drawn at full
resolution: curves, skeletons and graphs as Neuroglancer skeletons, mesh
objects as Neuroglancer meshes.

## Levels of detail

Each view draws one level of the store's pyramid: the finest level whose
vertices, for the region in view, fit about one vertex per
`renderScale`² pixels. The coarsest level is drawn while a finer one loads.
The **Render** tab's **Resolution (slice)** and **Resolution (mesh)**
sliders set the target.

- A store with only level 0 is drawn from level 0 at every zoom, so build a
  pyramid for a large store (for example with
  `ds.build_pyramid(factors=[(2.0, 1.0)])`).
- Selected mesh objects use the store's levels as mesh levels of detail
  when each level's chunks are twice the previous level's and the level
  stores faces; otherwise they are drawn from level 0.

The store's levels and their bin shapes are in the root attributes, under
`multiscales` and `zarr_vectors`; see
[Multiscale metadata](../spec/multiscale/multiscale_metadata.md).

## Stock Neuroglancer

Stock Neuroglancer does not read Zarr Vectors stores. To view a store
there, export it to Neuroglancer's precomputed format with
`zarr-vectors-tools`:

```bash
zvtools convert STORE OUTPUT --format precomputed
```

The tools
[Visualise](https://zarr-vectors-tools.readthedocs.io/en/latest/visualise.html)
page lists which geometry types it exports.
