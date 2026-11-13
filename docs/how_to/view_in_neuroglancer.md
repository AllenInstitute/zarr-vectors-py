# View stores in Neuroglancer with ngplatform

This page shows how to open a Zarr Vectors store in
[ngplatform](https://github.com/AllenInstitute/ngplatform), a browser
workspace around [Neuroglancer](https://github.com/google/neuroglancer). Its
Zarr Vectors plugin, on by default, reads a store into a segmentation layer
with one segment per object.

## Start ngplatform

Open <https://alleninstitute.github.io/ngplatform/>. Nothing needs installing.

To run it on your own computer, build its Neuroglancer branch, then start the
development server:

```bash
git clone -b ngplatform-upstream https://github.com/Andrew-Keenlyside/neuroglancer ../neuroglancer
(cd ../neuroglancer && npm ci && node build_tools/postpack.ts && rm -rf lib)
git clone https://github.com/AllenInstitute/ngplatform && cd ngplatform
npm install
npm run dev        # http://localhost:5173
```

## Open a store

**On your computer.** Use **Open → Open from device** or **Add → Local**, or
drop the `.zarrvectors` folder on the window. The browser reads it in place;
nothing is uploaded.

**At a URL.** Use **Add → URL**, or paste the source anywhere outside a text
field. The source is the store's URL followed by `/|zarr-vectors:`:

```text
https://example.org/tracts.zarrvectors/|zarr-vectors:
gs://bucket/tracts.zarrvectors/|zarr-vectors:
s3://bucket/tracts.zarrvectors/|zarr-vectors:#attributes=fa,obj:kind
```

`#attributes=` chooses which vertex attributes to load; an `obj:` prefix
names an object attribute column. A private Google Cloud bucket needs a
sign-in from the **Auth** tab.

An HTTP server must allow cross-origin and byte-range requests. It should
also list directories: without a listing, faces that cross chunk boundaries
are missing. The zarr-vectors-tools
[Visualise](https://zarr-vectors-tools.readthedocs.io/en/latest/visualise.html)
page covers serving stores.

## What you see

| Geometry type | Drawn as | Default colour |
|---|---|---|
| `point_cloud` | Points | Per object |
| `line`, `streamline`, `polyline` | Lines | Direction (absolute unit tangent as RGB) |
| `graph` | Lines along the stored edges | Direction |
| `skeleton` | Lines, including branches | Per object |
| `mesh` | Triangles in 3-D; the outline where faces cross a section in 2-D | Per object |

In the layer's shader, a vertex attribute `<name>` is `prop_<name>()`, an
object attribute column `<column>` is `prop_obj_<column>()`, and the tangent
is `prop_tangent()`. Object attributes are searchable in the **Segments** tab
(for example `length>50`). Selected objects are drawn at full resolution.

## Levels of detail

Each view draws the finest pyramid level that fits about one vertex per
`renderScale`² pixels, and the coarsest level while it loads. The **Render**
tab's **Resolution** sliders set the target. A store with only level 0 is
drawn from level 0 at every zoom, so build a pyramid for a large store, for
example with `ds.build_pyramid(factors=[(2.0, 1.0)])`.

## Stock Neuroglancer

Stock Neuroglancer cannot read a store. Export it to the precomputed format
with zarr-vectors-tools:

```bash
zvtools convert STORE OUTPUT --format precomputed
```

The tools
[Visualise](https://zarr-vectors-tools.readthedocs.io/en/latest/visualise.html)
page lists which geometry types it exports.
