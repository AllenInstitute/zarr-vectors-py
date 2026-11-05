# Root metadata

## Terms

**Root attributes**
: The `"attributes"` block of the root group's `zarr.json` (Zarr v3 has no
  separate `.zattrs` file; "root `.zattrs`" in older pages means this block).
  It carries three top-level keys: `zarr_vectors` (everything this format
  defines), `multiscales` (OME-Zarr / NGFF pyramid metadata, which also holds
  the **axes**) and `ome` (an RFC 8 node, 0.9.2+).

**`zarr_vectors`**
: The JSON object holding the format's own root metadata: version, chunk
  shape, bounds, geometry types, conventions and capabilities. Parsed into
  `RootMetadata` by `zarr_vectors.core.metadata`.

**`zv_version`**
: The format version the store was written with, e.g. `"0.9.2"`. Current
  value: `zarr_vectors.constants.FORMAT_VERSION`. Readers refuse any store older
  than `0.9.0` (the single-vlen-array layout is a hard break; there is no
  compatibility shim).

**`geometry_types`**
: A **list** of geometry type constants present in the store, drawn from
  `"point_cloud"`, `"line"`, `"polyline"`, `"streamline"`, `"graph"`,
  `"skeleton"`, `"mesh"`. It starts empty and each type writer appends its own.
  (A store written by one type writer has a single entry; a composite store
  can have several.)

**`chunk_shape`**
: The spatial extent of each chunk at the root grid, a D-tuple of positive
  floats. It is the *default* for every level; a level may override it with an
  integer multiple (see [Level groups](level_groups.md)).

**`bounds`**
: `[min_corner, max_corner]` — the extent the chunk grid is allocated for.
  Mandatory. A fresh store defaults to `[0, …]` – `[128, …]`; a write outside
  it raises unless `out_of_bounds="expand"` / `"ignore"` is passed.

**`base_bin_shape`**
: The bin shape at level 0, a D-tuple of positive floats. Optional; when absent
  it defaults to `chunk_shape` (one bin per chunk). Bin shapes of coarser
  levels are derived from it and the level's scale in `multiscales`.

**Axes**
: The list of NGFF axis objects `{name, type, unit?}`. They are stored **once**,
  in `multiscales[0].axes`; there is no `spatial_dims`, `axis_units` or
  `spatial_index_dims` key under `zarr_vectors`. The number of spatial
  dimensions `D` (`sid_ndim`) is `len(axes)`.

---

## Introduction

The root attributes are the entry point for any tool reading a Zarr Vectors
store. They declare what kind of data the store contains, how it is laid out
spatially, and how its resolution levels relate to one another. A reader can
determine everything it needs to open and query the store from the root
attributes plus each level's `zarr_vectors_level` block.

This page documents every key, distinguishing required keys (whose absence
makes the store unreadable) from optional ones.

---

## Technical reference

### Complete example (streamline store, 3-D, two resolution levels)

This is the `"attributes"` block of the root `zarr.json`:

```json
{
  "zarr_vectors": {
    "zv_version":              "0.9.2",
    "chunk_shape":             [200.0, 200.0, 200.0],
    "bounds":                  [[0.0, 0.0, 0.0], [8000.0, 6000.0, 4000.0]],
    "geometry_types":          ["streamline"],
    "base_bin_shape":          [50.0, 50.0, 50.0],
    "links_convention":        "implicit_sequential",
    "object_index_convention": "standard",
    "cross_chunk_strategy":    "explicit_links",
    "cross_level_depth":       1,
    "cross_level_storage":     "explicit",
    "reduction_factor":        8,
    "format_capabilities":     ["preserved_object_ids", "shared_fragments",
                                "fragment_index", "multiscale_links"]
  },
  "multiscales": [
    {
      "version": "0.4",
      "name":    "default",
      "axes": [
        {"name": "x", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "z", "type": "space", "unit": "micrometer"}
      ],
      "datasets": [
        {
          "path": "0",
          "coordinateTransformations": [
            {"type": "scale",       "scale":       [1.0, 1.0, 1.0]},
            {"type": "translation", "translation": [25.0, 25.0, 25.0]}
          ]
        },
        {
          "path": "1",
          "coordinateTransformations": [
            {"type": "scale",       "scale":       [2.0, 2.0, 2.0]},
            {"type": "translation", "translation": [50.0, 50.0, 50.0]}
          ]
        }
      ],
      "metadata": {"format": "zarr_vectors"}
    }
  ],
  "ome": {
    "version": "0.6",
    "type":    "collection",
    "name":    "tracts",
    "attributes": {
      "scene": {
        "coordinateSystems": [{"name": "world", "axes": [
          {"name": "x", "type": "space", "unit": "micrometer"},
          {"name": "y", "type": "space", "unit": "micrometer"},
          {"name": "z", "type": "space", "unit": "micrometer"}
        ]}],
        "coordinateTransformations": []
      }
    },
    "nodes": [
      {"type": "zv:level", "name": "0"},
      {"type": "zv:level", "name": "1"}
    ]
  }
}
```

Note what is **not** here: per-level `bin_ratio`, `bin_shape` and
`object_sparsity` do not appear in `multiscales`; the level's scale and
translation encode the first two (below), and `object_sparsity` lives in the
level's own block.

### `zarr_vectors` keys

#### Required

A store whose `zarr_vectors` block lacks any of these (or whose
`multiscales[0].axes` is missing) cannot be opened by the strict parser.

| Key | Type | Description |
|-----|------|-------------|
| `zv_version` | `string` | Format version. Must be at least `"0.9.0"`. |
| `chunk_shape` | `[float, …]` | D-tuple, every element `> 0`. |
| `bounds` | `[[float, …], [float, …]]` | `[min_corner, max_corner]`, each of length D. |
| `geometry_types` | `[string, …]` | May be empty on a freshly created store (L2 emits a warning), and every entry must be a known type. |

#### Optional (defaults apply when absent)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `base_bin_shape` | `[float, …]` | `chunk_shape` | Level-0 bin shape. `chunk_shape[i] / base_bin_shape[i]` must be an integer (within `1e-9`). |
| `crs` | `object` | absent | Coordinate reference system, informational; written only when the caller supplies one. |
| `links_convention` | `string` | `"implicit_sequential"` | One of `"explicit"`, `"implicit_sequential"`, `"implicit_sequential_with_branches"`. Stamped by the first type writer. |
| `object_index_convention` | `string` | `"standard"` | `"standard"` or `"identity"` (`identity` is valid only for single-chunk stores). |
| `cross_chunk_strategy` | `string` | `"explicit_links"` | `"boundary_deduplication"`, `"explicit_links"` or `"both"`. How a writer reconciles geometry that straddles a chunk boundary; unrelated to the removed `cross_chunk_links/` path. |
| `cross_level_depth` | `integer` | `1` | Max `\|delta\|` of cross-level link arrays `build_pyramid` materialises. `0` = none, `-1` = all, must be `>= -1`. |
| `cross_level_storage` | `string` | `"explicit"` | `"none"`, `"implicit"` or `"explicit"`. |
| `reduction_factor` | `integer` | `8` | Default vertex-count fold per pyramid step; must be `>= 2`. |
| `format_capabilities` | `[string, …]` | `[]` | Capability tokens (below). |
| `attribute_specs` | `object` | absent | 0.9.1+. Declared attributes by scope: `{"vertex": {name: {dtype, unit, description}}, "object": {…}, "link": {…}}`. A declaration is not a promise the array exists. Only non-empty scopes are written. |

Capability tokens (`zarr_vectors.constants`):

| Token | Meaning |
|-------|---------|
| `preserved_object_ids` | At least one level has `preserves_object_ids=true`; dropped objects keep their id as an empty manifest. |
| `shared_fragments` | At least one level has fragments that several manifests name. |
| `fragment_index` | The store uses the `ZVFG` fragment-index encoding. |
| `multiscale_links` | The store has `links/<delta>/…` arrays with `delta != 0`. |

#### Keys that do **not** exist

Earlier drafts of this page, and some tools, refer to `zarr_vectors_version`,
`geometry_type` (singular), `spatial_dims`, `coordinate_system`, `axis_units`,
`bounding_box`, `creation_timestamp`, `source_description`, `notes`,
`custom_metadata` and `draco_compressed`. None of these is written or read by
the library. The equivalents are `zv_version`, `geometry_types`, `len(axes)`,
`crs`, the axes' own `unit` fields, `bounds`, and the
[user metadata](#application-metadata) namespace. Whether a vertex array uses
Draco is recorded per array (`vertices` attribute `encoding`).

### `multiscales`

An OME-Zarr 0.4-style block, written eagerly at store creation with level 0
and extended as levels are added. Exactly one entry is used:

| Key | Value |
|-----|-------|
| `version` | `"0.4"` |
| `name` | `"default"` |
| `axes` | the store's axes — **the only copy** |
| `datasets` | one `{path, coordinateTransformations}` per level; `path` is the level group name (`"0"`, `"1"`, …) |
| `metadata` | `{"format": "zarr_vectors"}` — the format discriminator (NGFF reserves `type` for the downsampling method) |

Each dataset's transform is a `scale` and, unless all zeros, a `translation`:

| Level | `scale` | `translation` |
|-------|---------|---------------|
| 0 | `1.0` per axis | `base_bin_shape / 2` |
| N > 0 | the level's **bin ratio** — its bin shape divided by the root bin shape (a float, may be fractional) | the level's `bin_shape / 2` |

So the level's `bin_ratio` and `bin_shape` are *derived*: `bin_ratio = scale`
and `bin_shape = 2 × translation`. They are not stored in the level's own
attributes. `scale` is a dimensionless bin ratio sitting in a slot NGFF defines
as world-units-per-index; this is why the `ome` block below declares `world`
separately instead of reusing it.

Axis rules (`validate_axes`): at least two axes, at least two of type
`space`, unique names, and NGFF type order `time → channel → custom → space`
(no constraint among the space axes; the default names are `x, y, z, w`).
`unit` is omitted rather than set to a placeholder when unknown — NGFF
requires a UDUNITS-2 name when present.

### `ome` (0.9.2+)

An RFC 8 collection node so an OME collection can name the store by path. It
is additive and derived: nothing in this package reads it, and every field
restates `zarr_vectors` or `multiscales`, which remain the source of truth.
It carries one `zv:level` leaf per level group and a `scene` declaring one
coordinate system, `world`, holding the axes (vertices are stored in world
coordinates, so `coordinateTransformations` is empty). It is kept in step
automatically whenever a level is added or removed.

A store without it is valid; L1 emits a warning, and
`zarr_vectors.building.stamp_ome_node(store)` adds one in place without
rewriting data.

### Application metadata

Free-form application metadata does not belong in `zarr_vectors`. Put it under
the reserved `user_metadata` attribute, one namespace per application:

```python
ds.metadata["myapp"]["source"] = "subject 001, tractography v3"   # Dataset.metadata
```

stored as `attributes["user_metadata"]["myapp"]["source"]`. The library never
reads or validates it.

### Validation rules

**L1** requires the `zarr_vectors` block to exist, and reports (without
failing) whether `ome` exists.

**L2** (`validate_metadata`) parses the root metadata and checks:

1. `D = len(axes) >= 1`; `len(chunk_shape) == D`; every `chunk_shape[i] > 0`.
2. `bounds` has length `D` in both corners.
3. `links_convention`, `object_index_convention` and `cross_chunk_strategy`
   are known values.
4. `geometry_types` is non-empty (a warning, not an error).
5. If `base_bin_shape` is set: length `D`, all `> 0`, and
   `chunk_shape[i] / base_bin_shape[i]` is an integer.
6. Per level: `vertex_count` is a non-negative integer, `object_sparsity` is
   in `(0, 1]`, and a per-level `chunk_shape` override is a positive integer
   multiple of the root's.

L2 does **not** check that `multiscales` has an entry per level or that bin
ratios increase across levels.

### Writing and reading root metadata from Python

`write_*` functions create and extend the metadata automatically. To inspect it:

```python
from zarr_vectors.core.store import open_store, read_root_metadata, store_info

root = open_store("scan.zarrvectors", mode="r")
meta = read_root_metadata(root)          # RootMetadata
print(meta.zv_version, meta.chunk_shape, meta.bounds, meta.sid_ndim)
print(root.attrs.to_dict()["multiscales"][0]["axes"])
```

To change a root field after creation, go through the helpers rather than the
raw attribute blob:

```python
from zarr_vectors.core.store import open_store, set_bounds, update_root_metadata

root = open_store("scan.zarrvectors", mode="r+")
set_bounds(root, ([0, 0, 0], [9000, 6000, 4000]))          # expand; shrinking needs force=True
update_root_metadata(root, add_capabilities=["shared_fragments"])
```

To regenerate the `multiscales` datasets after manually adding a resolution
level:

```python
from zarr_vectors.core.multiscale import write_multiscale_metadata

write_multiscale_metadata(root)
```
