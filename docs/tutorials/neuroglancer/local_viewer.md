# LocalNeuroglancer — Python API

`LocalNeuroglancer` runs a local Neuroglancer instance that can load
`.zarrvectors` stores directly from disk or remote object stores. It
manages both a local file server (for serving local data) and a
Neuroglancer viewer tab in your browser. This tutorial covers the Python
API for loading stores, configuring layer rendering, and managing the
viewer state programmatically.

For the interactive shell console, see [Shell console](shell_console.md).

---

## Launching the viewer

```python
from ngtools.local.viewer import LocalNeuroglancer

viewer = LocalNeuroglancer()
# Output:
# fileserver:   http://127.0.0.1:9123/
# neuroglancer: http://127.0.0.1:9321/v/1/
```

A Neuroglancer tab opens in your default browser automatically. If it
does not open (e.g. on a remote server), navigate to the printed URL
manually.

The viewer runs in the background as a Tornado server. Your Python session
remains interactive.

### Controlling the port

```python
viewer = LocalNeuroglancer(
    fileserver_port=9123,       # local file server port
    neuroglancer_port=9321,     # Neuroglancer viewer port
    open_browser=True,          # set False to suppress auto-open
)
```

### Using as a context manager

```python
with LocalNeuroglancer() as viewer:
    viewer.add("scan.zarrvectors")
    input("Press Enter to close…")
# Viewer and server are shut down on exit
```

---

## Loading a ZVF store

### From local disk

```python
# Infer type from .zarrvectors extension
viewer.add("scan.zarrvectors")

# Explicit layer name
viewer.add("scan.zarrvectors", name="synchrotron_scan")

# Explicit scheme
viewer.add("zarr_vectors:///path/to/scan.zarrvectors")
```

`add()` reads the store's root `zarr.json`, determines the geometry type,
and registers the appropriate Neuroglancer layer type. A ZVF store is a Zarr
v3 group, so `zarr.json` is its root document — there is no `.zattrs`, which
is the Zarr v2 spelling. The geometry type is
`attributes.zarr_vectors.geometry_types`, and the layer transform comes from
`attributes.multiscales`; see
[Coordinate system alignment](overview.md#coordinate-system-alignment) for the
full shape of that document. The store is not read in full at this point —
data is fetched on demand as the viewer pans and zooms.

### From S3

```python
viewer.add("zarr_vectors://s3://open-neuro/datasets/scan.zarrvectors")
```

For private S3 buckets, credentials must be available via the standard
AWS credential chain (`~/.aws/credentials`, environment variables, or IAM
role). `zv-ngtools` uses `s3fs` under the hood.

### From GCS

```python
viewer.add("zarr_vectors://gs://my-bucket/tracts.zarrvectors")
```

### Loading multiple layers

```python
# Load a registered OME-Zarr image volume (upstream ngtools feature)
viewer.add("zarr:///path/to/em_volume.zarr", name="EM")

# Overlay ZVF layers on the same coordinate space
viewer.add("neurons.zarrvectors", name="neurons")
viewer.add("tracts.zarrvectors",  name="tracts")
viewer.add("vessels.zarrvectors", name="vessels")
```

All layers must share the same physical coordinate system. If your ZVF
store is in voxel space and the image is in RAS mm, supply a transform
(see [Coordinate transforms](#coordinate-transforms) below).

---

## Layer listing

```python
layers = viewer.list_layers()
print(layers)
# ['EM', 'neurons', 'tracts', 'vessels']
```

---

## Configuring layer rendering

### Point cloud rendering

```python
viewer.add("scan.zarrvectors", name="scan")

# Point size in screen pixels
viewer.set_point_size("scan", size=3)

# Colour by attribute value (maps float32 to a colour ramp)
viewer.set_shader("scan", """
void main() {
    float val = prop_intensity();
    emitRGB(vec3(val, 0.2, 1.0 - val));
}
""")

# Fixed colour
viewer.set_shader("scan", color="#ff2d9a")
```

### Streamline rendering

```python
viewer.add("tracts.zarrvectors", name="tracts")

# Line width in screen pixels
viewer.set_line_width("tracts", width=1.5)

# Colour by per-vertex FA (custom GLSL shader)
viewer.set_shader("tracts", """
void main() {
    float fa = prop_fa();
    emitRGB(colormapJet(fa));
}
""")

# Solid colour
viewer.set_shader("tracts", color="#4da6ff")

# Opacity
viewer.set_opacity("tracts", 0.7)
```

### Skeleton rendering

```python
viewer.add("neurons.zarrvectors", name="neurons")

# Colour axons and dendrites differently
viewer.set_shader("neurons", """
void main() {
    int t = prop_swc_type();
    if (t == 2) emitRGB(vec3(0.2, 0.8, 1.0));       // axon: blue
    else if (t == 3) emitRGB(vec3(1.0, 0.5, 0.2));  // basal: orange
    else if (t == 4) emitRGB(vec3(0.2, 1.0, 0.5));  // apical: green
    else emitRGB(vec3(0.8, 0.8, 0.8));              // other: grey
}
""")
```

### Mesh rendering

```python
viewer.add("brain.zarrvectors", name="brain_surface")

# Transparency
viewer.set_opacity("brain_surface", 0.4)

# Colour by attribute (e.g. cortical thickness)
viewer.set_shader("brain_surface", """
void main() {
    float t = prop_thickness();
    emitRGB(colormapViridis(clamp((t - 1.5) / 3.0, 0.0, 1.0)));
}
""")
```

---

## LOD and resolution level control

### Automatic LOD (default)

By default, `zv-ngtools` selects the resolution level based on the
Neuroglancer viewport's screen resolution at the current zoom. As the
user zooms out, coarser levels are served automatically.

The decision needs one number per level: that level's bin shape, the finest
region a spatial read can isolate. The supported api exposes it as
`Level.resolution`, so the selection is a plain scan over `Dataset.levels`:

```python
import zarr_vectors as zv

ds = zv.open("tracts.zarrvectors")

def select_level(ds, requested_resolution_um):
    for index in reversed(ds.levels):      # coarsest first
        if max(ds.level(index).resolution) <= requested_resolution_um:
            return index
    return ds.levels[0]                    # finest level if nothing coarser qualifies
```

`ds.levels` is ascending with the finest level first, so `ds.levels[0]` is the
fall-through when no coarser level is good enough.

### Force a specific level

To lock the viewer to a specific resolution level (useful for debugging
or when the automatic LOD selection is not appropriate):

```python
viewer.set_level("tracts", level=2)   # always serve level 2
viewer.set_level("tracts", level=None) # restore automatic LOD
```

---

## Coordinate transforms

If a ZVF store and an image volume are in different coordinate spaces,
apply a transform when loading:

```python
import numpy as np

# 4×4 affine (homogeneous): scales from voxel to µm and offsets origin
voxel_to_um = np.array([
    [4.0,  0,    0,    -500.0],   # x: 4 µm/vx, offset -500 µm
    [0,    4.0,  0,    -500.0],   # y
    [0,    0,    25.0, -250.0],   # z: 25 µm/vx
    [0,    0,    0,     1.0  ],
])

viewer.add(
    "scan.zarrvectors",
    name="scan",
    transform=voxel_to_um,
)
```

The transform is stored in the Neuroglancer layer state and applied when
rendering. It does not modify the ZVF store.

---

## Navigating the viewer

```python
# Move cursor to a specific physical position (in store coordinate units)
viewer.move_to([500.0, 300.0, 200.0])

# Set zoom level (voxels per screen pixel)
viewer.zoom(4.0)

# Set layout
viewer.set_layout("xy")          # single cross-section view
viewer.set_layout("xy-3d")       # cross-section + 3-D perspective
viewer.set_layout("4panel")      # xy, xz, yz, and 3-D views
```

---

## Hiding and removing layers

```python
# Hide a layer (keeps it loaded, stops rendering)
viewer.set_visible("tracts", visible=False)
viewer.set_visible("tracts", visible=True)

# Remove a layer entirely
viewer.remove("tracts")
```

---

## Getting and setting viewer state

The Neuroglancer state is a JSON object that encodes all layers, positions,
zoom, and rendering settings. It can be serialised and shared:

```python
# Get current state as a Python dict
state = viewer.get_state()

# Save to file
import json
with open("session_state.json", "w") as f:
    json.dump(state, f, indent=2)

# Restore from file
with open("session_state.json") as f:
    saved_state = json.load(f)
viewer.set_state(saved_state)
```

### Sharing a Neuroglancer link

If the ZVF stores are accessible via a public URL (S3 with public read,
or a running file server), the Neuroglancer state URL can be shared:

```python
# Get the shareable Neuroglancer URL
url = viewer.get_url()
print(url)
# http://127.0.0.1:9321/v/1/#!{"layers":[...], "position":[...], ...}
```

For collaborators without the local server, export the link using the
public URLs of remote stores:

```python
url = viewer.get_url(use_public_urls=True)
# https://neuroglancer-demo.appspot.com/#!{"layers":[{"source":"zarr_vectors://s3://..."}]}
```

Note: links with `zarr_vectors://` sources require the recipient to have
`zv-ngtools` installed and running. For static sharing, use the precomputed
export workflow instead.

---

## Programmatic screenshot

```python
# Save the current viewport as a PNG
viewer.screenshot("screenshot.png")

# Specify resolution
viewer.screenshot("hires.png", size=(3840, 2160))
```

---

## Common patterns

### Quick data inspection

```python
from ngtools.local.viewer import LocalNeuroglancer

viewer = LocalNeuroglancer()
viewer.add("scan.zarrvectors")
# Browse the data manually, then close the terminal to stop the server
```

### Scripted figure generation

```python
from ngtools.local.viewer import LocalNeuroglancer
import time

viewer = LocalNeuroglancer(open_browser=False)

viewer.add("em_volume.zarr", name="EM")
viewer.add("neurons.zarrvectors", name="neurons")
viewer.set_shader("neurons", color="#22d97a")
viewer.set_layout("3d")
viewer.move_to([1000., 800., 400.])
viewer.zoom(2.0)

time.sleep(1)   # allow rendering to complete
viewer.screenshot("figure_3a.png", size=(1920, 1080))
viewer.close()
```
