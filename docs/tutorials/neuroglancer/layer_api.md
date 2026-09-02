# Layer management API

This page documents the full Python API for managing layers in a
`LocalNeuroglancer` instance: programmatic rendering control, coordinate
transforms, compositing multiple Zarr Vectors layers, and Neuroglancer state
serialisation. For the interactive shell, see [Shell console](shell_console.md).

---

## Layer lifecycle

```python
from ngtools.local.viewer import LocalNeuroglancer

viewer = LocalNeuroglancer()

# Add layers
viewer.add("em_volume.zarr",     name="EM")
viewer.add("neurons.zarrvectors", name="neurons")
viewer.add("tracts.zarrvectors",  name="tracts")
viewer.add("vessels.zarrvectors", name="vessels")

# Inspect
print(viewer.list_layers())
# ['EM', 'neurons', 'tracts', 'vessels']

print(viewer.layer_info("neurons"))
# {'name': 'neurons', 'type': 'skeleton',
#  'source': 'zarr_vectors:///path/to/neurons.zarrvectors',
#  'visible': True, 'level': 'auto'}

# Rename
viewer.rename_layer("neurons", "pyramidal_cells")

# Hide / show
viewer.set_visible("tracts", False)
viewer.set_visible("tracts", True)

# Remove
viewer.remove_layer("vessels")
```

---

## Colour and shader control

### Built-in colour presets

```python
# Single solid colour (hex, name, or RGB tuple)
viewer.set_color("neurons", "#22d97a")
viewer.set_color("tracts",  "blue")
viewer.set_color("scan",    (1.0, 0.45, 0.0))   # RGB float

# Named colour ramps (maps attribute value to colour)
viewer.set_shader("scan",   "viridis")
viewer.set_shader("tracts", "jet")
viewer.set_shader("scan",   "blackwhite")
viewer.set_shader("scan",   "blackred")
```

### Attribute-driven colour

Colour a layer by a named per-vertex or per-object attribute. The
attribute values are normalised to the ramp range:

```python
# Colour point cloud by intensity, viridis ramp, auto range
viewer.set_attribute_shader("scan", attribute="intensity", ramp="viridis")

# Explicit data range (clamp values outside [0.2, 0.8] to ramp endpoints)
viewer.set_attribute_shader("tracts", attribute="fa",
                             ramp="jet", vmin=0.2, vmax=0.8)

# Per-object attribute (streamlines coloured by mean FA)
viewer.set_attribute_shader("tracts", attribute="mean_fa",
                             ramp="viridis", per_object=True)
```

### Custom GLSL shaders

For full rendering control, write a Neuroglancer GLSL shader string.
Zarr Vectors attribute names are accessible as `prop_<name>()` in the shader:

```python
# Streamlines: colour by FA, desaturate low-FA streamlines
viewer.set_shader("tracts", """
void main() {
    float fa = prop_fa();          // per-vertex FA value
    float a  = smoothstep(0.15, 0.5, fa);
    emitRGB(mix(vec3(0.4), colormapViridis(fa), a));
}
""")

# Point cloud: encode intensity as brightness, label as hue
viewer.set_shader("scan", """
void main() {
    float intensity = prop_intensity();
    int   label     = int(prop_label());
    vec3  hue = hsvToRgb(vec3(float(label) / 16.0, 0.9, intensity));
    emitRGB(hue);
}
""")

# Skeleton: compartment type colours
viewer.set_shader("neurons", """
void main() {
    int t = int(prop_swc_type());
    if      (t == 1) emitRGB(vec3(1.0, 0.9, 0.0));   // soma: yellow
    else if (t == 2) emitRGB(vec3(0.2, 0.7, 1.0));   // axon: blue
    else if (t == 3) emitRGB(vec3(1.0, 0.5, 0.1));   // basal: orange
    else if (t == 4) emitRGB(vec3(0.2, 1.0, 0.4));   // apical: green
    else             emitRGB(vec3(0.7, 0.7, 0.7));   // other: grey
}
""")

# Mesh: cortical thickness overlay
viewer.set_shader("brain_surface", """
void main() {
    float t = prop_thickness();
    emitRGB(colormapViridis(clamp((t - 1.5) / 3.0, 0.0, 1.0)));
    emitTransparency(0.3);
}
""")
```

---

## Opacity and visibility

```python
viewer.set_opacity("tracts",        0.6)   # 60% opacity
viewer.set_opacity("brain_surface", 0.25)  # transparent overlay
viewer.set_opacity("neurons",       1.0)   # fully opaque (default)

# Line width for streamlines and graphs (screen pixels)
viewer.set_line_width("tracts",  2.0)
viewer.set_line_width("neurons", 1.5)

# Point size for point clouds (screen pixels)
viewer.set_point_size("scan", 3)
```

---

## Layer ordering

The z-order determines which layers are rendered on top of which in the
3-D perspective view. Layers earlier in the list are rendered first (back):

```python
# Render order: EM volume (back) → brain surface → tracts → neurons (front)
viewer.set_layer_order(["EM", "brain_surface", "tracts", "neurons"])
```

---

## Coordinate transforms

Each layer can have an independent coordinate transform that maps its
own coordinate space to the viewer's global physical space.

### Setting a transform

```python
import numpy as np

# 4×4 homogeneous matrix: RAS rotation + scale + translation
transform = np.eye(4, dtype=float)
transform[:3, :3] = 0.004 * np.eye(3)   # 4 µm/voxel isotropic
transform[:3, 3]  = [-500., -400., -250.]  # origin offset in µm

viewer.set_transform("neurons", transform)
```

### Applying a transform from a file

```python
# LTA (FreeSurfer linear transform)
viewer.apply_transform_file("neurons", "/path/to/transform.lta")

# ITK affine
viewer.apply_transform_file("neurons", "/path/to/affine.mat")
```

### Resetting a transform

```python
viewer.reset_transform("neurons")   # removes any applied transform
```

---

## Level-of-detail control

```python
# Automatic LOD (default) — switches level based on viewport zoom
viewer.set_level("tracts", "auto")

# Force a specific level
viewer.set_level("tracts", 2)       # always serve level 2
viewer.set_level("scan",   0)       # always finest resolution

# Query which level is currently being served
info = viewer.layer_info("tracts")
print(info["current_level"])        # e.g. 1 (auto-selected)
```

---

## Compositing multiple Zarr Vectors layers

A common pattern in connectomics is to layer point clouds, skeletons,
tractography, and meshes over an image volume in a single viewer state.
Here is a complete compositing example:

```python
from ngtools.local.viewer import LocalNeuroglancer
import numpy as np

viewer = LocalNeuroglancer()

# 1. Background image volume (OME-Zarr, served natively)
viewer.add("zarr:///data/em_volume.zarr", name="EM")
viewer.set_shader("EM", "blackwhite")
viewer.set_opacity("EM", 1.0)

# 2. Brain surface mesh (transparent overlay)
viewer.add("brain_surface.zarrvectors", name="surface")
viewer.set_color("surface", "#b0b0b0")
viewer.set_opacity("surface", 0.15)

# 3. Tractography streamlines (coloured by FA)
viewer.add("tracts.zarrvectors", name="tracts")
viewer.set_attribute_shader("tracts", attribute="fa",
                             ramp="viridis", vmin=0.2, vmax=0.8)
viewer.set_line_width("tracts", 1.5)
viewer.set_opacity("tracts", 0.8)

# 4. Neuronal skeletons (compartment colours)
viewer.add("neurons.zarrvectors", name="neurons")
viewer.set_shader("neurons", """
void main() {
    int t = int(prop_swc_type());
    if (t == 2) emitRGB(vec3(0.2, 0.7, 1.0));
    else        emitRGB(vec3(1.0, 0.5, 0.1));
}
""")

# 5. Synapse point cloud (coloured by synapse type)
viewer.add("synapses.zarrvectors", name="synapses")
viewer.set_attribute_shader("synapses", attribute="synapse_type",
                             ramp="tab10", per_object=False)
viewer.set_point_size("synapses", 4)

# Layer ordering: EM at back, neurons in front
viewer.set_layer_order(["EM", "surface", "tracts", "synapses", "neurons"])

# Navigate to region of interest
viewer.move_to([1200., 800., 400.])
viewer.zoom(2.0)
viewer.set_layout("xy-3d")
```

---

## State serialisation

### Save and restore a session

```python
import json

# Capture full viewer state (all layers, positions, shaders)
state = viewer.get_state()

with open("session_subject_001.json", "w") as f:
    json.dump(state, f, indent=2)

# Later — restore exactly
with open("session_subject_001.json") as f:
    state = json.load(f)
viewer.set_state(state)
```

### Sharing with collaborators

If all layers are loaded from publicly accessible URLs, the full
Neuroglancer state can be embedded in a URL and opened by anyone with a
browser — no Python required:

```python
# Generate shareable URL (uses public store URLs)
url = viewer.get_url(use_public_urls=True)
print(url)
# https://neuroglancer-demo.appspot.com/#!{...json-encoded state...}
```

For Zarr Vectors layers, the URL embeds `zarr_vectors://s3://...` source URLs.
The recipient needs `zv-ngtools` installed and running locally to open
Zarr Vectors layers from such a link.

### Partial state: save only layer settings

To save rendering settings (shaders, opacity, transforms) without locking
in the camera position:

```python
state = viewer.get_state()
layer_state = {
    "layers": state["layers"],
    # omit "position", "crossSectionOrientation", "projectionOrientation"
}
with open("layer_settings.json", "w") as f:
    json.dump(layer_state, f, indent=2)
```

---

## Programmatic navigation

```python
# Move cursor (units = physical coordinate units of the global space)
viewer.move_to([500.0, 300.0, 200.0])

# Zoom
viewer.zoom(4.0)          # 4 units per screen pixel
viewer.zoom("fit")        # fit all layers in viewport

# Layout
viewer.set_layout("xy")         # single cross-section
viewer.set_layout("xy-3d")      # cross-section + 3-D
viewer.set_layout("4panel")     # xy, xz, yz, 3-D
viewer.set_layout("3d")         # 3-D perspective only

# Cross-section orientation
viewer.set_orientation("coronal")
viewer.set_orientation("sagittal")
viewer.set_orientation("axial")

# Screenshot
viewer.screenshot("figure.png", size=(1920, 1080))
```
