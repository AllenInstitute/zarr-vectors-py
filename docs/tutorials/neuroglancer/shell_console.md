# Shell console (`nglocal`)

The `nglocal` shell is an interactive command-line interface that
controls a Neuroglancer instance. It features tab completion for all
commands and file paths, a persistent command history, and inline help.
It is the fastest way to explore data interactively without writing
Python scripts.

---

## Starting the console

```bash
nglocal
```

Output:

```
             _              _
 _ __   __ _| |_ ___   ___ | |___
| '_ \ / _` | __/ _ \ / _ \| / __|
| | | | (_| | || (_) | (_) | \__ \
|_| |_|\__, |\__\___/ \___/|_|___/
       |___/

fileserver:   http://127.0.0.1:9123/
neuroglancer: http://127.0.0.1:9321/v/1/

Type help to list available commands, or help <command> for specific help.
Type Ctrl+C to interrupt the current command and Ctrl+D to exit the app.
[1]
```

A Neuroglancer browser tab opens automatically. The prompt `[N]` shows
the command number. Ctrl+C cancels the current command without exiting;
Ctrl+D (or `exit`) closes the viewer and exits the shell.

### Headless (no auto-open)

```bash
nglocal --no-browser
# fileserver:   http://127.0.0.1:9123/
# neuroglancer: http://127.0.0.1:9321/v/1/
# Navigate to the URL manually.
```

### Custom ports

```bash
nglocal --fileserver-port 9200 --neuroglancer-port 9400
```

---

## Loading Zarr Vectors stores

### By file path

```bash
[1] load scan.zarrvectors
# Loaded: scan → point_cloud layer
# http://127.0.0.1:9321/v/1/

[2] load tracts.zarrvectors --name white_matter
# Loaded: white_matter → streamline layer
```

Paths ending in `.zarrvectors` are automatically recognised as Zarr Vectors stores
and the `zarr_vectors://` scheme is inferred. Absolute paths:

```bash
[3] load /data/projects/bridge/neurons.zarrvectors
```

### By URL (remote stores)

```bash
[4] load zarr_vectors://s3://open-neuro/datasets/scan.zarrvectors
[5] load zarr_vectors://gs://my-bucket/tracts.zarrvectors --name remote_tracts
```

### Loading other formats (upstream ngtools)

`zv-ngtools` inherits the upstream `ngtools` format support:

```bash
[6] load /path/to/em_volume.nii.gz        # NIfTI image
[7] load zarr:///path/to/image.zarr        # OME-Zarr image volume
[8] load tiff:///path/to/image.tiff        # TIFF
[9] load /path/to/tracts.trk               # TRK tractography
```

### Tab completion for file paths

Tab completion works for local file paths and for Zarr Vectors stores: pressing
Tab after a partial path expands it, showing only `.zarrvectors` directories
and common image formats.

```bash
[1] load /data/p<Tab>
# /data/patient001/    /data/patient002/    /data/processed/
[1] load /data/processed/<Tab>
# neurons.zarrvectors  scan.zarrvectors  tracts.zarrvectors
```

---

## Listing and navigating

### List loaded layers

```bash
[10] ls
# LAYERS:
#   1. scan          (point_cloud)
#   2. white_matter  (streamline)
#   3. neurons       (skeleton)

# Long form — includes source paths and rendering settings
[11] ll
# 1. scan
#    source:  zarr_vectors:///data/scan.zarrvectors
#    type:    point_cloud
#    visible: true
#    level:   auto
```

### Change working directory

The shell maintains a working directory so you can load files with relative
paths:

```bash
[12] pwd
# /home/user

[13] cd /data/projects/bridge
[14] pwd
# /data/projects/bridge

[15] load scan.zarrvectors   # loads from /data/projects/bridge/
```

### List directory contents

```bash
[16] ls /data/projects/bridge/
# neurons.zarrvectors  scan.zarrvectors  tracts.zarrvectors  em.zarr
```

---

## Removing layers

```bash
[17] unload white_matter
# Removed layer: white_matter
```

---

## Renaming layers

```bash
[18] rename scan synchrotron_scan
# Renamed: scan → synchrotron_scan
```

---

## Applying coordinate transforms

The `transform` command applies an affine transform to a layer, mapping it
into the viewer's coordinate space.

### From an LTA or ITK file

```bash
[19] transform /path/to/affine.lta --layer neurons
[20] transform /path/to/transform.mat --layer neurons  # ITK format
```

### Inline scaling

```bash
# Scale layer by factor (useful when coordinates are in voxels and
# the image is in mm — multiply by voxel size to align)
[21] transform --scale 0.004 --layer scan         # 4 µm/voxel
[22] transform --scale 0.004 0.004 0.025 --layer scan  # anisotropic
```

### Inline translation

```bash
[23] transform --translation 100 200 -50 --layer scan   # offset in µm
```

---

## Shaders and rendering

The `shader` command applies a built-in colour preset to a layer.

### Built-in presets

```bash
# Named presets (colour ramps)
[24] shader blackred    --layer scan
[25] shader blackgreen  --layer tracts
[26] shader blackblue   --layer neurons
[27] shader viridis     --layer scan
[28] shader jet         --layer scan

# Single colour by hex
[29] shader "#ff2d9a"   --layer tracts
[30] shader "#22d97a"   --layer neurons
```

### Attribute-driven colour

For point clouds and streamlines with per-vertex attributes:

```bash
# Colour by the "intensity" attribute using the viridis ramp
[31] shader attribute:intensity:viridis --layer scan

# Colour by FA with a custom range
[32] shader attribute:fa:jet:0.2:0.8   --layer tracts
#                           ^ramp ^min  ^max
```

### Custom GLSL

For full control, pass a GLSL shader file:

```bash
[33] shader /path/to/custom.glsl --layer tracts
```

---

## Visibility and opacity

```bash
# Toggle visibility
[34] shader --opacity 0.5 --layer tracts
[35] shader --opacity 1.0 --layer tracts

# Reorder layers (z-order in 3-D view)
[36] zorder neurons tracts scan   # back-to-front: neurons drawn first
```

---

## Navigation commands

```bash
# Move cursor to physical coordinates (in store coordinate units)
[37] move 500 300 200
[38] position 500 300 200   # alias

# Zoom
[39] zoom 4        # 4 units per screen pixel
[40] unzoom        # reset to default

# Set cross-section orientation
[41] space coronal
[42] space sagittal
[43] space axial

# Set display dimensions
[44] display xyz   # show x,y,z in cross-sections
[45] layout 4panel  # xy, xz, yz + 3-D view
[46] layout 3d      # 3-D perspective only
[47] layout xy-3d   # cross-section + 3-D
```

---

## Level-of-detail control

```bash
# Force a specific resolution level for a layer
[48] level --layer tracts 2       # always show level 2
[49] level --layer tracts auto    # restore automatic LOD
[50] level --layer tracts 0       # force finest level
```

---

## State management

```bash
# Print current viewer state as JSON
[51] state

# Save state to a file
[52] state --save session.json

# Restore from a file
[53] state --load session.json

# Get the shareable Neuroglancer URL
[54] state --url
# http://127.0.0.1:9321/v/1/#!{"layers":[...]}
```

---

## Getting help

```bash
# List all commands
[55] help

# Help for a specific command
[56] help load
[57] help shader
[58] help transform
```

Each command's `--help` output documents every flag and provides an
example.

---

## Command history

The shell maintains a persistent command history in `~/.ngtools_history`.
Use the Up/Down arrow keys to navigate previous commands. Ctrl+R searches
the history interactively (same as bash reverse-i-search).

---

## Typical interactive session

```bash
$ nglocal
[1] cd /data/bridge/subject_001

[2] load em_volume.zarr --name EM
[3] load neurons.zarrvectors --name neurons
[4] load tracts.zarrvectors  --name tracts
[5] load vessels.zarrvectors --name vessels

[6] shader blackwhite --layer EM
[7] shader "#22d97a"  --layer neurons
[8] shader attribute:fa:viridis --layer tracts
[9] shader "#4da6ff"  --layer vessels

[10] layout xy-3d
[11] move 1200 800 400
[12] zoom 3

[13] shader --opacity 0.3 --layer EM
[14] level --layer tracts 1    # use coarser level for overview

[15] state --save session_subject_001.json
[16] state --url
# http://127.0.0.1:9321/v/1/#!{…}

[17] exit
```

---

## Scripting `nglocal` non-interactively

For reproducible figure generation, pipe commands to `nglocal`:

```bash
nglocal --no-browser << 'EOF'
cd /data/bridge/subject_001
load em_volume.zarr
load neurons.zarrvectors
shader "#22d97a" --layer neurons
layout 3d
move 1200 800 400
zoom 3
screenshot figure_3a.png --size 1920 1080
exit
EOF
```

Or pass a script file:

```bash
nglocal --script setup_viewer.ngl
```

A `.ngl` script is a plain text file with one `nglocal` command per line
(comments starting with `#` are ignored).
