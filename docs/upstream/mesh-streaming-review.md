# Raw mesh ownership reads: contribution packet

## Problem and resulting behavior

An interactive renderer needs complete triangles for a spatial ownership cell.
The current `read_mesh(chunks=...)` filters vertices and drops faces that
reference vertices in unselected chunks. It also reads the entire face-link
family, including unrelated geometry. Expanding the selected endpoint chunks
can recover the triangle but does not avoid the global connectivity read.

This branch adds `building.MeshReader`: select stored canonical face owners,
read only their connectivity, resolve referenced vertex chunks, preserve face
winding and reject missing/invalid dependencies. Existing `read_mesh` behavior
is unchanged. Bbox/vertex subsets and owner-cell subsets are different contracts.

The new reader snapshots read-only metadata for its lifetime. The level handle
is derived inside the root cache scope, offsets families are reused, and
same-level link anchoring avoids deriving two identical scales from a fresh
root lookup. Repeated queries still read geometry payloads. `close()` or the
context manager ends the metadata scope; mutate through a separate lifecycle
and construct a fresh reader afterward.

## Independent reproducer

No Lodstone, renderer, scientific download, credentials or GPU is needed:

```bash
uv run examples/mesh_read_reproducer.py --far-triangles 64 --output mesh-read-results.json
```

Or use this checkout's Python environment directly. The script creates one
triangle crossing three vertex chunks and 64 unrelated distant triangles.
It compares the actual oriented triangle coordinates, counts buffers returned
by a real Zarr Store wrapper (including batched range reads), and records
metadata discovery separately from query reads.

Recorded local result:

| Query | Returned faces | Encoded payload bytes | Metadata bytes |
| --- | ---: | ---: | ---: |
| Existing `read_mesh`, only one vertex chunk | 0 | See reproducer | See reproducer |
| Existing `read_mesh`, all three endpoint chunks | 1 | 2,320 | 21,149 |
| Owner read, first query after discovery | 1 | 272 | 1,719 |
| Owner read, repeated query | 1 | 272 | 0 |

Index discovery costs an additional **12,476 metadata bytes** on this fixture.
The two complete-query routes return exactly the same geometry here, although
their general query contracts differ. The 8.53× payload ratio applies to this
fixture, not a universal speedup. It is not a latency or HTTP wire-byte benchmark.
Store encoding/version changes can change exact sizes.

Raw results: [mesh-read-results.json](mesh-read-results.json).

## Biological integration evidence

The downstream Lodstone branch uses the public IDR0062 Nessys image 6001240,
paired labels and retained fine reference geometry. Its 81,566 fine triangles
match the streamed ownership partition exactly after declared float32 storage
conversion. Six closed nuclei retain component counts, Euler characteristic and
edge incidence after a 70% face reduction. These are geometry checks, not a new
claim about biological segmentation accuracy.

The downstream diagnostics found repeated metadata cost in the first reader
implementation. On the same local prepared store, coarse-plus-fine reads changed
from 45,285,804 to 5,787,767 returned bytes; metadata changed from 39,728,024 to
229,987 bytes. Encoded geometry payload remained **5,557,780 bytes**. This is a
local application I/O comparison, with no inference about remote wall-clock speed.

Downstream code, provenance, limits and launcher:
<https://github.com/kephale/lodstone/tree/feat/zarr-vectors-mesh-streaming>

## Scope and review questions

- Raw spatial 3-D meshes with canonical ownership only. Draco and duplicate
  ownership policies are rejected explicitly. No editing, mesh generation,
  decimator, mixed-LOD stitching or object-ID selection is added.
- Vertex dependency limits are checked before fetching vertex payloads, after
  reading owned connectivity and fragment indices. This is not a total RSS cap.
- Metadata/cell index discovery is resident and O(number of cells). Truly
  out-of-core metadata indexing remains separate work.
- The mesh format-spec page still contains historical centroid/global-index
  descriptions inconsistent with the current link implementation. This packet
  documents actual stored ownership and corrects the reader tutorial; normative
  format wording needs maintainer review rather than silently changing the format.
- Decide whether the physical-layout `building` surface is the right home for
  this additive reader. The implementation does not require a renderer dependency.

Tests cover boundary completeness/winding, disjoint ownership partitions,
unrelated-payload exclusion, missing dependencies, budget preflight, explicit
Draco rejection, metadata reuse, re-reading geometry and closed-reader behavior.
The focused mesh/link/API regression suite passes; no complete upstream suite
pass is claimed. No upstream PR or maintainer message has been sent.
