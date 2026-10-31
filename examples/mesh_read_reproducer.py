# /// script
# requires-python = ">=3.11"
# dependencies = ["zarr-vectors @ git+https://github.com/kephale/zarr-vectors-py@c2c16dcf3e0b475369b01be4f39484427888a628"]
# ///
"""Independent boundary-face and encoded-store I/O reproducer. No Lodstone needed."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from zarr.storage import LocalStore, WrapperStore

from zarr_vectors.building import MeshReader, write_mesh
from zarr_vectors.types.meshes import read_mesh


class ReadCounter(WrapperStore):
    def __init__(self, root):
        super().__init__(LocalStore(root, read_only=True))
        self.reset()

    def reset(self):
        self.requests = self.total = self.metadata = 0
        self.payload_keys = set()

    def record(self, key, result):
        self.requests += 1
        size = len(result) if result is not None else 0
        self.total += size
        if key.rsplit("/", 1)[-1] in {"zarr.json", ".zattrs", ".zarray", ".zgroup"}:
            self.metadata += size
        elif size:
            self.payload_keys.add(key)

    async def get(self, key, prototype, byte_range=None):
        result = await self._store.get(key, prototype, byte_range)
        self.record(key, result)
        return result

    async def get_partial_values(self, prototype, key_ranges):
        requests = list(key_ranges)
        results = await self._store.get_partial_values(prototype, requests)
        for (key, _), result in zip(requests, results, strict=True):
            self.record(key, result)
        return results

    def snapshot(self):
        return {
            "store_reads": self.requests,
            "encoded_bytes": self.total,
            "metadata_bytes": self.metadata,
            "payload_bytes": self.total - self.metadata,
            "payload_keys": sorted(self.payload_keys),
        }


def run(directory: Path, far_triangles: int = 64) -> dict:
    if far_triangles < 1:
        raise ValueError("far_triangles must be positive")
    points = [[1.0, 1.0, 1.0], [11.0, 1.0, 1.0], [1.0, 11.0, 1.0]]
    faces = [[2, 1, 0]]  # preserve this winding across three vertex chunks
    for i in range(far_triangles):
        x = 101.0 + i * 20.0
        offset = len(points)
        points.extend([[x, 1.0, 1.0], [x + 1.0, 2.0, 1.0], [x + 1.0, 1.0, 2.0]])
        faces.append([offset, offset + 1, offset + 2])
    points, faces = np.asarray(points, dtype=np.float32), np.asarray(faces, dtype=np.int64)
    path = directory / "fixture.zarrvectors"
    write_mesh(path, points, faces, chunk_shape=(10.0, 10.0, 10.0))
    legacy_store = ReadCounter(path)
    legacy_boundary = read_mesh(legacy_store, chunks=[(0, 0, 0)])
    legacy_store = ReadCounter(path)
    # Include all endpoint chunks so this comparison returns the same one
    # triangle; unlike the owner API this query has vertex-region semantics.
    legacy_complete = read_mesh(legacy_store, chunks=[(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    old = legacy_store.snapshot()
    store = ReadCounter(path)
    with MeshReader(store) as reader:
        discovery = store.snapshot()
        store.reset()
        owned = reader.read([(0, 0, 0)], max_bytes=1 << 20)
        first = store.snapshot()
        store.reset()
        repeated = reader.read([(0, 0, 0)], max_bytes=1 << 20)
        second = store.snapshot()
    expected = points[faces[:1]]
    np.testing.assert_array_equal(owned["vertices"][owned["faces"]], expected)
    np.testing.assert_array_equal(repeated["vertices"][repeated["faces"]], expected)
    np.testing.assert_array_equal(legacy_complete["vertices"][legacy_complete["faces"]], expected)
    if any("/c/10/0/0" in key for key in first["payload_keys"]):
        raise AssertionError("owned query fetched the unrelated far geometry")
    return {
        "far_triangles": far_triangles,
        "legacy_one_vertex_chunk_faces": legacy_boundary["face_count"],
        "legacy_complete_endpoint_query": {"faces": legacy_complete["face_count"], **old},
        "ownership_index_discovery": discovery,
        "ownership_first_query": {"faces": owned["face_count"], **first},
        "ownership_repeated_query": {"faces": repeated["face_count"], **second},
        "payload_ratio_legacy_to_owned": old["payload_bytes"] / first["payload_bytes"],
        "scope": (
            "Local encoded store buffers, not latency or HTTP wire bytes. Discovery is separate. "
            "Queries have different contracts but return identical geometry on this fixture. "
            "No decoded mesh cache is used in the reader repeat."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--far-triangles", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="zv-mesh-read-") as directory:
        report = run(Path(directory), args.far_triangles)
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
