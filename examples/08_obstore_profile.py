"""Write-section profile target for examples/08_obstore.ipynb.

Mirrors the create_store + write_points block of the notebook so we can
run cProfile on a real obstore->GCS write.  Each run writes to a fresh
gs:// subpath (timestamp-suffixed) to avoid hitting an already-populated
prefix.

Run::

    python -m cProfile -o examples/08_obstore_write.prof \
        examples/08_obstore_profile.py

Or just time it::

    python examples/08_obstore_profile.py
"""

from __future__ import annotations

import os
import time

import numpy as np

from zarr_vectors import create_store
from zarr_vectors.core.metadata import NgffAxis
from zarr_vectors.types.points import write_points


def main() -> None:
    bucket_root = os.environ.get(
        "ZV_PROFILE_STORE",
        "gs://allen_neuroglancer_ccf/zarrvectors/obstore_profile",
    )
    store = f"{bucket_root}_{int(time.time())}"
    print(f"Store path: {store}")

    # ---- Data ---------------------------------------------------------
    rng = np.random.default_rng(42)
    N = 200_000
    positions = rng.uniform(0, 1000, (N, 3)).astype(np.float32)
    intensity = rng.uniform(0, 1, N).astype(np.float32)
    label = rng.integers(0, 8, N).astype(np.int32)
    confidence = rng.uniform(0.5, 1, N).astype(np.float32)

    print(
        f"positions : {positions.shape}  dtype={positions.dtype}\n"
        f"intensity : {intensity.shape}\n"
        f"label     : {label.shape}  classes {np.unique(label)}\n"
        f"confidence: {confidence.shape}"
    )

    # ---- Write --------------------------------------------------------
    t0 = time.perf_counter()
    create_store(
        store,
        bounds=([0, 0, 0], [1000, 1000, 1000]),
        chunk_shape=(200.0, 200.0, 200.0),  # 200³ µm per chunk -> 125 chunks
        axes=[
            NgffAxis(name="x", type="space"),
            NgffAxis(name="y", type="space"),
            NgffAxis(name="z", type="space"),
        ],
        backend="obstore",
    )
    t_create = time.perf_counter() - t0

    t0 = time.perf_counter()
    write_points(
        store,
        positions,
        attributes={
            "intensity": intensity,
            "label": label,
            "confidence": confidence,
        },
        backend="obstore",
    )
    t_write = time.perf_counter() - t0

    print(
        f"\ncreate_store : {t_create * 1000:7.1f} ms\n"
        f"write_points : {t_write * 1000:7.1f} ms ({N / t_write:.0f} pts/s)\n"
        f"total        : {(t_create + t_write) * 1000:7.1f} ms\n"
    )


if __name__ == "__main__":
    main()
