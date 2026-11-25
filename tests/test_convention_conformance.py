"""Every node the writers produce conforms to the Zarr Vectors convention.

``schema/zarr_vectors.schema.json`` is the convention's JSON schema
(https://github.com/Andrew-Keenlyside/zarr-vectors, tag v0.1.0),
generated from the copied LinkML model; ``test_linkml_schema.py`` checks
it is the one the convention publishes.  It is the normative definition
of the on-disk metadata of format 0.10.0, so a writer change that drifts
from it fails here rather than in someone else's reader.  Update it when
the convention is bumped, never to make a failing writer pass.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

jsonschema = pytest.importorskip("jsonschema")

from zarr_vectors.multiresolution.coarsen import coarsen_level  # noqa: E402
from zarr_vectors.types.graphs import write_graph  # noqa: E402
from zarr_vectors.types.lines import write_lines  # noqa: E402
from zarr_vectors.types.meshes import write_mesh  # noqa: E402
from zarr_vectors.types.parametric import write_parametric_objects  # noqa: E402
from zarr_vectors.types.points import write_points  # noqa: E402
from zarr_vectors.types.polylines import write_polylines  # noqa: E402

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "schema" / "zarr_vectors.schema.json").read_text()
)
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)
BOX = ([0.0] * 3, [200.0] * 3)


def _nodes(store: str):
    for d, _, files in os.walk(store):
        if "zarr.json" in files:
            doc = json.loads(Path(d, "zarr.json").read_text())
            yield os.path.relpath(d, store), doc


def _assert_conforms(store: str) -> int:
    checked, problems = 0, []
    for rel, doc in _nodes(store):
        attrs = doc.get("attributes") or {}
        legacy = {"zv_array", "zarr_vectors_level", "zarr_vectors_presence",
                  "parametric_types"} & set(attrs)
        if legacy:
            problems.append(f"{rel}: pre-0.10.0 keys {sorted(legacy)}")
        if "zarr_vectors" not in attrs:
            continue
        checked += 1
        for err in VALIDATOR.iter_errors(doc):
            problems.append(f"{rel}: {err.json_path}: {err.message}")
    assert not problems, "\n".join(problems[:30])
    return checked


def _positions(n=1500, seed=0):
    return (np.random.default_rng(seed).random((n, 3)) * 200).astype("float32")


def test_points_with_a_pyramid_and_attributes(tmp_path):
    store = str(tmp_path / "p.zv")
    pos = _positions()
    write_points(store, pos, chunk_shape=(50.0,) * 3, bounds=BOX,
                 object_ids=np.arange(len(pos)) // 30,
                 vertex_attributes={"intensity": np.ones(len(pos), "float32"),
                                    "rgb": np.zeros((len(pos), 3), "uint8")},
                 object_attributes={"volume": np.ones(len(pos) // 30, "float32")})
    coarsen_level(store, 0, 1, coarsen_factor=2.0)
    root = json.loads(Path(store, "zarr.json").read_text())["attributes"]
    assert root["multiscales"] == {"layout": [{"asset": "0"}, {"asset": "1"}]}
    assert _assert_conforms(store) > 5


def test_a_sharded_graph(tmp_path):
    store = str(tmp_path / "g.zv")
    pos = _positions(seed=1)
    edges = np.stack([np.arange(len(pos) - 1), np.arange(1, len(pos))], axis=1)
    write_graph(store, pos, edges, chunk_shape=(50.0,) * 3, bounds=BOX,
                object_ids=np.arange(len(pos)) // 30, shard_shape=2)
    assert _assert_conforms(store) > 5


def test_a_mesh(tmp_path):
    store = str(tmp_path / "m.zv")
    verts = _positions(300, seed=2)
    faces = np.stack([np.arange(298), np.arange(1, 299), np.arange(2, 300)], axis=1)
    write_mesh(store, verts, faces, chunk_shape=(100.0,) * 3, bounds=BOX)
    assert _assert_conforms(store) > 3


def test_polylines_and_lines(tmp_path):
    rng = np.random.default_rng(3)
    poly = str(tmp_path / "pl.zv")
    write_polylines(poly, [(rng.random((20, 3)) * 200).astype("float32") for _ in range(10)],
                    chunk_shape=(100.0,) * 3, bounds=BOX)
    lines = str(tmp_path / "l.zv")
    write_lines(lines, (rng.random((10, 2, 3)) * 200).astype("float32"),
                chunk_shape=(100.0,) * 3, bounds=BOX)
    assert _assert_conforms(poly) > 3
    assert _assert_conforms(lines) > 3


def test_parametric_objects(tmp_path):
    store = str(tmp_path / "par.zv")
    write_parametric_objects(
        store,
        [{"type": "plane", "coefficients": [0, 1, 0, -50], "name": "coronal"},
         {"type": "sphere", "coefficients": [10, 10, 10, 5], "name": "ball"}],
        create_new_store=True,
    )
    assert _assert_conforms(store) >= 3
