"""Format 0.10.0 metadata migration, end to end against the real 0.9.4 writer.

Stores are written by the package as of the last 0.9.4 commit (extracted
from git history, as ``test_octree_guards`` does for its old reader), read
by that same old code, migrated in place, and read again by this build.
The reads must agree exactly: the migration moves metadata and touches no
cell.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from zarr_vectors.building import migrate_metadata_0_10
from zarr_vectors.core.store import open_store
from zarr_vectors.exceptions import UnsupportedFormatVersionError

REPO = Path(__file__).resolve().parents[1]
#: The last commit that writes format 0.9.4.
OLD_COMMIT = "fa333d112623ae529d023e5f98c7f88c555f7da3"

_WRITE = r"""
import sys, warnings
warnings.simplefilter("ignore")
import numpy as np
from zarr_vectors.types.points import write_points
from zarr_vectors.types.graphs import write_graph
from zarr_vectors.multiresolution.coarsen import coarsen_level

points, graph = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(0)
pos = (rng.random((2000, 3)) * 200).astype("float32")
oids = np.arange(len(pos)) // 40
write_points(points, pos, chunk_shape=(50.0,) * 3, bounds=([0] * 3, [200] * 3),
             object_ids=oids,
             vertex_attributes={"intensity": rng.random(len(pos)).astype("float32")})
coarsen_level(points, 0, 1, coarsen_factor=2.0)
edges = np.stack([np.arange(len(pos) - 1), np.arange(1, len(pos))], axis=1)
write_graph(graph, pos, edges, chunk_shape=(50.0,) * 3, bounds=([0] * 3, [200] * 3),
            object_ids=oids, shard_shape=2)
"""

# Run by the old build and by this one: the same summary of every read.
_SUMMARY = r"""
import sys, json, hashlib, warnings
warnings.simplefilter("ignore")
import numpy as np
from zarr_vectors.types.points import read_points
from zarr_vectors.types.graphs import read_graph

def digest(a):
    a = np.asarray(a)
    if a.ndim == 2 and len(a):
        a = a[np.lexsort(a.T[::-1])]
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]

points, graph = sys.argv[1], sys.argv[2]
out = {}
for level in (0, 1):
    r = read_points(points, level=level)
    out[f"points{level}"] = [int(r["vertex_count"]), digest(r["positions"])]
r = read_points(points, object_ids=[3, 7])
out["points_objects"] = [int(r["vertex_count"]), digest(r["positions"])]
g = read_graph(graph)
out["graph"] = [int(g["node_count"]), digest(g["positions"]), digest(g["edges"])]
print(json.dumps(out, sort_keys=True))
"""


@pytest.fixture(scope="module")
def old_tree(tmp_path_factory):
    try:
        subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{OLD_COMMIT}^{{commit}}"],
                       check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"commit {OLD_COMMIT} is not in this checkout")
    tree = tmp_path_factory.mktemp("old")
    archive = subprocess.run(["git", "-C", str(REPO), "archive", OLD_COMMIT, "zarr_vectors"],
                             check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tree)], input=archive, check=True)
    return tree


def _run(script: str, *args: str, pythonpath: Path | None, cwd: Path) -> str:
    env = dict(os.environ)
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    res = subprocess.run([sys.executable, "-c", script, *args], cwd=str(cwd),
                         capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr[-3000:]
    return res.stdout


@pytest.fixture
def old_stores(old_tree, tmp_path):
    points, graph = tmp_path / "p.zv", tmp_path / "g.zv"
    _run(_WRITE, str(points), str(graph), pythonpath=old_tree, cwd=tmp_path)
    return str(points), str(graph)


def test_the_old_writer_really_wrote_0_9(old_stores):
    root = json.loads((Path(old_stores[0]) / "zarr.json").read_text())["attributes"]
    assert root["zarr_vectors"]["zv_version"].startswith("0.9.")
    assert isinstance(root["multiscales"], list)


def test_this_build_refuses_an_unmigrated_store(old_stores):
    with pytest.raises(UnsupportedFormatVersionError, match="migrate_metadata_0_10"):
        open_store(old_stores[0])


def test_migrated_stores_read_exactly_as_the_old_build_read_them(old_tree, old_stores, tmp_path):
    before = _run(_SUMMARY, *old_stores, pythonpath=old_tree, cwd=tmp_path)

    for store in old_stores:
        counts = migrate_metadata_0_10(store)
        assert counts["root"] == 1 and counts["levels"] >= 1 and counts["arrays"] > 0

    after = _run(_SUMMARY, *old_stores, pythonpath=None, cwd=REPO)
    assert json.loads(after) == json.loads(before)


def test_migration_only_rewrites_metadata(old_stores):
    def chunks(path):
        out = {}
        for d, _, files in os.walk(path):
            for f in files:
                if f != "zarr.json":
                    p = os.path.join(d, f)
                    out[os.path.relpath(p, path)] = Path(p).read_bytes()
        return out

    before = [chunks(s) for s in old_stores]
    for s in old_stores:
        migrate_metadata_0_10(s)
    assert [chunks(s) for s in old_stores] == before


def test_migration_is_idempotent_and_dry_run_writes_nothing(old_stores):
    store = old_stores[0]
    doc = (Path(store) / "zarr.json").read_text()
    dry = migrate_metadata_0_10(store, dry_run=True)
    assert dry["root"] == 1
    assert (Path(store) / "zarr.json").read_text() == doc

    migrate_metadata_0_10(store)
    assert migrate_metadata_0_10(store) == {"root": 0, "levels": 0, "arrays": 0, "other": 0}


def test_no_legacy_key_survives(old_stores):
    legacy = {"zv_array", "zvf_array", "zarr_vectors_level", "zarr_vectors_presence",
              "nonempty_chunks", "chunk_grid_origin", "parametric_types"}
    for store in old_stores:
        migrate_metadata_0_10(store)
        for d, _, files in os.walk(store):
            if "zarr.json" in files:
                attrs = json.loads(Path(d, "zarr.json").read_text()).get("attributes", {})
                assert not legacy & set(attrs), (d, sorted(legacy & set(attrs)))
