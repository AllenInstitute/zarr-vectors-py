"""The ``building`` surface, and what it actually covers.

Two audiences need two contracts.  Applications read data and belong on
:mod:`zarr_vectors.api`.  Ingest converters, pyramid builders and
exporters work *on* the physical layout, so hiding it from them would be
hiding their subject; they get :mod:`zarr_vectors.building` instead — a
smaller surface with a promise attached, rather than the whole of
``core``.

The promise is only worth something if it is enough.  The coverage test
below measures that against the real consuming repositories when they are
present, so "this is the supported surface" is a measurement rather than
an assertion.
"""

from __future__ import annotations

import ast
import collections
import pathlib

import pytest

import zarr_vectors.building as building

# Where the consuming repositories live when checked out beside this one.
# Absent in CI, which is why every test that needs them skips rather than
# fails: a contract test that cannot run is not a contract violation.
_SIBLINGS = ("zarr-vectors-tools", "BRIDGE", "zv-ngtools")

# Modules whose contents may change without notice.
INTERNAL_PREFIXES = (
    "zarr_vectors.core",
    "zarr_vectors.encoding",
    "zarr_vectors.spatial",
    "zarr_vectors.lazy",
    "zarr_vectors.ops",
    "zarr_vectors.sharding",
    "zarr_vectors.multiresolution",
    "zarr_vectors._engine",
)


def _consumer_roots() -> list[pathlib.Path]:
    here = pathlib.Path(__file__).resolve().parents[2]
    return [here / name for name in _SIBLINGS if (here / name).is_dir()]


def _internal_imports() -> collections.Counter:
    """Every ``from zarr_vectors.<internal> import X`` downstream, counted."""
    counts: collections.Counter = collections.Counter()
    for root in _consumer_roots():
        for path in root.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(errors="ignore"))
            except (SyntaxError, OSError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if not node.module or not node.module.startswith(INTERNAL_PREFIXES):
                    continue
                for alias in node.names:
                    counts[(node.module, alias.name)] += 1
    return counts


class TestTheSurfaceItself:
    def test_every_exported_name_resolves(self):
        missing = [n for n in building.__all__ if not hasattr(building, n)]
        assert not missing, f"__all__ names nothing importable: {missing}"

    def test_nothing_public_is_private_by_name(self):
        # A leading underscore in the supported surface would say two
        # contradictory things at once.
        assert not [n for n in building.__all__ if n.startswith("_")]

    def test_the_promoted_helpers_take_names_not_zarr_nodes(self):
        # is_sharded's internal counterpart takes a resolved zarr node,
        # which means the caller has to obtain one -- and obtaining one is
        # the reach past the API this module exists to remove.
        import inspect

        params = list(inspect.signature(building.is_sharded).parameters)
        assert params == ["level_group", "array_name"]

    def test_rebuild_presence_is_offered_instead_of_a_zarr_group_walk(self):
        assert callable(building.rebuild_presence)

    def test_selective_manifest_reads_are_in_the_surface(self):
        # The absence of this is why a consumer imports a private block
        # expander and a layout sentinel.
        assert callable(building.read_object_manifests)
        assert callable(building.expand_manifest_blocks)


@pytest.mark.skipif(not _consumer_roots(), reason="consumer repos not checked out")
class TestCoverage:
    def test_it_covers_most_of_what_consumers_actually_import(self):
        counts = _internal_imports()
        exported = set(building.__all__)
        covered = sum(c for (_, name), c in counts.items() if name in exported)
        total = sum(counts.values())
        assert total > 0
        share = covered / total
        assert share >= 0.85, (
            f"building covers only {share:.0%} of the {total} internal import "
            f"sites downstream; the uncovered ones are: "
            f"{sorted({n for (_, n), _ in counts.items() if n not in exported})}"
        )

    def test_the_uncovered_remainder_is_reported_not_hidden(self):
        # Not an assertion about the number -- a way to see the list. A
        # silently shrinking surface is how "supported" stops meaning
        # anything.
        counts = _internal_imports()
        exported = set(building.__all__)
        uncovered = collections.Counter()
        for (module, name), c in counts.items():
            if name not in exported:
                uncovered[f"{module}.{name}"] += c
        print("\nStill imported from internals downstream:")
        for name, c in uncovered.most_common(20):
            print(f"  {c:3d}  {name}")
        assert True


class TestTheApiSurfaceIsSeparate:
    def test_building_is_not_the_data_api(self):
        # The two are deliberately disjoint in purpose: if a name is in
        # both, one of the two contracts is lying about what it is for.
        import zarr_vectors as zv

        overlap = set(building.__all__) & set(zv.__all__)
        # create_store/open_store are the documented crossover: a builder
        # needs them and they are already public.
        assert overlap <= {"create_store", "open_store", "Group"}, overlap
