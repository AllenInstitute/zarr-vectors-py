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
import subprocess

import pytest

import zarr_vectors.building as building
from zarr_vectors._stability import INTERNAL as INTERNAL_PREFIXES

# Where the consuming repositories live when checked out beside this one.
# Absent in CI, which is why every test that needs them skips rather than
# fails: a contract test that cannot run is not a contract violation.
_SIBLINGS = ("zarr-vectors-tools", "BRIDGE", "zv-ngtools")

# Directories that hold COPIES of source rather than source.  A stale
# ``build/lib/`` tree is still importable-looking Python, and counting it
# measured a pre-migration snapshot of the consumer alongside the
# consumer: 245 sites at 94.7% (passing) where the live tree had 16 at
# 62.5% (failing).  Every "still imported from internals" line the old
# version printed came from there, two of them naming an API generation
# that no longer exists.
_NOT_SOURCE = frozenset({
    "build", "dist", ".venv", "venv", ".tox", "site-packages",
    "__pycache__", ".git", ".eggs", "node_modules",
})

# Uncovered names that are a decision rather than a gap.  Asserting on
# this SET rather than on a percentage is deliberate: at n≈16 a ratio is
# noise, and a threshold only ever gets negotiated downward.  A new
# uncovered name fails the test and has to be either exported or admitted
# here, in one line, with this comment above it.
_ALLOWED_UNCOVERED = frozenset({
    # The lazy layer is deprecated; its tests import its entry points on
    # purpose and retire with it.
    "open_zv",
    "open_zvr",
    # Removed from core entirely; the remaining references are prose in
    # migration notes, and one consumer not yet moved to read_links.
    "read_cross_chunk_links",
})


def _consumer_roots() -> list[pathlib.Path]:
    here = pathlib.Path(__file__).resolve().parents[2]
    return [here / name for name in _SIBLINGS if (here / name).is_dir()]


def _source_files(root: pathlib.Path) -> list[pathlib.Path]:
    """The repo's own ``.py`` files — tracked, and not a build artefact.

    Prefers ``git ls-files`` so generated and ignored trees are excluded
    by the repo's own definition.  Falls back to a walk with a component
    blocklist, because at least one sibling checks example stores in and a
    git-only rule is not sufficient there.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "*.py"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
        paths = [root / line for line in out.splitlines() if line.strip()]
        if paths:
            return [p for p in paths if p.is_file()]
    except (OSError, subprocess.SubprocessError):
        pass
    return [
        p for p in root.rglob("*.py")
        if not _NOT_SOURCE & set(p.relative_to(root).parts)
    ]


def _internal_imports() -> collections.Counter:
    """Every ``from zarr_vectors.<internal> import X`` downstream, counted.

    Counted per import STATEMENT, not per name: a four-name
    ``from ... import (a, b, c, d)`` is one place to edit, and weighing it
    four times made the ratio a function of how the imports were wrapped.
    """
    counts: collections.Counter = collections.Counter()
    for root in _consumer_roots():
        for path in _source_files(root):
            try:
                tree = ast.parse(path.read_text(errors="ignore"))
            except (SyntaxError, OSError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if not node.module or not node.module.startswith(INTERNAL_PREFIXES):
                    continue
                counts[(node.module, tuple(a.name for a in node.names))] += 1
    return counts


def _uncovered_names() -> set[str]:
    exported = set(building.__all__)
    return {
        name
        for (_module, names), _count in _internal_imports().items()
        for name in names
        if name not in exported
    }


class TestTheSurfaceItself:
    def test_every_exported_name_resolves(self):
        missing = [n for n in building.__all__ if not hasattr(building, n)]
        assert not missing, f"__all__ names nothing importable: {missing}"

    def test_nothing_public_is_private_by_name(self):
        # A leading underscore in the supported surface would say two
        # contradictory things at once.
        assert not [n for n in building.__all__ if n.startswith("_")]

    def test_the_promoted_helpers_take_names_not_zarr_nodes(self):
        # array_is_sharded's internal counterpart takes a resolved zarr
        # node, which means the caller has to obtain one -- and obtaining
        # one is the reach past the API this module exists to remove.
        import inspect

        params = list(inspect.signature(building.array_is_sharded).parameters)
        assert params == ["level_group", "array_name"]

    def test_rebuild_presence_is_offered_instead_of_a_zarr_group_walk(self):
        assert callable(building.rebuild_presence)
        # ...and the walk it needs, so a consumer does not fork the
        # private _is_per_chunk_array to reimplement it.
        assert callable(building.per_chunk_array_paths)

    def test_selective_manifest_reads_are_in_the_surface(self):
        # The absence of this is why a consumer imports a private block
        # expander and a layout sentinel.
        assert callable(building.read_object_manifests)
        assert callable(building.expand_manifest_blocks)

    def test_the_store_creating_writers_are_here(self):
        # Not superseded by Dataset.add_*: write_points alone takes 18
        # keyword parameters, and their subject IS the physical layout.
        for name in ("write_points", "write_polylines", "write_lines",
                     "write_mesh", "write_graph", "init_skeleton_store"):
            assert callable(getattr(building, name)), name

    def test_the_supported_group_methods_all_exist(self):
        from zarr_vectors.core.group import Group

        missing = sorted(
            n for n in building.GROUP_SUPPORTED_METHODS if not hasattr(Group, n)
        )
        assert not missing, f"promised but absent from Group: {missing}"


@pytest.mark.skipif(not _consumer_roots(), reason="consumer repos not checked out")
class TestCoverage:
    def test_nothing_uncovered_but_the_deliberate_exceptions(self):
        uncovered = _uncovered_names()
        unexpected = sorted(uncovered - _ALLOWED_UNCOVERED)
        assert not unexpected, (
            f"downstream imports these from internals and building does not "
            f"export them: {unexpected}. Either add them to building.__all__, "
            f"or add them to _ALLOWED_UNCOVERED with the reason."
        )

    def test_the_allowlist_does_not_outlive_its_reasons(self):
        # An entry that no consumer imports any more is a stale excuse.
        stale = sorted(_ALLOWED_UNCOVERED - _uncovered_names())
        if stale:
            pytest.skip(f"allowlist entries no longer imported anywhere: {stale}")

    def test_the_remainder_is_reported_not_hidden(self):
        # Not an assertion -- a way to see the list. A silently shrinking
        # surface is how "supported" stops meaning anything.
        counts = _internal_imports()
        exported = set(building.__all__)
        uncovered: collections.Counter = collections.Counter()
        for (module, names), c in counts.items():
            for name in names:
                if name not in exported:
                    uncovered[f"{module}.{name}"] += c
        print("\nStill imported from internals downstream:")
        for name, c in uncovered.most_common(20):
            print(f"  {c:3d}  {name}")


class TestTheApiSurfaceIsSeparate:
    def test_a_shared_name_is_the_same_object(self):
        """Overlap is fine; DISAGREEMENT is not.

        The old assertion required the two surfaces to be near-disjoint,
        which says nothing about the failure that actually happened:
        ``building.is_sharded`` and ``sharding.io.is_sharded`` shared a
        name and took different arguments, so swapping one import for the
        other silently returned False.  What matters is that a name means
        one thing everywhere it appears.
        """
        import zarr_vectors as zv

        for name in set(building.__all__) & set(zv.__all__):
            assert getattr(building, name) is getattr(zv, name), (
                f"{name!r} is a different object on zarr_vectors and on "
                f"zarr_vectors.building; one of them is lying about what "
                f"the name means."
            )

    def test_building_does_not_shadow_an_internal_name_with_other_semantics(self):
        """A re-export must be the same function, not a same-named one."""
        import importlib
        import inspect

        for name in building.__all__:
            obj = getattr(building, name)
            if not inspect.isfunction(obj):
                continue
            origin = getattr(obj, "__module__", "")
            if not origin.startswith("zarr_vectors."):
                continue
            if origin == "zarr_vectors.building":
                continue  # defined here; nothing to shadow
            source = importlib.import_module(origin)
            assert getattr(source, name, obj) is obj, (
                f"building.{name} is not {origin}.{name}. A supported name "
                f"that differs from the internal one it appears to re-export "
                f"is how a caller swaps an import line and gets a different "
                f"answer with no error."
            )
