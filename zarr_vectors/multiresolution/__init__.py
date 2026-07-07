"""Multi-resolution (pyramid) coarsening.

Core ships the simplest, **dependency-free** pyramid:

- :func:`zarr_vectors.multiresolution.coarsen.build_pyramid` /
  :func:`~zarr_vectors.multiresolution.coarsen.coarsen_level` — the
  metavertex-binning coarsener (the ``method="per_object"`` default).
- :func:`zarr_vectors.multiresolution.object_selection.apply_sparsity` with
  the ``"random"`` object-selection strategy.

Advanced approaches — per-geometry coarsening (quadric mesh decimation,
Douglas–Peucker polyline simplification, graph/point metanodes) and
non-random object selection (spatial coverage, length/attribute-ranked,
point-thinning) — live in ``zarr-vectors-tools``.  That package registers
its implementations through
:mod:`zarr_vectors.multiresolution.registry` on import, so core can dispatch
to them by name (``method=...`` / ``sparsity_strategy=...``) without a hard
dependency.  Requesting one when it is not registered raises a clear
"install zarr-vectors-tools" error.
"""
