# How to cite

This page gives citation entries for `zarr-vectors-py` and for the Zarr
Vectors specification it implements. Neither has a DOI; cite the
repositories.

## zarr-vectors-py

Cite the software repository and the version you used
(`zarr_vectors.__version__`):

```bibtex
@software{zarr_vectors_py,
  author    = {Keenlyside, Andrew and Collman, Forrest},
  title     = {{zarr-vectors-py: Python utilities for Zarr Vectors}},
  version   = {<version>},
  publisher = {GitHub},
  url       = {https://github.com/AllenInstitute/zarr-vectors-py}
}
```

## The Zarr Vectors specification

The Zarr Vectors format was specified by Forrest Collman at the Allen
Institute for Brain Science:

```bibtex
@misc{collman_zarr_vectors,
  author    = {Collman, Forrest},
  title     = {{Zarr Vectors}},
  publisher = {GitHub},
  url       = {https://github.com/AllenInstitute/zarr_vectors}
}
```

## Dependencies

If your work relies on what one of these provides, cite it too:

| Functionality | Cite |
|--------------|------|
| Zarr v3 storage | [Zarr](https://zenodo.org/record/3773454) |
| OME-Zarr multiscale metadata | [Moore et al., 2021](https://doi.org/10.1007/s00418-021-02029-x) |
| Draco mesh compression (`draco` extra) | [Google Draco](https://github.com/google/draco) |
| Neuroglancer viewer | [Google Neuroglancer](https://github.com/google/neuroglancer) |

## Acknowledgement text

A suggested sentence for a methods section:

> Vector geometry data were stored in the Zarr Vectors format [Collman]
> using `zarr-vectors-py` [Keenlyside and Collman].
