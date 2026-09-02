# How to cite

## zarr-vectors-py

If you use `zarr-vectors-py` in research that leads to a publication,
please cite the software repository:

```bibtex
@software{zarr_vectors_py,
  author       = {{BRIDGE Neuroscience}},
  title        = {{zarr-vectors-py: Python tools for the Zarr Vectors}},
  year         = {2024},
  publisher    = {GitHub},
  url          = {https://github.com/BRIDGE-Neuroscience/zarr-vectors-py},
  note         = {Aligned to the Zarr Vectors specification by Forest Collman,
                  Allen Institute for Brain Sciences.
                  \url{https://github.com/AllenInstitute/zarr_vectors}}
}
```

## The Zarr Vectors specification

The Zarr Vectors format was originally specified by Forest Collman at the Allen
Institute for Brain Sciences. Please also cite the upstream specification:

```bibtex
@misc{collman_zarr_vectors,
  author       = {Collman, Forest},
  title        = {{Zarr Vectors: a cloud-native format for spatial vector data}},
  year         = {2023},
  publisher    = {GitHub},
  url          = {https://github.com/AllenInstitute/zarr_vectors}
}
```

## zv-ngtools

If you use the Neuroglancer integration (`zv-ngtools`) in your work:

```bibtex
@software{zv_ngtools,
  author       = {{BRIDGE Neuroscience}},
  title        = {{zv-ngtools: Neuroglancer integration for zarr-vectors}},
  year         = {2024},
  publisher    = {GitHub},
  url          = {https://github.com/BRIDGE-Neuroscience/zv-ngtools},
  note         = {Fork of ngtools by Yael Balbastre (neuroscales/ngtools).
                  \url{https://github.com/neuroscales/ngtools}}
}
```

## Dependency citations

If your work uses specific functionality provided by upstream dependencies,
please also cite:

| Functionality | Cite |
|--------------|------|
| Zarr v3 storage | [Zarr development team](https://zenodo.org/record/3773454) |
| OME-Zarr multiscale metadata | [Moore et al., 2021](https://doi.org/10.1007/s00418-021-02029-x) |
| Draco mesh compression | [Google Draco](https://github.com/google/draco) |
| TRX format ingest | [TRX format](https://tractography-file-format.github.io/) |
| Neuroglancer viewer | [Google Neuroglancer](https://github.com/google/neuroglancer) |

## Acknowledgement text

A suggested acknowledgement sentence for methods sections:

> "Vector geometry data (streamlines / point clouds / skeletons) were
> stored and served using the Zarr Vectors [Collman, 2023] as
> implemented in `zarr-vectors-py` [BRIDGE Neuroscience, 2024] and
> visualised using Neuroglancer [Google] via `zv-ngtools`
> [BRIDGE Neuroscience, 2024]."
