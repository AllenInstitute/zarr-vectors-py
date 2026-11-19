"""Format 0.10.0 input checks: the root version string and the CRS."""

from __future__ import annotations

import pytest

from zarr_vectors import building as zb
from zarr_vectors.core.metadata import check_root_layout
from zarr_vectors.exceptions import MetadataError, UnsupportedFormatVersionError

BOUNDS = ([0.0, 0.0, 0.0], [10.0, 10.0, 10.0])


def test_an_unparseable_version_is_malformed():
    with pytest.raises(MetadataError, match="not a valid"):
        check_root_layout({"kind": "root", "zv_version": "ten"})


def test_a_newer_minor_version_is_refused():
    with pytest.raises(UnsupportedFormatVersionError, match="newer"):
        check_root_layout({"kind": "root", "zv_version": "0.11.0"})


def test_a_crs_must_be_a_projjson_object(tmp_path):
    with pytest.raises(MetadataError, match="PROJJSON"):
        zb.create_store(str(tmp_path / "s.zv"), bounds=BOUNDS, chunk_shape=(5.0,) * 3,
                        crs="EPSG:4326")


def test_a_crs_lands_in_proj_projjson(tmp_path):
    crs = {"type": "EngineeringCRS", "name": "scanner"}
    root = zb.create_store(str(tmp_path / "s.zv"), bounds=BOUNDS, chunk_shape=(5.0,) * 3,
                           crs=crs)
    attrs = root.attrs.to_dict()
    assert attrs["proj:projjson"] == crs
    assert "crs" not in attrs["zarr_vectors"]
    uuids = {e["uuid"] for e in attrs["zarr_conventions"]}
    assert "f17cb550-5864-4468-aeb7-f3180cfb622f" in uuids
