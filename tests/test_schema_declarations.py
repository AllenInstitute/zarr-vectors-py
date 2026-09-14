"""A declared attribute is recorded, checked, and read back.

``Schema`` has always carried ``vertex_attributes`` / ``object_attributes``
/ ``link_attributes`` maps of :class:`AttributeSpec`.  Nothing read them:
``AttributeSpec`` appeared in five places in the package, all of them its
own definition or a re-export list.  Declaring an attribute created
nothing, pinned no dtype, recorded no unit, did not drive ``categorical``,
and did not survive ``Schema.from_store`` -- so the surface read as a
promise it did not keep.

These tests pin the three things that make it load-bearing: the store
records the declaration, the writer is held to it, and the annotation
reaches the array.
"""

from __future__ import annotations

import numpy as np
import pytest

import zarr_vectors as zv
from zarr_vectors.api.schema import SchemaConflict

BOUNDS = ([0.0, 0.0, 0.0], [400.0, 400.0, 400.0])
CELL = zv.Layout(cell_size=[200.0, 200.0, 200.0])


def _schema(**kw):
    kw.setdefault("bounds", BOUNDS)
    kw.setdefault("kind", "point_cloud")
    kw.setdefault("layout", CELL)
    return zv.Schema(**kw)


def _points(n=50, seed=3):
    return np.random.default_rng(seed).uniform(0, 400, (n, 3)).astype(np.float32)


@pytest.fixture
def declared(tmp_path):
    schema = _schema(vertex_attributes={
        "intensity": zv.AttributeSpec(
            dtype="float32", unit="microvolt",
            description="raw detector signal",
        ),
        "rgb": zv.AttributeSpec(dtype="float32", channels=3),
    })
    ds = zv.create(tmp_path / "d.zarrvectors", schema=schema)
    rng = np.random.default_rng(3)
    ds.add_points(
        _points(),
        attributes={
            "intensity": rng.random(50).astype(np.float32),
            "rgb": rng.random((50, 3)).astype(np.float32),
        },
    )
    return tmp_path / "d.zarrvectors"


class TestTheStoreRecordsIt:
    def test_schema_round_trips_through_the_store(self, declared):
        back = zv.Schema.from_store(zv.open(declared)._root_meta)
        assert set(back.vertex_attributes) == {"intensity", "rgb"}
        spec = back.vertex_attributes["intensity"]
        assert spec.dtype == "float32"
        assert spec.unit == "microvolt"
        assert spec.description == "raw detector signal"
        assert back.vertex_attributes["rgb"].channels == 3

    def test_it_is_an_additive_format_change(self, declared):
        """0.9.1, and a store that declares nothing carries no key."""
        assert zv.open(declared).format_version >= (0, 9, 1)

    def test_an_undeclared_store_is_simply_undeclared(self, tmp_path):
        ds = zv.create(tmp_path / "u.zarrvectors", schema=_schema())
        ds.add_points(_points())
        raw = ds.store.attrs.to_dict()["zarr_vectors"]
        assert "attribute_specs" not in raw
        back = zv.Schema.from_store(zv.open(tmp_path / "u.zarrvectors")._root_meta)
        assert back.vertex_attributes == {}


class TestTheWriterIsHeldToIt:
    def test_a_contradicting_dtype_is_refused(self, tmp_path):
        """Declared float32, handed float64, and nothing used to notice."""
        ds = zv.create(tmp_path / "w.zarrvectors", schema=_schema(
            vertex_attributes={"intensity": zv.AttributeSpec(dtype="float32")},
        ))
        with pytest.raises(SchemaConflict, match="declared 'float32'"):
            ds.add_points(
                _points(10),
                attributes={"intensity": np.zeros(10, dtype=np.float64)},
            )

    def test_a_contradicting_channel_count_is_refused(self, tmp_path):
        ds = zv.create(tmp_path / "c.zarrvectors", schema=_schema(
            vertex_attributes={"rgb": zv.AttributeSpec(dtype="float32", channels=3)},
        ))
        with pytest.raises(SchemaConflict, match="channel"):
            ds.add_points(
                _points(10),
                attributes={"rgb": np.zeros((10, 4), dtype=np.float32)},
            )

    def test_an_undeclared_attribute_is_still_accepted(self, tmp_path):
        """Declaring is how you pin a type, not how you restrict the set.

        Passing something the schema never mentioned is how a store is
        extended, and must stay allowed.
        """
        ds = zv.create(tmp_path / "x.zarrvectors", schema=_schema(
            vertex_attributes={"intensity": zv.AttributeSpec(dtype="float32")},
        ))
        ds.add_points(
            _points(10),
            attributes={"anything": np.zeros(10, dtype=np.int16)},
        )
        assert zv.open(tmp_path / "x.zarrvectors").level(0).attribute_names(
            "vertex",
        ) == ("anything",)


class TestTheAnnotationReachesTheArray:
    def test_unit_and_description_land_on_the_attribute_array(self, declared):
        """So a store can say what its intensity is measured in."""
        meta = zv.open(declared).level(0).store.read_array_meta(
            "vertex_attributes/intensity",
        )
        assert meta["unit"] == "microvolt"
        assert meta["description"] == "raw detector signal"

    def test_the_writers_own_fields_are_untouched(self, declared):
        """The declaration adds; it does not overwrite what the writer knows."""
        meta = zv.open(declared).level(0).store.read_array_meta(
            "vertex_attributes/rgb",
        )
        assert meta["dtype"] == "float32"
        assert len(meta["channel_names"]) == 3


class TestOpenOrCreateCanSeeIt:
    def test_a_disagreeing_declaration_is_a_conflict(self, declared):
        with pytest.raises(SchemaConflict, match="intensity"):
            zv.open_or_create(declared, schema=_schema(
                vertex_attributes={"intensity": zv.AttributeSpec(dtype="int16")},
            ))

    def test_an_agreeing_declaration_opens(self, declared):
        ds = zv.open_or_create(declared, schema=_schema(
            vertex_attributes={
                "intensity": zv.AttributeSpec(
                    dtype="float32", unit="microvolt",
                    description="raw detector signal",
                ),
            },
        ))
        assert ds.level(0).attribute_names("vertex") == ("intensity", "rgb")

    def test_declaring_something_new_is_not_a_conflict(self, declared):
        """Extending a store is legitimate; only disagreement is not."""
        ds = zv.open_or_create(declared, schema=_schema(
            vertex_attributes={"brand_new": zv.AttributeSpec(dtype="int8")},
        ))
        assert ds is not None

    def test_a_disagreeing_kind_is_a_conflict(self, declared):
        """``kind`` pins how a store is decoded, so it must be compared."""
        with pytest.raises(SchemaConflict, match="kind"):
            zv.open_or_create(declared, schema=_schema(kind="mesh"))


def test_profile_is_gone():
    """It named a concept that did not exist and nothing read it."""
    assert not hasattr(zv.Schema(), "profile")


class TestAutomaticLayout:
    """``Layout()`` untouched should produce a store, not one giant chunk.

    ``cells="auto"`` resolved to a single cell per axis while the class
    docstring promised "a caller who never touches Layout gets a sensible
    store".  ``Schema.expected.n_vertices`` was right there -- it already
    sized the shards -- and nothing consulted it for the grid.
    """

    def test_a_size_hint_produces_a_real_grid(self):
        schema = _schema(
            bounds=([0.0] * 3, [1000.0] * 3),
            expected=zv.SizeHints(n_vertices=200_000_000),
            layout=zv.Layout(),
        )
        resolved = schema.layout.resolve(schema)
        grid = zv.Grid.plan(schema.bounds, cell_size=resolved.chunk_shape)
        assert grid.cells > 1
        # ...and the grid it picks is one Grid.capacity signs off on.
        assert grid.capacity(n_vertices=200_000_000).fits

    def test_more_data_means_more_cells(self):
        def cells(n):
            schema = _schema(
                bounds=([0.0] * 3, [1000.0] * 3),
                expected=zv.SizeHints(n_vertices=n), layout=zv.Layout(),
            )
            r = schema.layout.resolve(schema)
            return zv.Grid.plan(schema.bounds, cell_size=r.chunk_shape).cells

        assert cells(10_000_000) < cells(200_000_000) < cells(2_000_000_000)

    def test_no_hint_still_means_one_cell_per_axis(self):
        """The honest answer when there is nothing to divide by.

        Pinned deliberately: guessing a grid for data of unknown size
        trades a known cost for an unknown one.
        """
        schema = _schema(bounds=([0.0] * 3, [1000.0] * 3), layout=zv.Layout())
        resolved = schema.layout.resolve(schema)
        assert resolved.chunk_shape == (1000.0, 1000.0, 1000.0)

    def test_an_explicit_cells_still_wins(self):
        schema = _schema(
            bounds=([0.0] * 3, [1000.0] * 3),
            expected=zv.SizeHints(n_vertices=2_000_000_000),
            layout=zv.Layout(cells=4),
        )
        assert schema.layout.resolve(schema).chunk_shape == (250.0,) * 3

    def test_automatic_compression_means_compression(self, monkeypatch):
        """"auto" resolved to *no* compression, which is not what it says."""
        monkeypatch.delenv("ZARR_VECTORS_COMPRESSION", raising=False)
        schema = _schema(bounds=([0.0] * 3, [1000.0] * 3), layout=zv.Layout())
        assert schema.layout.resolve(schema).compressor == "zstd"

    def test_the_environment_variable_still_wins(self, monkeypatch):
        monkeypatch.setenv("ZARR_VECTORS_COMPRESSION", "blosc")
        schema = _schema(bounds=([0.0] * 3, [1000.0] * 3), layout=zv.Layout())
        assert schema.layout.resolve(schema).compressor == "blosc"

    def test_explicitly_asking_for_none_is_honoured(self):
        schema = _schema(
            bounds=([0.0] * 3, [1000.0] * 3), layout=zv.Layout(compression=None),
        )
        assert schema.layout.resolve(schema).compressor is None
