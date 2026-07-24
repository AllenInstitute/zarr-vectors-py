# Zarr Vectors (ZV) format

Reference schema for the Zarr Vectors (ZV) on-disk metadata (0.5.0). Covers the root-level ``.zattrs`` block (``zarr_vectors`` envelope), the NGFF ``multiscales`` block (which now carries the canonical axes list — ZV no longer duplicates them under ``spatial_index_dims``), the per-resolution-level ``.zattrs`` block (``zarr_vectors_level`` envelope), and the per-array ``.zattrs`` shapes emitted by the writers in ``zarr_vectors/core/arrays.py`` (discriminator slot ``zv_array``).


URI: https://w3id.org/zarr-vectors/schema/0.5

Name: zarr_vectors



## Classes

| Class | Description |
| --- | --- |
| [AttributeMeta](AttributeMeta.md) | `` |
| [Axis](Axis.md) | One axis of the spatial index |
| [BoundingBox](BoundingBox.md) | Two parallel ``ndim``-length arrays representing the global ``(min_corner, ma... |
| [CRS](CRS.md) | Free-form coordinate reference system metadata |
| [FragmentAttributeMeta](FragmentAttributeMeta.md) | `` |
| [GroupingsAttributeMeta](GroupingsAttributeMeta.md) | `` |
| [GroupingsMeta](GroupingsMeta.md) | `` |
| [LevelMetadata](LevelMetadata.md) | Per-resolution-level `` |
| [LinkAttributeFamilyMeta](LinkAttributeFamilyMeta.md) | `` |
| [LinkAttributeMeta](LinkAttributeMeta.md) | `` |
| [LinksFamilyMeta](LinksFamilyMeta.md) | `` |
| [LinksMeta](LinksMeta.md) | `` |
| [ObjectAttributeMeta](ObjectAttributeMeta.md) | `` |
| [ObjectIndexMeta](ObjectIndexMeta.md) | `` |
| [RootMetadata](RootMetadata.md) | Root-level `` |
| [VerticesMeta](VerticesMeta.md) | `` |



## Slots

| Slot | Description |
| --- | --- |
| [arrays_present](arrays_present.md) | Names of arrays present in the level group |
| [base_bin_shape](base_bin_shape.md) | Supervoxel bin edge lengths at level 0 |
| [bin_ratio](bin_ratio.md) | Integer fold-change per axis relative to level 0 |
| [bin_shape](bin_shape.md) | Per-axis supervoxel edge lengths at this level |
| [bounds](bounds.md) | Global vertex bounding box |
| [channel_names](channel_names.md) | For multi-channel per-vertex attributes, the channel labels |
| [chunk_attribute_name](chunk_attribute_name.md) | Name of the per-vertex attribute used as the leading chunk axis |
| [chunk_attribute_values](chunk_attribute_values.md) | Ordered list mapping attribute-bin index to original attribute value |
| [chunk_dims](chunk_dims.md) | Chunk-key axis names; the leading axis names appear first |
| [chunk_shape](chunk_shape.md) | Physical spatial chunk size per axis (all values > 0) |
| [coarsening_method](coarsening_method.md) | How this level was generated (e |
| [cross_chunk_strategy](cross_chunk_strategy.md) |  |
| [cross_level_depth](cross_level_depth.md) | Maximum absolute level delta for which cross-pyramid-level link arrays are ma... |
| [cross_level_storage](cross_level_storage.md) | Whether cross-level link arrays are written in both directions (``explicit``:... |
| [crs](crs.md) | Optional coordinate reference system metadata (free-form dict matching whatev... |
| [directed](directed.md) | When true, endpoint order is meaningful and preserved (no canonical sort): ``... |
| [dtype](dtype.md) | Numpy dtype string of the array's value type (e |
| [encoding](encoding.md) | How the chunk bytes are encoded |
| [format_capabilities](format_capabilities.md) | Optional 0 |
| [geometry_types](geometry_types.md) | One or more geometry kinds present in the store |
| [has_perm](has_perm.md) | Whether each row of this links array carries a leading ``perm_idx`` column — ... |
| [inherited_num_objects](inherited_num_objects.md) | OID-space size inherited from the parent level (= ``parent_level |
| [level](level.md) | Resolution level index (0 = full resolution) |
| [level_delta](level_delta.md) | Pyramid-level delta between the source side (the level that owns this array) ... |
| [link_width](link_width.md) | Width of a links row (1 for parent→child metanode references, 2 for edges, 3 ... |
| [links_convention](links_convention.md) |  |
| [max_corner](max_corner.md) | Per-axis maxima |
| [min_corner](min_corner.md) | Per-axis minima |
| [name](name.md) | NGFF axis or attribute name (e |
| [num_groups](num_groups.md) | Total grouping count |
| [num_links](num_links.md) | Family-wide *logical* link count — one per input record, however many physica... |
| [num_objects](num_objects.md) | Total object count this array carries |
| [num_physical_records](num_physical_records.md) | On-disk record row count across every offsets array and cell of a ``links/<de... |
| [object_index_convention](object_index_convention.md) |  |
| [object_sparsity](object_sparsity.md) | Fraction of objects retained at this level |
| [offsets](offsets.md) | The ``link_width - 1`` relative offsets naming this array, each of ``sid_ndim... |
| [parent_level](parent_level.md) | Source level index (None for level 0) |
| [preserves_object_ids](preserves_object_ids.md) | True for levels written by the per-object pyramid regime |
| [reduction_factor](reduction_factor.md) | Multi-resolution coarsening factor (≥ 2) |
| [row_shape](row_shape.md) | Tail dimensions of one attribute row (``[]`` for a 1-D per-link attribute), l... |
| [shape](shape.md) | Shape of a dense per-object/per-group array |
| [shared_fragments](shared_fragments.md) | True when per-chunk fragments may be referenced by multiple objects' manifest... |
| [sid_ndim](sid_ndim.md) | Number of spatial-index dimensions encoded in chunk keys |
| [store](store.md) | Cell-duplication policy for a links family; see CrossChunkStore |
| [type](type.md) | NGFF axis type — "space", "time", or "channel" |
| [unit](unit.md) | NGFF unit string (e |
| [vertex_count](vertex_count.md) | Total number of vertices at this level |
| [zv_array](zv_array.md) | Discriminator slot identifying the kind of per-array `` |
| [zv_version](zv_version.md) | ZV spec version this store was written against (e |


## Enumerations

| Enumeration | Description |
| --- | --- |
| [CrossChunkStore](CrossChunkStore.md) | Cell-duplication policy for a ``links/<delta>/`` family |
| [CrossChunkStrategy](CrossChunkStrategy.md) | How connectivity that crosses chunk boundaries is represented |
| [CrossLevelStorage](CrossLevelStorage.md) | How cross-pyramid-level edges are stored in the multiscale links layout (``li... |
| [Encoding](Encoding.md) | Per-array encoding of vertex data |
| [FormatCapability](FormatCapability.md) | Optional feature tokens a store advertises in :attr:`RootMetadata |
| [GeometryType](GeometryType.md) | The kind of geometry a store (or one of its sub-types) holds |
| [LinksConvention](LinksConvention.md) | How intra-chunk links are represented for a polyline/graph/mesh |
| [ObjectIndexConvention](ObjectIndexConvention.md) | How the object_id → fragment mapping is encoded |
| [ZvArrayTag](ZvArrayTag.md) | Discriminator value for per-array `` |


## Types

| Type | Description |
| --- | --- |
| [Boolean](Boolean.md) | A binary (true or false) value |
| [Curie](Curie.md) | a compact URI |
| [Date](Date.md) | a date (year, month and day) in an idealized calendar |
| [DateOrDatetime](DateOrDatetime.md) | Either a date or a datetime |
| [Datetime](Datetime.md) | The combination of a date and time |
| [Decimal](Decimal.md) | A real number with arbitrary precision that conforms to the xsd:decimal speci... |
| [Double](Double.md) | A real number that conforms to the xsd:double specification |
| [Float](Float.md) | A real number that conforms to the xsd:float specification |
| [Integer](Integer.md) | An integer |
| [Jsonpath](Jsonpath.md) | A string encoding a JSON Path |
| [Jsonpointer](Jsonpointer.md) | A string encoding a JSON Pointer |
| [Ncname](Ncname.md) | Prefix part of CURIE |
| [Nodeidentifier](Nodeidentifier.md) | A URI, CURIE or BNODE that represents a node in a model |
| [Objectidentifier](Objectidentifier.md) | A URI or CURIE that represents an object in the model |
| [Sparqlpath](Sparqlpath.md) | A string encoding a SPARQL Property Path |
| [String](String.md) | A character string |
| [Time](Time.md) | A time object represents a (local) time of day, independent of any particular... |
| [Uri](Uri.md) | a complete URI |
| [Uriorcurie](Uriorcurie.md) | a URI or a CURIE |


## Subsets

| Subset | Description |
| --- | --- |



---

---
search:
  boost: 5.0
---

# Slot: arrays_present 


_Names of arrays present in the level group._



<div data-search-exclude markdown="1">



URI: [zv:arrays_present](https://w3id.org/zarr-vectors/schema/0.5/arrays_present)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LevelMetadata](LevelMetadata.md) | Per-resolution-level `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [LevelMetadata](LevelMetadata.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
| Multivalued | Yes |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:arrays_present |
| native | zv:arrays_present |




## LinkML Source

<details>
```yaml
name: arrays_present
description: Names of arrays present in the level group.
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- LevelMetadata
range: string
required: true
multivalued: true

```
</details></div>


---

---
search:
  boost: 10.0
---

# Class: AttributeMeta 


_``.zattrs`` for each ``attributes/<name>/`` array._



<div data-search-exclude markdown="1">



URI: [zv:AttributeMeta](https://w3id.org/zarr-vectors/schema/0.5/AttributeMeta)





```mermaid
 classDiagram
    class AttributeMeta
    click AttributeMeta href "../AttributeMeta/"
      AttributeMeta : channel_names
        
      AttributeMeta : dtype
        
      AttributeMeta : name
        
      AttributeMeta : zv_array
        
          
    
        
        
        AttributeMeta --> "1" ZvArrayTag : zv_array
        click ZvArrayTag href "../ZvArrayTag/"
    

        
      
```




<!-- no inheritance hierarchy -->

## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [zv_array](zv_array.md) | 1 <br/> [ZvArrayTag](ZvArrayTag.md) | Discriminator slot identifying the kind of per-array `` | direct |
| [name](name.md) | 1 <br/> [String](String.md) | NGFF axis or attribute name (e | direct |
| [dtype](dtype.md) | 1 <br/> [String](String.md) | Numpy dtype string of the array's value type (e | direct |
| [channel_names](channel_names.md) | * <br/> [String](String.md) | For multi-channel per-vertex attributes, the channel labels | direct |















## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:AttributeMeta |
| native | zv:AttributeMeta |






## LinkML Source

### Direct

<details>
```yaml
name: AttributeMeta
description: '``.zattrs`` for each ``attributes/<name>/`` array.'
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slots:
- zv_array
- name
- dtype
- channel_names
slot_usage:
  zv_array:
    name: zv_array
    required: true
    equals_string: attribute

```
</details>

### Induced

<details>
```yaml
name: AttributeMeta
description: '``.zattrs`` for each ``attributes/<name>/`` array.'
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slot_usage:
  zv_array:
    name: zv_array
    required: true
    equals_string: attribute
attributes:
  zv_array:
    name: zv_array
    description: 'Discriminator slot identifying the kind of per-array ``.zattrs``
      block.  Each writer in ``core/arrays.py`` stamps the corresponding token from
      :class:`ZvArrayTag`.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: AttributeMeta
    domain_of:
    - VerticesMeta
    - LinksFamilyMeta
    - LinksMeta
    - AttributeMeta
    - FragmentAttributeMeta
    - ObjectIndexMeta
    - ObjectAttributeMeta
    - GroupingsMeta
    - GroupingsAttributeMeta
    - LinkAttributeFamilyMeta
    - LinkAttributeMeta
    range: ZvArrayTag
    required: true
    equals_string: attribute
  name:
    name: name
    description: NGFF axis or attribute name (e.g. "x", "intensity").
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    slot_uri: schema:name
    owner: AttributeMeta
    domain_of:
    - Axis
    - AttributeMeta
    - FragmentAttributeMeta
    - ObjectAttributeMeta
    - GroupingsAttributeMeta
    - LinkAttributeFamilyMeta
    - LinkAttributeMeta
    range: string
    required: true
  dtype:
    name: dtype
    description: Numpy dtype string of the array's value type (e.g. "float32").
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: AttributeMeta
    domain_of:
    - VerticesMeta
    - LinksMeta
    - AttributeMeta
    - FragmentAttributeMeta
    - ObjectAttributeMeta
    - GroupingsAttributeMeta
    - LinkAttributeMeta
    range: string
    required: true
  channel_names:
    name: channel_names
    description: For multi-channel per-vertex attributes, the channel labels.
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: AttributeMeta
    domain_of:
    - AttributeMeta
    - FragmentAttributeMeta
    range: string
    multivalued: true

```
</details></div>


---

---
search:
  boost: 10.0
---

# Class: Axis 


_One axis of the spatial index.  Mirrors the OME-Zarr NGFF axis object (RFC 4/5)._

__



<div data-search-exclude markdown="1">



URI: [ngff:Axis](https://ngff.openmicroscopy.org/0.4/Axis)





```mermaid
 classDiagram
    class Axis
    click Axis href "../Axis/"
      Axis : name
        
      Axis : type
        
      Axis : unit
        
      
```




<!-- no inheritance hierarchy -->

## Class Properties

| Property | Value |
| --- | --- |
| Class URI | [ngff:Axis](https://ngff.openmicroscopy.org/0.4/Axis) |


## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [name](name.md) | 1 <br/> [String](String.md) | NGFF axis or attribute name (e | direct |
| [type](type.md) | 1 <br/> [String](String.md) | NGFF axis type — "space", "time", or "channel" | direct |
| [unit](unit.md) | 0..1 <br/> [String](String.md) | NGFF unit string (e | direct |















## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | ngff:Axis |
| native | zv:Axis |






## LinkML Source

### Direct

<details>
```yaml
name: Axis
description: 'One axis of the spatial index.  Mirrors the OME-Zarr NGFF axis object
  (RFC 4/5).

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slots:
- name
- type
- unit
class_uri: ngff:Axis

```
</details>

### Induced

<details>
```yaml
name: Axis
description: 'One axis of the spatial index.  Mirrors the OME-Zarr NGFF axis object
  (RFC 4/5).

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
attributes:
  name:
    name: name
    description: NGFF axis or attribute name (e.g. "x", "intensity").
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    slot_uri: schema:name
    owner: Axis
    domain_of:
    - Axis
    - AttributeMeta
    - FragmentAttributeMeta
    - ObjectAttributeMeta
    - GroupingsAttributeMeta
    - LinkAttributeFamilyMeta
    - LinkAttributeMeta
    range: string
    required: true
  type:
    name: type
    description: NGFF axis type — "space", "time", or "channel".
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: Axis
    domain_of:
    - Axis
    range: string
    required: true
  unit:
    name: unit
    description: NGFF unit string (e.g. "um", "nanometer", "second").
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: Axis
    domain_of:
    - Axis
    range: string
class_uri: ngff:Axis

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: base_bin_shape 


_Supervoxel bin edge lengths at level 0.  When set, every value must be > 0 and ``chunk_shape`` must be an integer multiple along every axis (enforced runtime-side in ``RootMetadata.validate``)._

__



<div data-search-exclude markdown="1">



URI: [zv:base_bin_shape](https://w3id.org/zarr-vectors/schema/0.5/base_bin_shape)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [RootMetadata](RootMetadata.md) | Root-level `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Float](Float.md) |
| Domain Of | [RootMetadata](RootMetadata.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Multivalued | Yes |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:base_bin_shape |
| native | zv:base_bin_shape |




## LinkML Source

<details>
```yaml
name: base_bin_shape
description: 'Supervoxel bin edge lengths at level 0.  When set, every value must
  be > 0 and ``chunk_shape`` must be an integer multiple along every axis (enforced
  runtime-side in ``RootMetadata.validate``).

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- RootMetadata
range: float
multivalued: true

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: bin_ratio 


_Integer fold-change per axis relative to level 0._



<div data-search-exclude markdown="1">



URI: [zv:bin_ratio](https://w3id.org/zarr-vectors/schema/0.5/bin_ratio)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LevelMetadata](LevelMetadata.md) | Per-resolution-level `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Integer](Integer.md) |
| Domain Of | [LevelMetadata](LevelMetadata.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Multivalued | Yes |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:bin_ratio |
| native | zv:bin_ratio |




## LinkML Source

<details>
```yaml
name: bin_ratio
description: Integer fold-change per axis relative to level 0.
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- LevelMetadata
range: integer
multivalued: true

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: bin_shape 


_Per-axis supervoxel edge lengths at this level.  Must be ``None`` for level 0 (inherits ``base_bin_shape``); must be set for level > 0._

__



<div data-search-exclude markdown="1">



URI: [zv:bin_shape](https://w3id.org/zarr-vectors/schema/0.5/bin_shape)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LevelMetadata](LevelMetadata.md) | Per-resolution-level `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Float](Float.md) |
| Domain Of | [LevelMetadata](LevelMetadata.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Multivalued | Yes |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:bin_shape |
| native | zv:bin_shape |




## LinkML Source

<details>
```yaml
name: bin_shape
description: 'Per-axis supervoxel edge lengths at this level.  Must be ``None`` for
  level 0 (inherits ``base_bin_shape``); must be set for level > 0.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- LevelMetadata
range: float
multivalued: true

```
</details></div>


---

---
search:
  boost: 1.0
---# Type: Boolean 




_A binary (true or false) value_



<div data-search-exclude markdown="1">

URI: [xsd:boolean](http://www.w3.org/2001/XMLSchema#boolean)

## Type Properties

| Property | Value |
| --- | --- |
| Base | `Bool` |
| Type URI | [xsd:boolean](http://www.w3.org/2001/XMLSchema#boolean) |
| Representation | `bool` |






## Notes

* If you are authoring schemas in LinkML YAML, the type is referenced with the lower case "boolean".



## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | xsd:boolean |
| native | zv:boolean |
| exact | schema:Boolean |




</div>


---

---
search:
  boost: 10.0
---

# Class: BoundingBox 


_Two parallel ``ndim``-length arrays representing the global ``(min_corner, max_corner)`` of the store's vertices.  Stored on disk as ``[[min...], [max...]]``; modelled here as a class so cross-field rules can require matching dimensionality._

__



<div data-search-exclude markdown="1">



URI: [zv:BoundingBox](https://w3id.org/zarr-vectors/schema/0.5/BoundingBox)





```mermaid
 classDiagram
    class BoundingBox
    click BoundingBox href "../BoundingBox/"
      BoundingBox : max_corner
        
      BoundingBox : min_corner
        
      
```




<!-- no inheritance hierarchy -->

## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [min_corner](min_corner.md) | 1..* <br/> [Float](Float.md) | Per-axis minima | direct |
| [max_corner](max_corner.md) | 1..* <br/> [Float](Float.md) | Per-axis maxima | direct |





## Usages

| used by | used in | type | used |
| ---  | --- | --- | --- |
| [RootMetadata](RootMetadata.md) | [bounds](bounds.md) | range | [BoundingBox](BoundingBox.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:BoundingBox |
| native | zv:BoundingBox |






## LinkML Source

### Direct

<details>
```yaml
name: BoundingBox
description: 'Two parallel ``ndim``-length arrays representing the global ``(min_corner,
  max_corner)`` of the store''s vertices.  Stored on disk as ``[[min...], [max...]]``;
  modelled here as a class so cross-field rules can require matching dimensionality.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slots:
- min_corner
- max_corner

```
</details>

### Induced

<details>
```yaml
name: BoundingBox
description: 'Two parallel ``ndim``-length arrays representing the global ``(min_corner,
  max_corner)`` of the store''s vertices.  Stored on disk as ``[[min...], [max...]]``;
  modelled here as a class so cross-field rules can require matching dimensionality.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
attributes:
  min_corner:
    name: min_corner
    description: Per-axis minima.  Length must equal ``len(spatial_index_dims)``.
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: BoundingBox
    domain_of:
    - BoundingBox
    range: float
    required: true
    multivalued: true
  max_corner:
    name: max_corner
    description: Per-axis maxima.  Length must equal ``len(spatial_index_dims)``.
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: BoundingBox
    domain_of:
    - BoundingBox
    range: float
    required: true
    multivalued: true

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: bounds 


_Global vertex bounding box._



<div data-search-exclude markdown="1">



URI: [zv:bounds](https://w3id.org/zarr-vectors/schema/0.5/bounds)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [RootMetadata](RootMetadata.md) | Root-level `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [BoundingBox](BoundingBox.md) |
| Domain Of | [RootMetadata](RootMetadata.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:bounds |
| native | zv:bounds |




## LinkML Source

<details>
```yaml
name: bounds
description: Global vertex bounding box.
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- RootMetadata
range: BoundingBox
required: true

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: channel_names 


_For multi-channel per-vertex attributes, the channel labels._



<div data-search-exclude markdown="1">



URI: [zv:channel_names](https://w3id.org/zarr-vectors/schema/0.5/channel_names)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [AttributeMeta](AttributeMeta.md) | `` |  no  |
| [FragmentAttributeMeta](FragmentAttributeMeta.md) | `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [AttributeMeta](AttributeMeta.md), [FragmentAttributeMeta](FragmentAttributeMeta.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Multivalued | Yes |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:channel_names |
| native | zv:channel_names |




## LinkML Source

<details>
```yaml
name: channel_names
description: For multi-channel per-vertex attributes, the channel labels.
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- AttributeMeta
- FragmentAttributeMeta
range: string
multivalued: true

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: chunk_attribute_name 


_Name of the per-vertex attribute used as the leading chunk axis._



<div data-search-exclude markdown="1">



URI: [zv:chunk_attribute_name](https://w3id.org/zarr-vectors/schema/0.5/chunk_attribute_name)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LevelMetadata](LevelMetadata.md) | Per-resolution-level `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [LevelMetadata](LevelMetadata.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:chunk_attribute_name |
| native | zv:chunk_attribute_name |




## LinkML Source

<details>
```yaml
name: chunk_attribute_name
description: Name of the per-vertex attribute used as the leading chunk axis.
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- LevelMetadata
range: string

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: chunk_attribute_values 


_Ordered list mapping attribute-bin index to original attribute value.  Must be non-empty when set, and coherent with ``chunk_attribute_name`` (both set or both absent)._

__



<div data-search-exclude markdown="1">



URI: [zv:chunk_attribute_values](https://w3id.org/zarr-vectors/schema/0.5/chunk_attribute_values)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LevelMetadata](LevelMetadata.md) | Per-resolution-level `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [LevelMetadata](LevelMetadata.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Multivalued | Yes |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:chunk_attribute_values |
| native | zv:chunk_attribute_values |




## LinkML Source

<details>
```yaml
name: chunk_attribute_values
description: 'Ordered list mapping attribute-bin index to original attribute value.  Must
  be non-empty when set, and coherent with ``chunk_attribute_name`` (both set or both
  absent).

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- LevelMetadata
range: string
multivalued: true

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: chunk_dims 


_Chunk-key axis names; the leading axis names appear first.  Set when the level uses attribute chunking (e.g. ``["gene", "z", "y", "x"]``)._

__



<div data-search-exclude markdown="1">



URI: [zv:chunk_dims](https://w3id.org/zarr-vectors/schema/0.5/chunk_dims)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LevelMetadata](LevelMetadata.md) | Per-resolution-level `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [LevelMetadata](LevelMetadata.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Multivalued | Yes |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:chunk_dims |
| native | zv:chunk_dims |




## LinkML Source

<details>
```yaml
name: chunk_dims
description: 'Chunk-key axis names; the leading axis names appear first.  Set when
  the level uses attribute chunking (e.g. ``["gene", "z", "y", "x"]``).

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- LevelMetadata
range: string
multivalued: true

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: chunk_shape 


_Physical spatial chunk size per axis (all values > 0)._



<div data-search-exclude markdown="1">



URI: [zv:chunk_shape](https://w3id.org/zarr-vectors/schema/0.5/chunk_shape)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [RootMetadata](RootMetadata.md) | Root-level `` |  no  |
| [LevelMetadata](LevelMetadata.md) | Per-resolution-level `` |  yes  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Float](Float.md) |
| Domain Of | [RootMetadata](RootMetadata.md), [LevelMetadata](LevelMetadata.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
| Multivalued | Yes |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:chunk_shape |
| native | zv:chunk_shape |




## LinkML Source

<details>
```yaml
name: chunk_shape
description: Physical spatial chunk size per axis (all values > 0).
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- RootMetadata
- LevelMetadata
range: float
required: true
multivalued: true

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: coarsening_method 


_How this level was generated (e.g. "grid_metanode")._



<div data-search-exclude markdown="1">



URI: [zv:coarsening_method](https://w3id.org/zarr-vectors/schema/0.5/coarsening_method)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LevelMetadata](LevelMetadata.md) | Per-resolution-level `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [LevelMetadata](LevelMetadata.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:coarsening_method |
| native | zv:coarsening_method |




## LinkML Source

<details>
```yaml
name: coarsening_method
description: How this level was generated (e.g. "grid_metanode").
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- LevelMetadata
range: string

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: cross_chunk_strategy 

<div data-search-exclude markdown="1">



URI: [zv:cross_chunk_strategy](https://w3id.org/zarr-vectors/schema/0.5/cross_chunk_strategy)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [RootMetadata](RootMetadata.md) | Root-level `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [CrossChunkStrategy](CrossChunkStrategy.md) |
| Domain Of | [RootMetadata](RootMetadata.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:cross_chunk_strategy |
| native | zv:cross_chunk_strategy |




## LinkML Source

<details>
```yaml
name: cross_chunk_strategy
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- RootMetadata
range: CrossChunkStrategy

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: cross_level_depth 


_Maximum absolute level delta for which cross-pyramid-level link arrays are materialized.  ``0`` = none (no ``+N`` or ``-N`` arrays), ``N`` = generate up to ``±N`` (or ``+N`` only when ``cross_level_storage="implicit"``), ``-1`` = all available pyramid levels.  Default ``1``._

__



<div data-search-exclude markdown="1">



URI: [zv:cross_level_depth](https://w3id.org/zarr-vectors/schema/0.5/cross_level_depth)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [RootMetadata](RootMetadata.md) | Root-level `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Integer](Integer.md) |
| Domain Of | [RootMetadata](RootMetadata.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
### Value Constraints

| Property | Value |
| --- | --- |
| Minimum Value | -1 |












## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:cross_level_depth |
| native | zv:cross_level_depth |




## LinkML Source

<details>
```yaml
name: cross_level_depth
description: 'Maximum absolute level delta for which cross-pyramid-level link arrays
  are materialized.  ``0`` = none (no ``+N`` or ``-N`` arrays), ``N`` = generate up
  to ``±N`` (or ``+N`` only when ``cross_level_storage="implicit"``), ``-1`` = all
  available pyramid levels.  Default ``1``.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- RootMetadata
range: integer
minimum_value: -1

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: cross_level_storage 


_Whether cross-level link arrays are written in both directions (``explicit``: ``+N`` at the finer level AND ``-N`` at the coarser level) or only positive deltas (``implicit``: only ``+N``, with ``-N`` reconstructed on read).  Default ``explicit``._

__



<div data-search-exclude markdown="1">



URI: [zv:cross_level_storage](https://w3id.org/zarr-vectors/schema/0.5/cross_level_storage)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [RootMetadata](RootMetadata.md) | Root-level `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [CrossLevelStorage](CrossLevelStorage.md) |
| Domain Of | [RootMetadata](RootMetadata.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:cross_level_storage |
| native | zv:cross_level_storage |




## LinkML Source

<details>
```yaml
name: cross_level_storage
description: 'Whether cross-level link arrays are written in both directions (``explicit``:
  ``+N`` at the finer level AND ``-N`` at the coarser level) or only positive deltas
  (``implicit``: only ``+N``, with ``-N`` reconstructed on read).  Default ``explicit``.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- RootMetadata
range: CrossLevelStorage

```
</details></div>


---

---
search:
  boost: 2.0
---


# Enum: CrossChunkStore 




_Cell-duplication policy for a ``links/<delta>/`` family.  Governs only the boundary-crossing (non-zero offsets) arrays: an intra-chunk record has a single incident chunk, so both values place it identically._

__



<div data-search-exclude markdown="1">

URI: [zv:CrossChunkStore](https://w3id.org/zarr-vectors/schema/0.5/CrossChunkStore)

## Permissible Values
| Value | Meaning | Description |
| --- | --- | --- |
| canonical | None | Each record is filed in exactly one cell — the source chunk of its canonical-... |
| duplicate | None | Each record is filed under one cell per distinct incident chunk as independen... |




## Slots

| Name | Description |
| ---  | --- |
| [store](store.md) | Cell-duplication policy for a links family; see CrossChunkStore |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5






## LinkML Source

<details>
```yaml
name: CrossChunkStore
description: 'Cell-duplication policy for a ``links/<delta>/`` family.  Governs only
  the boundary-crossing (non-zero offsets) arrays: an intra-chunk record has a single
  incident chunk, so both values place it identically.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
permissible_values:
  canonical:
    text: canonical
    description: 'Each record is filed in exactly one cell — the source chunk of its
      canonical-sorted endpoints, or of its input order when ``directed``.  Fewest
      cells.

      '
  duplicate:
    text: duplicate
    description: 'Each record is filed under one cell per distinct incident chunk
      as independent physical copies, so incidence reads become a prefix scan.  ``num_physical_records``
      then exceeds ``num_links``.

      '

```
</details>

</div>


---

---
search:
  boost: 2.0
---


# Enum: CrossChunkStrategy 




_How connectivity that crosses chunk boundaries is represented._



<div data-search-exclude markdown="1">

URI: [zv:CrossChunkStrategy](https://w3id.org/zarr-vectors/schema/0.5/CrossChunkStrategy)

## Permissible Values
| Value | Meaning | Description |
| --- | --- | --- |
| boundary_deduplication | None | Vertices on a chunk boundary are duplicated in each chunk |
| explicit_links | None | Links with a non-zero offsets segment bridge the boundary |
| both | None | Both boundary duplication and explicit links are present |




## Slots

| Name | Description |
| ---  | --- |
| [cross_chunk_strategy](cross_chunk_strategy.md) |  |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5






## LinkML Source

<details>
```yaml
name: CrossChunkStrategy
description: How connectivity that crosses chunk boundaries is represented.
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
permissible_values:
  boundary_deduplication:
    text: boundary_deduplication
    description: Vertices on a chunk boundary are duplicated in each chunk.
  explicit_links:
    text: explicit_links
    description: 'Links with a non-zero offsets segment bridge the boundary. This
      value is semantic — it records the authoring intent, not a distinct physical
      family: since 0.9.0 a boundary-crossing link is an ordinary record in ``links/<delta>/<offsets>/``
      whose offsets are not all zero.

      '
  both:
    text: both
    description: Both boundary duplication and explicit links are present.

```
</details>

</div>


---

---
search:
  boost: 2.0
---


# Enum: CrossLevelStorage 




_How cross-pyramid-level edges are stored in the multiscale links layout (``links/<delta>/``)._

__



<div data-search-exclude markdown="1">

URI: [zv:CrossLevelStorage](https://w3id.org/zarr-vectors/schema/0.5/CrossLevelStorage)

## Permissible Values
| Value | Meaning | Description |
| --- | --- | --- |
| none | None | No cross-level link arrays are emitted (``cross_level_depth=0``) |
| implicit | None | Only positive deltas (``+1``, ``+2``,  |
| explicit | None | Both positive and negative deltas are materialized |




## Slots

| Name | Description |
| ---  | --- |
| [cross_level_storage](cross_level_storage.md) | Whether cross-level link arrays are written in both directions (``explicit``:... |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5






## LinkML Source

<details>
```yaml
name: CrossLevelStorage
description: 'How cross-pyramid-level edges are stored in the multiscale links layout
  (``links/<delta>/``).

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
permissible_values:
  none:
    text: none
    description: No cross-level link arrays are emitted (``cross_level_depth=0``).
  implicit:
    text: implicit
    description: 'Only positive deltas (``+1``, ``+2``, ...) are materialized at the
      finer level.  Negative-delta edges are reconstructed on read by flipping the
      matching positive-delta array at the coarser level.

      '
  explicit:
    text: explicit
    description: 'Both positive and negative deltas are materialized.  Writes ``+N``
      at the finer level and ``-N`` at the coarser level.

      '

```
</details>

</div>


---

---
search:
  boost: 5.0
---

# Slot: crs 


_Optional coordinate reference system metadata (free-form dict matching whatever CRS vocabulary the store uses, e.g. WKT, PROJ4, EPSG, CF conventions)._

__



<div data-search-exclude markdown="1">



URI: [schema:coordinateReferenceSystem](http://schema.org/coordinateReferenceSystem)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [RootMetadata](RootMetadata.md) | Root-level `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [CRS](CRS.md) |
| Domain Of | [RootMetadata](RootMetadata.md) |
| Slot URI | [schema:coordinateReferenceSystem](http://schema.org/coordinateReferenceSystem) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | schema:coordinateReferenceSystem |
| native | zv:crs |




## LinkML Source

<details>
```yaml
name: crs
description: 'Optional coordinate reference system metadata (free-form dict matching
  whatever CRS vocabulary the store uses, e.g. WKT, PROJ4, EPSG, CF conventions).

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slot_uri: schema:coordinateReferenceSystem
domain_of:
- RootMetadata
range: CRS
inlined: true

```
</details></div>


---

---
search:
  boost: 1.0
---# Type: Curie 




_a compact URI_



<div data-search-exclude markdown="1">

URI: [xsd:string](http://www.w3.org/2001/XMLSchema#string)

## Type Properties

| Property | Value |
| --- | --- |
| Base | `Curie` |
| Type URI | [xsd:string](http://www.w3.org/2001/XMLSchema#string) |
| Representation | `str` |






## Comments

* in RDF serializations this MUST be expanded to a URI
* in non-RDF serializations MAY be serialized as the compact representation

## Notes

* If you are authoring schemas in LinkML YAML, the type is referenced with the lower case "curie".



## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | xsd:string |
| native | zv:curie |




</div>


---

---
search:
  boost: 1.0
---# Type: Date 




_a date (year, month and day) in an idealized calendar_



<div data-search-exclude markdown="1">

URI: [xsd:date](http://www.w3.org/2001/XMLSchema#date)

## Type Properties

| Property | Value |
| --- | --- |
| Base | `XSDDate` |
| Type URI | [xsd:date](http://www.w3.org/2001/XMLSchema#date) |
| Representation | `str` |






## Notes

* URI is dateTime because OWL reasoners don't work with straight date or time
* If you are authoring schemas in LinkML YAML, the type is referenced with the lower case "date".



## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | xsd:date |
| native | zv:date |
| exact | schema:Date |




</div>


---

---
search:
  boost: 1.0
---# Type: DateOrDatetime 




_Either a date or a datetime_



<div data-search-exclude markdown="1">

URI: [linkml:DateOrDatetime](https://w3id.org/linkml/DateOrDatetime)

## Type Properties

| Property | Value |
| --- | --- |
| Base | `str` |
| Type URI | [linkml:DateOrDatetime](https://w3id.org/linkml/DateOrDatetime) |
| Representation | `str` |






## Notes

* If you are authoring schemas in LinkML YAML, the type is referenced with the lower case "date_or_datetime".



## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | linkml:DateOrDatetime |
| native | zv:date_or_datetime |




</div>


---

---
search:
  boost: 1.0
---# Type: Datetime 




_The combination of a date and time_



<div data-search-exclude markdown="1">

URI: [xsd:dateTime](http://www.w3.org/2001/XMLSchema#dateTime)

## Type Properties

| Property | Value |
| --- | --- |
| Base | `XSDDateTime` |
| Type URI | [xsd:dateTime](http://www.w3.org/2001/XMLSchema#dateTime) |
| Representation | `str` |






## Notes

* If you are authoring schemas in LinkML YAML, the type is referenced with the lower case "datetime".



## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | xsd:dateTime |
| native | zv:datetime |
| exact | schema:DateTime |




</div>


---

---
search:
  boost: 1.0
---# Type: Decimal 




_A real number with arbitrary precision that conforms to the xsd:decimal specification_



<div data-search-exclude markdown="1">

URI: [xsd:decimal](http://www.w3.org/2001/XMLSchema#decimal)

## Type Properties

| Property | Value |
| --- | --- |
| Base | `Decimal` |
| Type URI | [xsd:decimal](http://www.w3.org/2001/XMLSchema#decimal) |






## Notes

* If you are authoring schemas in LinkML YAML, the type is referenced with the lower case "decimal".



## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | xsd:decimal |
| native | zv:decimal |
| broad | schema:Number |




</div>


---

---
search:
  boost: 5.0
---

# Slot: directed 


_When true, endpoint order is meaningful and preserved (no canonical sort): ``A→B`` and ``B→A`` occupy distinct cells and every stored ``perm_idx`` is 0.  Family-wide; defaults to false (undirected)._

__



<div data-search-exclude markdown="1">



URI: [zv:directed](https://w3id.org/zarr-vectors/schema/0.5/directed)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LinksFamilyMeta](LinksFamilyMeta.md) | `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Boolean](Boolean.md) |
| Domain Of | [LinksFamilyMeta](LinksFamilyMeta.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:directed |
| native | zv:directed |




## LinkML Source

<details>
```yaml
name: directed
description: 'When true, endpoint order is meaningful and preserved (no canonical
  sort): ``A→B`` and ``B→A`` occupy distinct cells and every stored ``perm_idx`` is
  0.  Family-wide; defaults to false (undirected).

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- LinksFamilyMeta
range: boolean
required: false

```
</details></div>


---

---
search:
  boost: 1.0
---# Type: Double 




_A real number that conforms to the xsd:double specification_



<div data-search-exclude markdown="1">

URI: [xsd:double](http://www.w3.org/2001/XMLSchema#double)

## Type Properties

| Property | Value |
| --- | --- |
| Base | `float` |
| Type URI | [xsd:double](http://www.w3.org/2001/XMLSchema#double) |






## Notes

* If you are authoring schemas in LinkML YAML, the type is referenced with the lower case "double".



## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | xsd:double |
| native | zv:double |
| close | schema:Float |




</div>


---

---
search:
  boost: 5.0
---

# Slot: dtype 


_Numpy dtype string of the array's value type (e.g. "float32")._



<div data-search-exclude markdown="1">



URI: [zv:dtype](https://w3id.org/zarr-vectors/schema/0.5/dtype)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [VerticesMeta](VerticesMeta.md) | `` |  no  |
| [LinksMeta](LinksMeta.md) | `` |  no  |
| [AttributeMeta](AttributeMeta.md) | `` |  no  |
| [FragmentAttributeMeta](FragmentAttributeMeta.md) | `` |  no  |
| [ObjectAttributeMeta](ObjectAttributeMeta.md) | `` |  no  |
| [GroupingsAttributeMeta](GroupingsAttributeMeta.md) | `` |  no  |
| [LinkAttributeMeta](LinkAttributeMeta.md) | `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [VerticesMeta](VerticesMeta.md), [LinksMeta](LinksMeta.md), [AttributeMeta](AttributeMeta.md), [FragmentAttributeMeta](FragmentAttributeMeta.md), [ObjectAttributeMeta](ObjectAttributeMeta.md), [GroupingsAttributeMeta](GroupingsAttributeMeta.md), [LinkAttributeMeta](LinkAttributeMeta.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:dtype |
| native | zv:dtype |




## LinkML Source

<details>
```yaml
name: dtype
description: Numpy dtype string of the array's value type (e.g. "float32").
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- VerticesMeta
- LinksMeta
- AttributeMeta
- FragmentAttributeMeta
- ObjectAttributeMeta
- GroupingsAttributeMeta
- LinkAttributeMeta
range: string
required: true

```
</details></div>


---

---
search:
  boost: 2.0
---


# Enum: Encoding 




_Per-array encoding of vertex data._



<div data-search-exclude markdown="1">

URI: [zv:Encoding](https://w3id.org/zarr-vectors/schema/0.5/Encoding)

## Permissible Values
| Value | Meaning | Description |
| --- | --- | --- |
| raw | None | Raw little-endian binary in the dtype's natural layout |
| draco | None | Google Draco mesh / point-cloud compression |




## Slots

| Name | Description |
| ---  | --- |
| [encoding](encoding.md) | How the chunk bytes are encoded |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5






## LinkML Source

<details>
```yaml
name: Encoding
description: Per-array encoding of vertex data.
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
permissible_values:
  raw:
    text: raw
    description: Raw little-endian binary in the dtype's natural layout.
  draco:
    text: draco
    description: Google Draco mesh / point-cloud compression.

```
</details>

</div>


---

---
search:
  boost: 1.0
---# Type: Float 




_A real number that conforms to the xsd:float specification_



<div data-search-exclude markdown="1">

URI: [xsd:float](http://www.w3.org/2001/XMLSchema#float)

## Type Properties

| Property | Value |
| --- | --- |
| Base | `float` |
| Type URI | [xsd:float](http://www.w3.org/2001/XMLSchema#float) |






## Notes

* If you are authoring schemas in LinkML YAML, the type is referenced with the lower case "float".



## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | xsd:float |
| native | zv:float |
| exact | schema:Float |




</div>


---

---
search:
  boost: 5.0
---

# Slot: format_capabilities 


_Optional 0.3+ feature tokens advertised by this store._



<div data-search-exclude markdown="1">



URI: [zv:format_capabilities](https://w3id.org/zarr-vectors/schema/0.5/format_capabilities)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [RootMetadata](RootMetadata.md) | Root-level `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [FormatCapability](FormatCapability.md) |
| Domain Of | [RootMetadata](RootMetadata.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Multivalued | Yes |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:format_capabilities |
| native | zv:format_capabilities |




## LinkML Source

<details>
```yaml
name: format_capabilities
description: Optional 0.3+ feature tokens advertised by this store.
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- RootMetadata
range: FormatCapability
multivalued: true

```
</details></div>


---

---
search:
  boost: 2.0
---


# Enum: FormatCapability 




_Optional feature tokens a store advertises in :attr:`RootMetadata.format_capabilities`.  Open-set: new tokens will be added in future spec revisions; readers must tolerate unknown values._

__



<div data-search-exclude markdown="1">

URI: [zv:FormatCapability](https://w3id.org/zarr-vectors/schema/0.5/FormatCapability)

## Permissible Values
| Value | Meaning | Description |
| --- | --- | --- |
| preserved_object_ids | None | At least one level was written with ID-preserving sparsification (``LevelMeta... |
| shared_fragments | None | At least one level stores per-chunk fragments that may be referenced by multi... |
| fragment_index | None | The store uses the v0 |
| multiscale_links | None | Store uses the multiscale links layout (``links/<delta>/<offsets>/`` and ``li... |




## Slots

| Name | Description |
| ---  | --- |
| [format_capabilities](format_capabilities.md) | Optional 0 |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5






## LinkML Source

<details>
```yaml
name: FormatCapability
description: 'Optional feature tokens a store advertises in :attr:`RootMetadata.format_capabilities`.  Open-set:
  new tokens will be added in future spec revisions; readers must tolerate unknown
  values.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
permissible_values:
  preserved_object_ids:
    text: preserved_object_ids
    description: 'At least one level was written with ID-preserving sparsification
      (``LevelMetadata.preserves_object_ids=True``).  Dropped objects appear as empty
      manifest slots; ``parent_level`` carries semantic weight on those levels.

      '
  shared_fragments:
    text: shared_fragments
    description: 'At least one level stores per-chunk fragments that may be referenced
      by multiple objects'' manifests (the v0.6 successor to ``shared_vertex_groups``;
      the sharing primitive is now a fragment rather than a contiguous-byte vertex
      group).

      '
  fragment_index:
    text: fragment_index
    description: 'The store uses the v0.6 fragment-index encoding for ``vertex_fragments``
      and ``link_fragments`` (a single uint8 blob per chunk with a tagged-bitmap header
      + range table + explicit CSR; see :mod:`zarr_vectors.encoding.fragments`).

      '
  multiscale_links:
    text: multiscale_links
    description: 'Store uses the multiscale links layout (``links/<delta>/<offsets>/``
      and ``link_attributes/<name>/<delta>/<offsets>/``) and may contain cross-pyramid-level
      edges (``delta != 0``).  Since 0.9.0 there is no separate cross-chunk family
      — a cross-chunk link is a link with a non-zero offsets segment — so this token
      now marks only the presence of ``delta != 0`` arrays.

      '

```
</details>

</div>


---

---
search:
  boost: 10.0
---

# Class: FragmentAttributeMeta 


_``.zattrs`` for each ``fragment_attributes/<name>/`` array.  Stored as a dense per-chunk byte blob; each chunk holds a row per fragment, with shape ``(num_fragments_in_chunk,)`` or ``(num_fragments_in_chunk, len(channel_names))``.  Per-chunk fragment counts are derived from ``vertex_fragments/<chunk>`` at read time.  Replace-only at the chunk level._

__



<div data-search-exclude markdown="1">



URI: [zv:FragmentAttributeMeta](https://w3id.org/zarr-vectors/schema/0.5/FragmentAttributeMeta)





```mermaid
 classDiagram
    class FragmentAttributeMeta
    click FragmentAttributeMeta href "../FragmentAttributeMeta/"
      FragmentAttributeMeta : channel_names
        
      FragmentAttributeMeta : dtype
        
      FragmentAttributeMeta : name
        
      FragmentAttributeMeta : zv_array
        
          
    
        
        
        FragmentAttributeMeta --> "1" ZvArrayTag : zv_array
        click ZvArrayTag href "../ZvArrayTag/"
    

        
      
```




<!-- no inheritance hierarchy -->

## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [zv_array](zv_array.md) | 1 <br/> [ZvArrayTag](ZvArrayTag.md) | Discriminator slot identifying the kind of per-array `` | direct |
| [name](name.md) | 1 <br/> [String](String.md) | NGFF axis or attribute name (e | direct |
| [dtype](dtype.md) | 1 <br/> [String](String.md) | Numpy dtype string of the array's value type (e | direct |
| [channel_names](channel_names.md) | * <br/> [String](String.md) | For multi-channel per-vertex attributes, the channel labels | direct |















## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:FragmentAttributeMeta |
| native | zv:FragmentAttributeMeta |






## LinkML Source

### Direct

<details>
```yaml
name: FragmentAttributeMeta
description: '``.zattrs`` for each ``fragment_attributes/<name>/`` array.  Stored
  as a dense per-chunk byte blob; each chunk holds a row per fragment, with shape
  ``(num_fragments_in_chunk,)`` or ``(num_fragments_in_chunk, len(channel_names))``.  Per-chunk
  fragment counts are derived from ``vertex_fragments/<chunk>`` at read time.  Replace-only
  at the chunk level.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slots:
- zv_array
- name
- dtype
- channel_names
slot_usage:
  zv_array:
    name: zv_array
    required: true
    equals_string: fragment_attribute

```
</details>

### Induced

<details>
```yaml
name: FragmentAttributeMeta
description: '``.zattrs`` for each ``fragment_attributes/<name>/`` array.  Stored
  as a dense per-chunk byte blob; each chunk holds a row per fragment, with shape
  ``(num_fragments_in_chunk,)`` or ``(num_fragments_in_chunk, len(channel_names))``.  Per-chunk
  fragment counts are derived from ``vertex_fragments/<chunk>`` at read time.  Replace-only
  at the chunk level.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slot_usage:
  zv_array:
    name: zv_array
    required: true
    equals_string: fragment_attribute
attributes:
  zv_array:
    name: zv_array
    description: 'Discriminator slot identifying the kind of per-array ``.zattrs``
      block.  Each writer in ``core/arrays.py`` stamps the corresponding token from
      :class:`ZvArrayTag`.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: FragmentAttributeMeta
    domain_of:
    - VerticesMeta
    - LinksFamilyMeta
    - LinksMeta
    - AttributeMeta
    - FragmentAttributeMeta
    - ObjectIndexMeta
    - ObjectAttributeMeta
    - GroupingsMeta
    - GroupingsAttributeMeta
    - LinkAttributeFamilyMeta
    - LinkAttributeMeta
    range: ZvArrayTag
    required: true
    equals_string: fragment_attribute
  name:
    name: name
    description: NGFF axis or attribute name (e.g. "x", "intensity").
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    slot_uri: schema:name
    owner: FragmentAttributeMeta
    domain_of:
    - Axis
    - AttributeMeta
    - FragmentAttributeMeta
    - ObjectAttributeMeta
    - GroupingsAttributeMeta
    - LinkAttributeFamilyMeta
    - LinkAttributeMeta
    range: string
    required: true
  dtype:
    name: dtype
    description: Numpy dtype string of the array's value type (e.g. "float32").
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: FragmentAttributeMeta
    domain_of:
    - VerticesMeta
    - LinksMeta
    - AttributeMeta
    - FragmentAttributeMeta
    - ObjectAttributeMeta
    - GroupingsAttributeMeta
    - LinkAttributeMeta
    range: string
    required: true
  channel_names:
    name: channel_names
    description: For multi-channel per-vertex attributes, the channel labels.
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: FragmentAttributeMeta
    domain_of:
    - AttributeMeta
    - FragmentAttributeMeta
    range: string
    multivalued: true

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: geometry_types 


_One or more geometry kinds present in the store._



<div data-search-exclude markdown="1">



URI: [zv:geometry_types](https://w3id.org/zarr-vectors/schema/0.5/geometry_types)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [RootMetadata](RootMetadata.md) | Root-level `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [GeometryType](GeometryType.md) |
| Domain Of | [RootMetadata](RootMetadata.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
| Multivalued | Yes |
| Minimum Cardinality | 1 |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:geometry_types |
| native | zv:geometry_types |




## LinkML Source

<details>
```yaml
name: geometry_types
description: One or more geometry kinds present in the store.
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- RootMetadata
range: GeometryType
required: true
multivalued: true
minimum_cardinality: 1

```
</details></div>


---

---
search:
  boost: 2.0
---


# Enum: GeometryType 




_The kind of geometry a store (or one of its sub-types) holds._



<div data-search-exclude markdown="1">

URI: [zv:GeometryType](https://w3id.org/zarr-vectors/schema/0.5/GeometryType)

## Permissible Values
| Value | Meaning | Description |
| --- | --- | --- |
| point_cloud | None |  |
| line | None |  |
| polyline | None |  |
| streamline | None |  |
| skeleton | None |  |
| graph | None |  |
| mesh | None |  |




## Slots

| Name | Description |
| ---  | --- |
| [geometry_types](geometry_types.md) | One or more geometry kinds present in the store |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5






## LinkML Source

<details>
```yaml
name: GeometryType
description: The kind of geometry a store (or one of its sub-types) holds.
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
permissible_values:
  point_cloud:
    text: point_cloud
  line:
    text: line
  polyline:
    text: polyline
  streamline:
    text: streamline
  skeleton:
    text: skeleton
  graph:
    text: graph
  mesh:
    text: mesh

```
</details>

</div>


---

---
search:
  boost: 10.0
---

# Class: GroupingsAttributeMeta 


_``.zattrs`` for each ``groupings_attributes/<name>/`` array._



<div data-search-exclude markdown="1">



URI: [zv:GroupingsAttributeMeta](https://w3id.org/zarr-vectors/schema/0.5/GroupingsAttributeMeta)





```mermaid
 classDiagram
    class GroupingsAttributeMeta
    click GroupingsAttributeMeta href "../GroupingsAttributeMeta/"
      GroupingsAttributeMeta : dtype
        
      GroupingsAttributeMeta : name
        
      GroupingsAttributeMeta : shape
        
      GroupingsAttributeMeta : zv_array
        
          
    
        
        
        GroupingsAttributeMeta --> "1" ZvArrayTag : zv_array
        click ZvArrayTag href "../ZvArrayTag/"
    

        
      
```




<!-- no inheritance hierarchy -->

## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [zv_array](zv_array.md) | 1 <br/> [ZvArrayTag](ZvArrayTag.md) | Discriminator slot identifying the kind of per-array `` | direct |
| [name](name.md) | 1 <br/> [String](String.md) | NGFF axis or attribute name (e | direct |
| [dtype](dtype.md) | 1 <br/> [String](String.md) | Numpy dtype string of the array's value type (e | direct |
| [shape](shape.md) | 1..* <br/> [Integer](Integer.md) | Shape of a dense per-object/per-group array | direct |















## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:GroupingsAttributeMeta |
| native | zv:GroupingsAttributeMeta |






## LinkML Source

### Direct

<details>
```yaml
name: GroupingsAttributeMeta
description: '``.zattrs`` for each ``groupings_attributes/<name>/`` array.'
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slots:
- zv_array
- name
- dtype
- shape
slot_usage:
  zv_array:
    name: zv_array
    required: true
    equals_string: groupings_attribute

```
</details>

### Induced

<details>
```yaml
name: GroupingsAttributeMeta
description: '``.zattrs`` for each ``groupings_attributes/<name>/`` array.'
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slot_usage:
  zv_array:
    name: zv_array
    required: true
    equals_string: groupings_attribute
attributes:
  zv_array:
    name: zv_array
    description: 'Discriminator slot identifying the kind of per-array ``.zattrs``
      block.  Each writer in ``core/arrays.py`` stamps the corresponding token from
      :class:`ZvArrayTag`.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: GroupingsAttributeMeta
    domain_of:
    - VerticesMeta
    - LinksFamilyMeta
    - LinksMeta
    - AttributeMeta
    - FragmentAttributeMeta
    - ObjectIndexMeta
    - ObjectAttributeMeta
    - GroupingsMeta
    - GroupingsAttributeMeta
    - LinkAttributeFamilyMeta
    - LinkAttributeMeta
    range: ZvArrayTag
    required: true
    equals_string: groupings_attribute
  name:
    name: name
    description: NGFF axis or attribute name (e.g. "x", "intensity").
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    slot_uri: schema:name
    owner: GroupingsAttributeMeta
    domain_of:
    - Axis
    - AttributeMeta
    - FragmentAttributeMeta
    - ObjectAttributeMeta
    - GroupingsAttributeMeta
    - LinkAttributeFamilyMeta
    - LinkAttributeMeta
    range: string
    required: true
  dtype:
    name: dtype
    description: Numpy dtype string of the array's value type (e.g. "float32").
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: GroupingsAttributeMeta
    domain_of:
    - VerticesMeta
    - LinksMeta
    - AttributeMeta
    - FragmentAttributeMeta
    - ObjectAttributeMeta
    - GroupingsAttributeMeta
    - LinkAttributeMeta
    range: string
    required: true
  shape:
    name: shape
    description: Shape of a dense per-object/per-group array.
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: GroupingsAttributeMeta
    domain_of:
    - ObjectAttributeMeta
    - GroupingsAttributeMeta
    range: integer
    required: true
    multivalued: true

```
</details></div>


---

---
search:
  boost: 10.0
---

# Class: GroupingsMeta 


_``.zattrs`` for ``groupings/``._



<div data-search-exclude markdown="1">



URI: [zv:GroupingsMeta](https://w3id.org/zarr-vectors/schema/0.5/GroupingsMeta)





```mermaid
 classDiagram
    class GroupingsMeta
    click GroupingsMeta href "../GroupingsMeta/"
      GroupingsMeta : num_groups
        
      GroupingsMeta : zv_array
        
          
    
        
        
        GroupingsMeta --> "1" ZvArrayTag : zv_array
        click ZvArrayTag href "../ZvArrayTag/"
    

        
      
```




<!-- no inheritance hierarchy -->

## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [zv_array](zv_array.md) | 1 <br/> [ZvArrayTag](ZvArrayTag.md) | Discriminator slot identifying the kind of per-array `` | direct |
| [num_groups](num_groups.md) | 1 <br/> [Integer](Integer.md) | Total grouping count | direct |















## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:GroupingsMeta |
| native | zv:GroupingsMeta |






## LinkML Source

### Direct

<details>
```yaml
name: GroupingsMeta
description: '``.zattrs`` for ``groupings/``.'
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slots:
- zv_array
- num_groups
slot_usage:
  zv_array:
    name: zv_array
    required: true
    equals_string: groupings

```
</details>

### Induced

<details>
```yaml
name: GroupingsMeta
description: '``.zattrs`` for ``groupings/``.'
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slot_usage:
  zv_array:
    name: zv_array
    required: true
    equals_string: groupings
attributes:
  zv_array:
    name: zv_array
    description: 'Discriminator slot identifying the kind of per-array ``.zattrs``
      block.  Each writer in ``core/arrays.py`` stamps the corresponding token from
      :class:`ZvArrayTag`.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: GroupingsMeta
    domain_of:
    - VerticesMeta
    - LinksFamilyMeta
    - LinksMeta
    - AttributeMeta
    - FragmentAttributeMeta
    - ObjectIndexMeta
    - ObjectAttributeMeta
    - GroupingsMeta
    - GroupingsAttributeMeta
    - LinkAttributeFamilyMeta
    - LinkAttributeMeta
    range: ZvArrayTag
    required: true
    equals_string: groupings
  num_groups:
    name: num_groups
    description: Total grouping count.
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: GroupingsMeta
    domain_of:
    - GroupingsMeta
    range: integer
    required: true
    minimum_value: 0

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: has_perm 


_Whether each row of this links array carries a leading ``perm_idx`` column — i.e. whether rows are ``1 + link_width`` ints rather than ``link_width``.  True exactly when a non-identity endpoint placement is possible: non-intra ``offsets`` AND ``level_delta == 0`` AND (``store="duplicate"`` OR not ``directed``).  Readers must take the record width from this stamp rather than inferring it: decoding physical bytes at the logical width does not reliably raise, it silently fabricates rows._

__



<div data-search-exclude markdown="1">



URI: [zv:has_perm](https://w3id.org/zarr-vectors/schema/0.5/has_perm)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LinksMeta](LinksMeta.md) | `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Boolean](Boolean.md) |
| Domain Of | [LinksMeta](LinksMeta.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:has_perm |
| native | zv:has_perm |




## LinkML Source

<details>
```yaml
name: has_perm
description: 'Whether each row of this links array carries a leading ``perm_idx``
  column — i.e. whether rows are ``1 + link_width`` ints rather than ``link_width``.  True
  exactly when a non-identity endpoint placement is possible: non-intra ``offsets``
  AND ``level_delta == 0`` AND (``store="duplicate"`` OR not ``directed``).  Readers
  must take the record width from this stamp rather than inferring it: decoding physical
  bytes at the logical width does not reliably raise, it silently fabricates rows.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- LinksMeta
range: boolean
required: true

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: inherited_num_objects 


_OID-space size inherited from the parent level (= ``parent_level.num_objects``).  Required when ``preserves_object_ids`` is true; absent on standalone levels._

__



<div data-search-exclude markdown="1">



URI: [zv:inherited_num_objects](https://w3id.org/zarr-vectors/schema/0.5/inherited_num_objects)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LevelMetadata](LevelMetadata.md) | Per-resolution-level `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Integer](Integer.md) |
| Domain Of | [LevelMetadata](LevelMetadata.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
### Value Constraints

| Property | Value |
| --- | --- |
| Minimum Value | 0 |












## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:inherited_num_objects |
| native | zv:inherited_num_objects |




## LinkML Source

<details>
```yaml
name: inherited_num_objects
description: 'OID-space size inherited from the parent level (= ``parent_level.num_objects``).  Required
  when ``preserves_object_ids`` is true; absent on standalone levels.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- LevelMetadata
range: integer
minimum_value: 0

```
</details></div>


---

---
search:
  boost: 1.0
---# Type: Integer 




_An integer_



<div data-search-exclude markdown="1">

URI: [xsd:integer](http://www.w3.org/2001/XMLSchema#integer)

## Type Properties

| Property | Value |
| --- | --- |
| Base | `int` |
| Type URI | [xsd:integer](http://www.w3.org/2001/XMLSchema#integer) |






## Notes

* If you are authoring schemas in LinkML YAML, the type is referenced with the lower case "integer".



## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | xsd:integer |
| native | zv:integer |
| exact | schema:Integer |




</div>


---

---
search:
  boost: 1.0
---# Type: Jsonpath 




_A string encoding a JSON Path. The value of the string MUST conform to JSON Point syntax and SHOULD dereference to zero or more valid objects within the current instance document when encoded in tree form._



<div data-search-exclude markdown="1">

URI: [xsd:string](http://www.w3.org/2001/XMLSchema#string)

## Type Properties

| Property | Value |
| --- | --- |
| Base | `str` |
| Type URI | [xsd:string](http://www.w3.org/2001/XMLSchema#string) |
| Representation | `str` |






## Notes

* If you are authoring schemas in LinkML YAML, the type is referenced with the lower case "jsonpath".



## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | xsd:string |
| native | zv:jsonpath |




</div>


---

---
search:
  boost: 1.0
---# Type: Jsonpointer 




_A string encoding a JSON Pointer. The value of the string MUST conform to JSON Point syntax and SHOULD dereference to a valid object within the current instance document when encoded in tree form._



<div data-search-exclude markdown="1">

URI: [xsd:string](http://www.w3.org/2001/XMLSchema#string)

## Type Properties

| Property | Value |
| --- | --- |
| Base | `str` |
| Type URI | [xsd:string](http://www.w3.org/2001/XMLSchema#string) |
| Representation | `str` |






## Notes

* If you are authoring schemas in LinkML YAML, the type is referenced with the lower case "jsonpointer".



## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | xsd:string |
| native | zv:jsonpointer |




</div>


---

---
search:
  boost: 5.0
---

# Slot: level 


_Resolution level index (0 = full resolution)._



<div data-search-exclude markdown="1">



URI: [zv:level](https://w3id.org/zarr-vectors/schema/0.5/level)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LevelMetadata](LevelMetadata.md) | Per-resolution-level `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Integer](Integer.md) |
| Domain Of | [LevelMetadata](LevelMetadata.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
### Value Constraints

| Property | Value |
| --- | --- |
| Minimum Value | 0 |












## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:level |
| native | zv:level |




## LinkML Source

<details>
```yaml
name: level
description: Resolution level index (0 = full resolution).
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- LevelMetadata
range: integer
required: true
minimum_value: 0

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: level_delta 


_Pyramid-level delta between the source side (the level that owns this array) and the target side of the edges.  ``0`` for intra-level arrays (the only kind written pre-0.4), ``+N`` for edges from this level to ``this_level + N`` (coarser), ``-N`` for edges to ``this_level - N`` (finer)._

__



<div data-search-exclude markdown="1">



URI: [zv:level_delta](https://w3id.org/zarr-vectors/schema/0.5/level_delta)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LinksFamilyMeta](LinksFamilyMeta.md) | `` |  no  |
| [LinksMeta](LinksMeta.md) | `` |  no  |
| [LinkAttributeFamilyMeta](LinkAttributeFamilyMeta.md) | `` |  no  |
| [LinkAttributeMeta](LinkAttributeMeta.md) | `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Integer](Integer.md) |
| Domain Of | [LinksFamilyMeta](LinksFamilyMeta.md), [LinksMeta](LinksMeta.md), [LinkAttributeFamilyMeta](LinkAttributeFamilyMeta.md), [LinkAttributeMeta](LinkAttributeMeta.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:level_delta |
| native | zv:level_delta |




## LinkML Source

<details>
```yaml
name: level_delta
description: 'Pyramid-level delta between the source side (the level that owns this
  array) and the target side of the edges.  ``0`` for intra-level arrays (the only
  kind written pre-0.4), ``+N`` for edges from this level to ``this_level + N`` (coarser),
  ``-N`` for edges to ``this_level - N`` (finer).

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- LinksFamilyMeta
- LinksMeta
- LinkAttributeFamilyMeta
- LinkAttributeMeta
range: integer
required: true

```
</details></div>


---

---
search:
  boost: 10.0
---

# Class: LevelMetadata 


_Per-resolution-level ``.zattrs`` payload, persisted under the key ``zarr_vectors_level``.  Validates the runtime :class:`zarr_vectors.core.metadata.LevelMetadata` dataclass._

__



<div data-search-exclude markdown="1">



URI: [zv:LevelMetadata](https://w3id.org/zarr-vectors/schema/0.5/LevelMetadata)





```mermaid
 classDiagram
    class LevelMetadata
    click LevelMetadata href "../LevelMetadata/"
      LevelMetadata : arrays_present
        
      LevelMetadata : bin_ratio
        
      LevelMetadata : bin_shape
        
      LevelMetadata : chunk_attribute_name
        
      LevelMetadata : chunk_attribute_values
        
      LevelMetadata : chunk_dims
        
      LevelMetadata : chunk_shape
        
      LevelMetadata : coarsening_method
        
      LevelMetadata : inherited_num_objects
        
      LevelMetadata : level
        
      LevelMetadata : object_sparsity
        
      LevelMetadata : parent_level
        
      LevelMetadata : preserves_object_ids
        
      LevelMetadata : shared_fragments
        
      LevelMetadata : vertex_count
        
      
```




<!-- no inheritance hierarchy -->

## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [level](level.md) | 1 <br/> [Integer](Integer.md) | Resolution level index (0 = full resolution) | direct |
| [vertex_count](vertex_count.md) | 1 <br/> [Integer](Integer.md) | Total number of vertices at this level | direct |
| [arrays_present](arrays_present.md) | 1..* <br/> [String](String.md) | Names of arrays present in the level group | direct |
| [bin_shape](bin_shape.md) | * <br/> [Float](Float.md) | Per-axis supervoxel edge lengths at this level | direct |
| [bin_ratio](bin_ratio.md) | * <br/> [Integer](Integer.md) | Integer fold-change per axis relative to level 0 | direct |
| [chunk_shape](chunk_shape.md) | * <br/> [Float](Float.md) | Per-level physical chunk size override | direct |
| [object_sparsity](object_sparsity.md) | 0..1 <br/> [Float](Float.md) | Fraction of objects retained at this level | direct |
| [coarsening_method](coarsening_method.md) | 0..1 <br/> [String](String.md) | How this level was generated (e | direct |
| [parent_level](parent_level.md) | 0..1 <br/> [Integer](Integer.md) | Source level index (None for level 0) | direct |
| [chunk_dims](chunk_dims.md) | * <br/> [String](String.md) | Chunk-key axis names; the leading axis names appear first | direct |
| [chunk_attribute_name](chunk_attribute_name.md) | 0..1 <br/> [String](String.md) | Name of the per-vertex attribute used as the leading chunk axis | direct |
| [chunk_attribute_values](chunk_attribute_values.md) | * <br/> [String](String.md) | Ordered list mapping attribute-bin index to original attribute value | direct |
| [preserves_object_ids](preserves_object_ids.md) | 0..1 <br/> [Boolean](Boolean.md) | True for levels written by the per-object pyramid regime | direct |
| [inherited_num_objects](inherited_num_objects.md) | 0..1 <br/> [Integer](Integer.md) | OID-space size inherited from the parent level (= ``parent_level | direct |
| [shared_fragments](shared_fragments.md) | 0..1 <br/> [Boolean](Boolean.md) | True when per-chunk fragments may be referenced by multiple objects' manifest... | direct |















## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:LevelMetadata |
| native | zv:LevelMetadata |






## LinkML Source

### Direct

<details>
```yaml
name: LevelMetadata
description: 'Per-resolution-level ``.zattrs`` payload, persisted under the key ``zarr_vectors_level``.  Validates
  the runtime :class:`zarr_vectors.core.metadata.LevelMetadata` dataclass.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slots:
- level
- vertex_count
- arrays_present
- bin_shape
- bin_ratio
- chunk_shape
- object_sparsity
- coarsening_method
- parent_level
- chunk_dims
- chunk_attribute_name
- chunk_attribute_values
- preserves_object_ids
- inherited_num_objects
- shared_fragments
slot_usage:
  chunk_shape:
    name: chunk_shape
    description: 'Per-level physical chunk size override.  When omitted, the level
      inherits ``RootMetadata.chunk_shape``.  When present, each axis must be a positive
      integer multiple of the root chunk_shape (nested chunk grids).

      '
    required: false

```
</details>

### Induced

<details>
```yaml
name: LevelMetadata
description: 'Per-resolution-level ``.zattrs`` payload, persisted under the key ``zarr_vectors_level``.  Validates
  the runtime :class:`zarr_vectors.core.metadata.LevelMetadata` dataclass.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slot_usage:
  chunk_shape:
    name: chunk_shape
    description: 'Per-level physical chunk size override.  When omitted, the level
      inherits ``RootMetadata.chunk_shape``.  When present, each axis must be a positive
      integer multiple of the root chunk_shape (nested chunk grids).

      '
    required: false
attributes:
  level:
    name: level
    description: Resolution level index (0 = full resolution).
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LevelMetadata
    domain_of:
    - LevelMetadata
    range: integer
    required: true
    minimum_value: 0
  vertex_count:
    name: vertex_count
    description: Total number of vertices at this level.
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LevelMetadata
    domain_of:
    - LevelMetadata
    range: integer
    required: true
    minimum_value: 0
  arrays_present:
    name: arrays_present
    description: Names of arrays present in the level group.
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LevelMetadata
    domain_of:
    - LevelMetadata
    range: string
    required: true
    multivalued: true
  bin_shape:
    name: bin_shape
    description: 'Per-axis supervoxel edge lengths at this level.  Must be ``None``
      for level 0 (inherits ``base_bin_shape``); must be set for level > 0.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LevelMetadata
    domain_of:
    - LevelMetadata
    range: float
    multivalued: true
  bin_ratio:
    name: bin_ratio
    description: Integer fold-change per axis relative to level 0.
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LevelMetadata
    domain_of:
    - LevelMetadata
    range: integer
    multivalued: true
  chunk_shape:
    name: chunk_shape
    description: 'Per-level physical chunk size override.  When omitted, the level
      inherits ``RootMetadata.chunk_shape``.  When present, each axis must be a positive
      integer multiple of the root chunk_shape (nested chunk grids).

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LevelMetadata
    domain_of:
    - RootMetadata
    - LevelMetadata
    range: float
    required: false
    multivalued: true
  object_sparsity:
    name: object_sparsity
    description: Fraction of objects retained at this level.
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LevelMetadata
    domain_of:
    - LevelMetadata
    range: float
    minimum_value: 0.0
    maximum_value: 1.0
  coarsening_method:
    name: coarsening_method
    description: How this level was generated (e.g. "grid_metanode").
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LevelMetadata
    domain_of:
    - LevelMetadata
    range: string
  parent_level:
    name: parent_level
    description: Source level index (None for level 0).
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LevelMetadata
    domain_of:
    - LevelMetadata
    range: integer
    minimum_value: 0
  chunk_dims:
    name: chunk_dims
    description: 'Chunk-key axis names; the leading axis names appear first.  Set
      when the level uses attribute chunking (e.g. ``["gene", "z", "y", "x"]``).

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LevelMetadata
    domain_of:
    - LevelMetadata
    range: string
    multivalued: true
  chunk_attribute_name:
    name: chunk_attribute_name
    description: Name of the per-vertex attribute used as the leading chunk axis.
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LevelMetadata
    domain_of:
    - LevelMetadata
    range: string
  chunk_attribute_values:
    name: chunk_attribute_values
    description: 'Ordered list mapping attribute-bin index to original attribute value.  Must
      be non-empty when set, and coherent with ``chunk_attribute_name`` (both set
      or both absent).

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LevelMetadata
    domain_of:
    - LevelMetadata
    range: string
    multivalued: true
  preserves_object_ids:
    name: preserves_object_ids
    description: 'True for levels written by the per-object pyramid regime.  When
      set, ``num_objects`` and ``object_attributes`` row count inherit from the parent
      level''s OID space; dropped objects leave empty manifest slots and zero ``present_mask``
      bytes.  ``parent_level`` is load-bearing under this flag.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LevelMetadata
    domain_of:
    - LevelMetadata
    range: boolean
  inherited_num_objects:
    name: inherited_num_objects
    description: 'OID-space size inherited from the parent level (= ``parent_level.num_objects``).  Required
      when ``preserves_object_ids`` is true; absent on standalone levels.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LevelMetadata
    domain_of:
    - LevelMetadata
    range: integer
    minimum_value: 0
  shared_fragments:
    name: shared_fragments
    description: 'True when per-chunk fragments may be referenced by multiple objects''
      manifests (shared metavertices in the per-object pyramid regime).  v0.6 successor
      to ``shared_vertex_groups``.  Readers MAY use this to short-circuit dedup work.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LevelMetadata
    domain_of:
    - LevelMetadata
    range: boolean

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: link_width 


_Width of a links row (1 for parent→child metanode references, 2 for edges, 3 for triangle faces, 4 for quads)._

__



<div data-search-exclude markdown="1">



URI: [zv:link_width](https://w3id.org/zarr-vectors/schema/0.5/link_width)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LinksFamilyMeta](LinksFamilyMeta.md) | `` |  no  |
| [LinksMeta](LinksMeta.md) | `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Integer](Integer.md) |
| Domain Of | [LinksFamilyMeta](LinksFamilyMeta.md), [LinksMeta](LinksMeta.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
### Value Constraints

| Property | Value |
| --- | --- |
| Minimum Value | 1 |












## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:link_width |
| native | zv:link_width |




## LinkML Source

<details>
```yaml
name: link_width
description: 'Width of a links row (1 for parent→child metanode references, 2 for
  edges, 3 for triangle faces, 4 for quads).

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- LinksFamilyMeta
- LinksMeta
range: integer
required: true
minimum_value: 1

```
</details></div>


---

---
search:
  boost: 10.0
---

# Class: LinkAttributeFamilyMeta 


_``.zattrs`` for a ``link_attributes/<name>/<delta>/`` **group**. There is no separate cross-chunk attribute family: an intra-chunk link's attributes live under the all-zero offsets segment, exactly like the links themselves._

_``num_links`` is stamped by the whole-family writer and MUST then equal the parallel ``links/<delta>/`` family's ``num_links``.  It is absent when the family was created without data (the per-cell writers do not stamp it)._

__



<div data-search-exclude markdown="1">



URI: [zv:LinkAttributeFamilyMeta](https://w3id.org/zarr-vectors/schema/0.5/LinkAttributeFamilyMeta)





```mermaid
 classDiagram
    class LinkAttributeFamilyMeta
    click LinkAttributeFamilyMeta href "../LinkAttributeFamilyMeta/"
      LinkAttributeFamilyMeta : level_delta
        
      LinkAttributeFamilyMeta : name
        
      LinkAttributeFamilyMeta : num_links
        
      LinkAttributeFamilyMeta : zv_array
        
          
    
        
        
        LinkAttributeFamilyMeta --> "1" ZvArrayTag : zv_array
        click ZvArrayTag href "../ZvArrayTag/"
    

        
      
```




<!-- no inheritance hierarchy -->

## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [zv_array](zv_array.md) | 1 <br/> [ZvArrayTag](ZvArrayTag.md) | Discriminator slot identifying the kind of per-array `` | direct |
| [name](name.md) | 1 <br/> [String](String.md) | NGFF axis or attribute name (e | direct |
| [level_delta](level_delta.md) | 1 <br/> [Integer](Integer.md) | Pyramid-level delta between the source side (the level that owns this array) ... | direct |
| [num_links](num_links.md) | 0..1 <br/> [Integer](Integer.md) | Family-wide *logical* link count — one per input record, however many physica... | direct |















## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:LinkAttributeFamilyMeta |
| native | zv:LinkAttributeFamilyMeta |






## LinkML Source

### Direct

<details>
```yaml
name: LinkAttributeFamilyMeta
description: '``.zattrs`` for a ``link_attributes/<name>/<delta>/`` **group**. There
  is no separate cross-chunk attribute family: an intra-chunk link''s attributes live
  under the all-zero offsets segment, exactly like the links themselves.

  ``num_links`` is stamped by the whole-family writer and MUST then equal the parallel
  ``links/<delta>/`` family''s ``num_links``.  It is absent when the family was created
  without data (the per-cell writers do not stamp it).

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slots:
- zv_array
- name
- level_delta
- num_links
slot_usage:
  zv_array:
    name: zv_array
    required: true
    equals_string: link_attribute_family

```
</details>

### Induced

<details>
```yaml
name: LinkAttributeFamilyMeta
description: '``.zattrs`` for a ``link_attributes/<name>/<delta>/`` **group**. There
  is no separate cross-chunk attribute family: an intra-chunk link''s attributes live
  under the all-zero offsets segment, exactly like the links themselves.

  ``num_links`` is stamped by the whole-family writer and MUST then equal the parallel
  ``links/<delta>/`` family''s ``num_links``.  It is absent when the family was created
  without data (the per-cell writers do not stamp it).

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slot_usage:
  zv_array:
    name: zv_array
    required: true
    equals_string: link_attribute_family
attributes:
  zv_array:
    name: zv_array
    description: 'Discriminator slot identifying the kind of per-array ``.zattrs``
      block.  Each writer in ``core/arrays.py`` stamps the corresponding token from
      :class:`ZvArrayTag`.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LinkAttributeFamilyMeta
    domain_of:
    - VerticesMeta
    - LinksFamilyMeta
    - LinksMeta
    - AttributeMeta
    - FragmentAttributeMeta
    - ObjectIndexMeta
    - ObjectAttributeMeta
    - GroupingsMeta
    - GroupingsAttributeMeta
    - LinkAttributeFamilyMeta
    - LinkAttributeMeta
    range: ZvArrayTag
    required: true
    equals_string: link_attribute_family
  name:
    name: name
    description: NGFF axis or attribute name (e.g. "x", "intensity").
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    slot_uri: schema:name
    owner: LinkAttributeFamilyMeta
    domain_of:
    - Axis
    - AttributeMeta
    - FragmentAttributeMeta
    - ObjectAttributeMeta
    - GroupingsAttributeMeta
    - LinkAttributeFamilyMeta
    - LinkAttributeMeta
    range: string
    required: true
  level_delta:
    name: level_delta
    description: 'Pyramid-level delta between the source side (the level that owns
      this array) and the target side of the edges.  ``0`` for intra-level arrays
      (the only kind written pre-0.4), ``+N`` for edges from this level to ``this_level
      + N`` (coarser), ``-N`` for edges to ``this_level - N`` (finer).

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LinkAttributeFamilyMeta
    domain_of:
    - LinksFamilyMeta
    - LinksMeta
    - LinkAttributeFamilyMeta
    - LinkAttributeMeta
    range: integer
    required: true
  num_links:
    name: num_links
    description: 'Family-wide *logical* link count — one per input record, however
      many physical copies ``store`` filed.  Absent on a ``links/<delta>/`` group
      until ``finalize_links`` has run, and on a ``link_attributes/<name>/<delta>/``
      group created without data.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LinkAttributeFamilyMeta
    domain_of:
    - LinksFamilyMeta
    - LinkAttributeFamilyMeta
    range: integer
    required: false
    minimum_value: 0

```
</details></div>


---

---
search:
  boost: 10.0
---

# Class: LinkAttributeMeta 


_``.zattrs`` for one ``link_attributes/<name>/<delta>/<offsets>/`` array.  Mirrors the ``links/<delta>/<offsets>/`` array of the same ``<delta>`` and ``<offsets>`` cell-for-cell — same cells, same per-cell row order — so attribute rows align 1:1 with link records without storing a row id._

_Rows never carry the links array's ``perm_idx`` column: that column describes a record's placement, not its value, so ``has_perm`` has no analogue here and a row is exactly ``row_shape`` worth of ``dtype``.  A ``store="duplicate"`` family replicates an attribute value alongside each physical copy of its record, so cells stay row-aligned with the links array they mirror._

__



<div data-search-exclude markdown="1">



URI: [zv:LinkAttributeMeta](https://w3id.org/zarr-vectors/schema/0.5/LinkAttributeMeta)





```mermaid
 classDiagram
    class LinkAttributeMeta
    click LinkAttributeMeta href "../LinkAttributeMeta/"
      LinkAttributeMeta : dtype
        
      LinkAttributeMeta : level_delta
        
      LinkAttributeMeta : name
        
      LinkAttributeMeta : offsets
        
      LinkAttributeMeta : row_shape
        
      LinkAttributeMeta : zv_array
        
          
    
        
        
        LinkAttributeMeta --> "1" ZvArrayTag : zv_array
        click ZvArrayTag href "../ZvArrayTag/"
    

        
      
```




<!-- no inheritance hierarchy -->

## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [zv_array](zv_array.md) | 1 <br/> [ZvArrayTag](ZvArrayTag.md) | Discriminator slot identifying the kind of per-array `` | direct |
| [name](name.md) | 1 <br/> [String](String.md) | NGFF axis or attribute name (e | direct |
| [dtype](dtype.md) | 1 <br/> [String](String.md) | Numpy dtype string of the array's value type (e | direct |
| [offsets](offsets.md) | 1 <br/> [Integer](Integer.md) | The ``link_width - 1`` relative offsets naming this array, each of ``sid_ndim... | direct |
| [row_shape](row_shape.md) | * <br/> [Integer](Integer.md) | Tail dimensions of one attribute row (``[]`` for a 1-D per-link attribute), l... | direct |
| [level_delta](level_delta.md) | 1 <br/> [Integer](Integer.md) | Pyramid-level delta between the source side (the level that owns this array) ... | direct |















## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:LinkAttributeMeta |
| native | zv:LinkAttributeMeta |






## LinkML Source

### Direct

<details>
```yaml
name: LinkAttributeMeta
description: '``.zattrs`` for one ``link_attributes/<name>/<delta>/<offsets>/`` array.  Mirrors
  the ``links/<delta>/<offsets>/`` array of the same ``<delta>`` and ``<offsets>``
  cell-for-cell — same cells, same per-cell row order — so attribute rows align 1:1
  with link records without storing a row id.

  Rows never carry the links array''s ``perm_idx`` column: that column describes a
  record''s placement, not its value, so ``has_perm`` has no analogue here and a row
  is exactly ``row_shape`` worth of ``dtype``.  A ``store="duplicate"`` family replicates
  an attribute value alongside each physical copy of its record, so cells stay row-aligned
  with the links array they mirror.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slots:
- zv_array
- name
- dtype
- offsets
- row_shape
- level_delta
slot_usage:
  zv_array:
    name: zv_array
    required: true
    equals_string: link_attribute

```
</details>

### Induced

<details>
```yaml
name: LinkAttributeMeta
description: '``.zattrs`` for one ``link_attributes/<name>/<delta>/<offsets>/`` array.  Mirrors
  the ``links/<delta>/<offsets>/`` array of the same ``<delta>`` and ``<offsets>``
  cell-for-cell — same cells, same per-cell row order — so attribute rows align 1:1
  with link records without storing a row id.

  Rows never carry the links array''s ``perm_idx`` column: that column describes a
  record''s placement, not its value, so ``has_perm`` has no analogue here and a row
  is exactly ``row_shape`` worth of ``dtype``.  A ``store="duplicate"`` family replicates
  an attribute value alongside each physical copy of its record, so cells stay row-aligned
  with the links array they mirror.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slot_usage:
  zv_array:
    name: zv_array
    required: true
    equals_string: link_attribute
attributes:
  zv_array:
    name: zv_array
    description: 'Discriminator slot identifying the kind of per-array ``.zattrs``
      block.  Each writer in ``core/arrays.py`` stamps the corresponding token from
      :class:`ZvArrayTag`.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LinkAttributeMeta
    domain_of:
    - VerticesMeta
    - LinksFamilyMeta
    - LinksMeta
    - AttributeMeta
    - FragmentAttributeMeta
    - ObjectIndexMeta
    - ObjectAttributeMeta
    - GroupingsMeta
    - GroupingsAttributeMeta
    - LinkAttributeFamilyMeta
    - LinkAttributeMeta
    range: ZvArrayTag
    required: true
    equals_string: link_attribute
  name:
    name: name
    description: NGFF axis or attribute name (e.g. "x", "intensity").
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    slot_uri: schema:name
    owner: LinkAttributeMeta
    domain_of:
    - Axis
    - AttributeMeta
    - FragmentAttributeMeta
    - ObjectAttributeMeta
    - GroupingsAttributeMeta
    - LinkAttributeFamilyMeta
    - LinkAttributeMeta
    range: string
    required: true
  dtype:
    name: dtype
    description: Numpy dtype string of the array's value type (e.g. "float32").
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LinkAttributeMeta
    domain_of:
    - VerticesMeta
    - LinksMeta
    - AttributeMeta
    - FragmentAttributeMeta
    - ObjectAttributeMeta
    - GroupingsAttributeMeta
    - LinkAttributeMeta
    range: string
    required: true
  offsets:
    name: offsets
    description: 'The ``link_width - 1`` relative offsets naming this array, each
      of ``sid_ndim`` signed components, as a list of integer lists. Offsets are relative
      to the record''s SOURCE chunk: ``vi_k`` is local to ``src + o_k``.  The implicit
      ``o_0 = 0`` (the source itself) is never encoded, so ``link_width == 1`` carries
      an empty list.  Redundant with the path segment that encodes the same offsets,
      but stamping it lets readers decode without knowing ``sid_ndim``.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LinkAttributeMeta
    domain_of:
    - LinksMeta
    - LinkAttributeMeta
    range: integer
    required: true
    array:
      dimensions:
      - alias: offset
      - alias: component
  row_shape:
    name: row_shape
    description: 'Tail dimensions of one attribute row (``[]`` for a 1-D per-link
      attribute), letting a reader reconstruct shape from a bare byte blob.  Stamped
      by the writers that lay down attribute bytes; absent on an array created empty.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LinkAttributeMeta
    domain_of:
    - LinkAttributeMeta
    range: integer
    required: false
    multivalued: true
  level_delta:
    name: level_delta
    description: 'Pyramid-level delta between the source side (the level that owns
      this array) and the target side of the edges.  ``0`` for intra-level arrays
      (the only kind written pre-0.4), ``+N`` for edges from this level to ``this_level
      + N`` (coarser), ``-N`` for edges to ``this_level - N`` (finer).

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LinkAttributeMeta
    domain_of:
    - LinksFamilyMeta
    - LinksMeta
    - LinkAttributeFamilyMeta
    - LinkAttributeMeta
    range: integer
    required: true

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: links_convention 

<div data-search-exclude markdown="1">



URI: [zv:links_convention](https://w3id.org/zarr-vectors/schema/0.5/links_convention)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [RootMetadata](RootMetadata.md) | Root-level `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [LinksConvention](LinksConvention.md) |
| Domain Of | [RootMetadata](RootMetadata.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:links_convention |
| native | zv:links_convention |




## LinkML Source

<details>
```yaml
name: links_convention
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- RootMetadata
range: LinksConvention

```
</details></div>


---

---
search:
  boost: 2.0
---


# Enum: LinksConvention 




_How intra-chunk links are represented for a polyline/graph/mesh._

__



<div data-search-exclude markdown="1">

URI: [zv:LinksConvention](https://w3id.org/zarr-vectors/schema/0.5/LinksConvention)

## Permissible Values
| Value | Meaning | Description |
| --- | --- | --- |
| explicit | None | An explicit per-chunk ``links/<chunk_key>`` array is written |
| implicit_sequential | None | Vertex ordering encodes the polyline; no links array |
| implicit_sequential_with_branches | None | Skeleton convention — implicit sequential plus branch list |




## Slots

| Name | Description |
| ---  | --- |
| [links_convention](links_convention.md) |  |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5






## LinkML Source

<details>
```yaml
name: LinksConvention
description: 'How intra-chunk links are represented for a polyline/graph/mesh.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
permissible_values:
  explicit:
    text: explicit
    description: An explicit per-chunk ``links/<chunk_key>`` array is written.
  implicit_sequential:
    text: implicit_sequential
    description: Vertex ordering encodes the polyline; no links array.
  implicit_sequential_with_branches:
    text: implicit_sequential_with_branches
    description: Skeleton convention — implicit sequential plus branch list.

```
</details>

</div>


---

---
search:
  boost: 10.0
---

# Class: LinksFamilyMeta 


_``.zattrs`` for a ``links/<delta>/`` **group** — the family-wide policy every offsets array beneath it decodes against, which is why it lives on the group rather than being repeated per array._

_Since 0.9.0 connectivity is one family: there is no separate ``cross_chunk_links/``.  An intra-chunk link is a link whose relative offsets are all zero; a cross-chunk link is one whose offsets are non-zero._

_Two policy flags govern how a boundary-crossing record is placed: ``directed`` (endpoint order is data, so ``A→B`` and ``B→A`` stay distinct) and ``store`` (``canonical`` = one cell per record; ``duplicate`` = one cell per distinct incident chunk).  The source-side endpoint (input endpoint 0) lives at the level owning the array; target-side endpoints live at ``this_level + level_delta``._

_``num_links`` is the family-wide *logical* record count (one per input record) and ``num_physical_records`` the on-disk row count; both are absent until ``finalize_links`` has run, since the decentralized per-cell writers leave them for it to fill in. ``sid_ndim`` is likewise optional: a family may be stamped without it, and readers then recover offsets from each array's own ``offsets`` meta._

__



<div data-search-exclude markdown="1">



URI: [zv:LinksFamilyMeta](https://w3id.org/zarr-vectors/schema/0.5/LinksFamilyMeta)





```mermaid
 classDiagram
    class LinksFamilyMeta
    click LinksFamilyMeta href "../LinksFamilyMeta/"
      LinksFamilyMeta : directed
        
      LinksFamilyMeta : level_delta
        
      LinksFamilyMeta : link_width
        
      LinksFamilyMeta : num_links
        
      LinksFamilyMeta : num_physical_records
        
      LinksFamilyMeta : sid_ndim
        
      LinksFamilyMeta : store
        
          
    
        
        
        LinksFamilyMeta --> "0..1" CrossChunkStore : store
        click CrossChunkStore href "../CrossChunkStore/"
    

        
      LinksFamilyMeta : zv_array
        
          
    
        
        
        LinksFamilyMeta --> "1" ZvArrayTag : zv_array
        click ZvArrayTag href "../ZvArrayTag/"
    

        
      
```




<!-- no inheritance hierarchy -->

## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [zv_array](zv_array.md) | 1 <br/> [ZvArrayTag](ZvArrayTag.md) | Discriminator slot identifying the kind of per-array `` | direct |
| [level_delta](level_delta.md) | 1 <br/> [Integer](Integer.md) | Pyramid-level delta between the source side (the level that owns this array) ... | direct |
| [link_width](link_width.md) | 1 <br/> [Integer](Integer.md) | Width of a links row (1 for parent→child metanode references, 2 for edges, 3 ... | direct |
| [directed](directed.md) | 0..1 <br/> [Boolean](Boolean.md) | When true, endpoint order is meaningful and preserved (no canonical sort): ``... | direct |
| [store](store.md) | 0..1 <br/> [CrossChunkStore](CrossChunkStore.md) | Cell-duplication policy for a links family; see CrossChunkStore | direct |
| [sid_ndim](sid_ndim.md) | 0..1 <br/> [Integer](Integer.md) | Number of spatial-index dimensions encoded in chunk keys and in each offset | direct |
| [num_links](num_links.md) | 0..1 <br/> [Integer](Integer.md) | Family-wide *logical* link count — one per input record, however many physica... | direct |
| [num_physical_records](num_physical_records.md) | 0..1 <br/> [Integer](Integer.md) | On-disk record row count across every offsets array and cell of a ``links/<de... | direct |















## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:LinksFamilyMeta |
| native | zv:LinksFamilyMeta |






## LinkML Source

### Direct

<details>
```yaml
name: LinksFamilyMeta
description: '``.zattrs`` for a ``links/<delta>/`` **group** — the family-wide policy
  every offsets array beneath it decodes against, which is why it lives on the group
  rather than being repeated per array.

  Since 0.9.0 connectivity is one family: there is no separate ``cross_chunk_links/``.  An
  intra-chunk link is a link whose relative offsets are all zero; a cross-chunk link
  is one whose offsets are non-zero.

  Two policy flags govern how a boundary-crossing record is placed: ``directed`` (endpoint
  order is data, so ``A→B`` and ``B→A`` stay distinct) and ``store`` (``canonical``
  = one cell per record; ``duplicate`` = one cell per distinct incident chunk).  The
  source-side endpoint (input endpoint 0) lives at the level owning the array; target-side
  endpoints live at ``this_level + level_delta``.

  ``num_links`` is the family-wide *logical* record count (one per input record) and
  ``num_physical_records`` the on-disk row count; both are absent until ``finalize_links``
  has run, since the decentralized per-cell writers leave them for it to fill in.
  ``sid_ndim`` is likewise optional: a family may be stamped without it, and readers
  then recover offsets from each array''s own ``offsets`` meta.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slots:
- zv_array
- level_delta
- link_width
- directed
- store
- sid_ndim
- num_links
- num_physical_records
slot_usage:
  zv_array:
    name: zv_array
    required: true
    equals_string: links_family
  sid_ndim:
    name: sid_ndim
    description: 'Number of spatial-index dimensions encoded in chunk keys and in
      each offset.  Optional on a links family: writers stamp it only when known,
      and ``finalize_links`` preserves its absence rather than inventing one.

      '
    required: false

```
</details>

### Induced

<details>
```yaml
name: LinksFamilyMeta
description: '``.zattrs`` for a ``links/<delta>/`` **group** — the family-wide policy
  every offsets array beneath it decodes against, which is why it lives on the group
  rather than being repeated per array.

  Since 0.9.0 connectivity is one family: there is no separate ``cross_chunk_links/``.  An
  intra-chunk link is a link whose relative offsets are all zero; a cross-chunk link
  is one whose offsets are non-zero.

  Two policy flags govern how a boundary-crossing record is placed: ``directed`` (endpoint
  order is data, so ``A→B`` and ``B→A`` stay distinct) and ``store`` (``canonical``
  = one cell per record; ``duplicate`` = one cell per distinct incident chunk).  The
  source-side endpoint (input endpoint 0) lives at the level owning the array; target-side
  endpoints live at ``this_level + level_delta``.

  ``num_links`` is the family-wide *logical* record count (one per input record) and
  ``num_physical_records`` the on-disk row count; both are absent until ``finalize_links``
  has run, since the decentralized per-cell writers leave them for it to fill in.
  ``sid_ndim`` is likewise optional: a family may be stamped without it, and readers
  then recover offsets from each array''s own ``offsets`` meta.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slot_usage:
  zv_array:
    name: zv_array
    required: true
    equals_string: links_family
  sid_ndim:
    name: sid_ndim
    description: 'Number of spatial-index dimensions encoded in chunk keys and in
      each offset.  Optional on a links family: writers stamp it only when known,
      and ``finalize_links`` preserves its absence rather than inventing one.

      '
    required: false
attributes:
  zv_array:
    name: zv_array
    description: 'Discriminator slot identifying the kind of per-array ``.zattrs``
      block.  Each writer in ``core/arrays.py`` stamps the corresponding token from
      :class:`ZvArrayTag`.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LinksFamilyMeta
    domain_of:
    - VerticesMeta
    - LinksFamilyMeta
    - LinksMeta
    - AttributeMeta
    - FragmentAttributeMeta
    - ObjectIndexMeta
    - ObjectAttributeMeta
    - GroupingsMeta
    - GroupingsAttributeMeta
    - LinkAttributeFamilyMeta
    - LinkAttributeMeta
    range: ZvArrayTag
    required: true
    equals_string: links_family
  level_delta:
    name: level_delta
    description: 'Pyramid-level delta between the source side (the level that owns
      this array) and the target side of the edges.  ``0`` for intra-level arrays
      (the only kind written pre-0.4), ``+N`` for edges from this level to ``this_level
      + N`` (coarser), ``-N`` for edges to ``this_level - N`` (finer).

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LinksFamilyMeta
    domain_of:
    - LinksFamilyMeta
    - LinksMeta
    - LinkAttributeFamilyMeta
    - LinkAttributeMeta
    range: integer
    required: true
  link_width:
    name: link_width
    description: 'Width of a links row (1 for parent→child metanode references, 2
      for edges, 3 for triangle faces, 4 for quads).

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LinksFamilyMeta
    domain_of:
    - LinksFamilyMeta
    - LinksMeta
    range: integer
    required: true
    minimum_value: 1
  directed:
    name: directed
    description: 'When true, endpoint order is meaningful and preserved (no canonical
      sort): ``A→B`` and ``B→A`` occupy distinct cells and every stored ``perm_idx``
      is 0.  Family-wide; defaults to false (undirected).

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LinksFamilyMeta
    domain_of:
    - LinksFamilyMeta
    range: boolean
    required: false
  store:
    name: store
    description: 'Cell-duplication policy for a links family; see CrossChunkStore.
      Family-wide; defaults to ``canonical``.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LinksFamilyMeta
    domain_of:
    - LinksFamilyMeta
    range: CrossChunkStore
    required: false
  sid_ndim:
    name: sid_ndim
    description: 'Number of spatial-index dimensions encoded in chunk keys and in
      each offset.  Optional on a links family: writers stamp it only when known,
      and ``finalize_links`` preserves its absence rather than inventing one.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LinksFamilyMeta
    domain_of:
    - LinksFamilyMeta
    - ObjectIndexMeta
    range: integer
    required: false
    minimum_value: 1
  num_links:
    name: num_links
    description: 'Family-wide *logical* link count — one per input record, however
      many physical copies ``store`` filed.  Absent on a ``links/<delta>/`` group
      until ``finalize_links`` has run, and on a ``link_attributes/<name>/<delta>/``
      group created without data.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LinksFamilyMeta
    domain_of:
    - LinksFamilyMeta
    - LinkAttributeFamilyMeta
    range: integer
    required: false
    minimum_value: 0
  num_physical_records:
    name: num_physical_records
    description: 'On-disk record row count across every offsets array and cell of
      a ``links/<delta>/`` family.  Equals ``num_links`` for ``store=canonical``;
      larger when ``store=duplicate`` replicates a record across its incident-chunk
      cells.  Stamped by ``finalize_links``.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LinksFamilyMeta
    domain_of:
    - LinksFamilyMeta
    range: integer
    required: false
    minimum_value: 0

```
</details></div>


---

---
search:
  boost: 10.0
---

# Class: LinksMeta 


_``.zattrs`` for a single ``links/<delta>/<offsets>/`` array: a rank-D vlen array over the level's chunk grid, one cell per **source** chunk, holding the records whose source chunk is that cell._

_The ``offsets`` segment names the array: ``link_width - 1`` relative offsets, each of ``sid_ndim`` signed components.  Record ``vi_k`` is local to chunk ``src + o_k``; the implicit ``o_0 = 0`` (the source itself) is never encoded.  All-zero offsets is the intra-chunk array; non-zero offsets hold records whose other endpoints sit that far away.  ``link_width == 1`` (parent→child metanode refs) has no other endpoint and so carries an empty ``offsets`` list, written at the ``self`` segment._

_``has_perm`` is the record-width discriminator.  Rows are ``L`` ints normally; when ``has_perm`` is true they are ``1 + L`` (``[perm_idx, vi_0, ..., vi_{L-1}]``), where ``perm_idx`` is the Lehmer code of the canonical→input-order permutation, letting readers recover the original endpoint order (mesh-face winding, directed-edge direction).  It is true exactly when a non-identity placement is possible — non-intra offsets AND ``level_delta == 0`` AND (``store="duplicate"`` OR not ``directed``) — so an intra-chunk link never pays the extra 8 bytes for a value that is always 0._

__



<div data-search-exclude markdown="1">



URI: [zv:LinksMeta](https://w3id.org/zarr-vectors/schema/0.5/LinksMeta)





```mermaid
 classDiagram
    class LinksMeta
    click LinksMeta href "../LinksMeta/"
      LinksMeta : dtype
        
      LinksMeta : has_perm
        
      LinksMeta : level_delta
        
      LinksMeta : link_width
        
      LinksMeta : offsets
        
      LinksMeta : zv_array
        
          
    
        
        
        LinksMeta --> "1" ZvArrayTag : zv_array
        click ZvArrayTag href "../ZvArrayTag/"
    

        
      
```




<!-- no inheritance hierarchy -->

## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [zv_array](zv_array.md) | 1 <br/> [ZvArrayTag](ZvArrayTag.md) | Discriminator slot identifying the kind of per-array `` | direct |
| [dtype](dtype.md) | 1 <br/> [String](String.md) | Numpy dtype string of the array's value type (e | direct |
| [offsets](offsets.md) | 1 <br/> [Integer](Integer.md) | The ``link_width - 1`` relative offsets naming this array, each of ``sid_ndim... | direct |
| [has_perm](has_perm.md) | 1 <br/> [Boolean](Boolean.md) | Whether each row of this links array carries a leading ``perm_idx`` column — ... | direct |
| [link_width](link_width.md) | 1 <br/> [Integer](Integer.md) | Width of a links row (1 for parent→child metanode references, 2 for edges, 3 ... | direct |
| [level_delta](level_delta.md) | 1 <br/> [Integer](Integer.md) | Pyramid-level delta between the source side (the level that owns this array) ... | direct |















## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:LinksMeta |
| native | zv:LinksMeta |






## LinkML Source

### Direct

<details>
```yaml
name: LinksMeta
description: '``.zattrs`` for a single ``links/<delta>/<offsets>/`` array: a rank-D
  vlen array over the level''s chunk grid, one cell per **source** chunk, holding
  the records whose source chunk is that cell.

  The ``offsets`` segment names the array: ``link_width - 1`` relative offsets, each
  of ``sid_ndim`` signed components.  Record ``vi_k`` is local to chunk ``src + o_k``;
  the implicit ``o_0 = 0`` (the source itself) is never encoded.  All-zero offsets
  is the intra-chunk array; non-zero offsets hold records whose other endpoints sit
  that far away.  ``link_width == 1`` (parent→child metanode refs) has no other endpoint
  and so carries an empty ``offsets`` list, written at the ``self`` segment.

  ``has_perm`` is the record-width discriminator.  Rows are ``L`` ints normally; when
  ``has_perm`` is true they are ``1 + L`` (``[perm_idx, vi_0, ..., vi_{L-1}]``), where
  ``perm_idx`` is the Lehmer code of the canonical→input-order permutation, letting
  readers recover the original endpoint order (mesh-face winding, directed-edge direction).  It
  is true exactly when a non-identity placement is possible — non-intra offsets AND
  ``level_delta == 0`` AND (``store="duplicate"`` OR not ``directed``) — so an intra-chunk
  link never pays the extra 8 bytes for a value that is always 0.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slots:
- zv_array
- dtype
- offsets
- has_perm
- link_width
- level_delta
slot_usage:
  zv_array:
    name: zv_array
    required: true
    equals_string: links

```
</details>

### Induced

<details>
```yaml
name: LinksMeta
description: '``.zattrs`` for a single ``links/<delta>/<offsets>/`` array: a rank-D
  vlen array over the level''s chunk grid, one cell per **source** chunk, holding
  the records whose source chunk is that cell.

  The ``offsets`` segment names the array: ``link_width - 1`` relative offsets, each
  of ``sid_ndim`` signed components.  Record ``vi_k`` is local to chunk ``src + o_k``;
  the implicit ``o_0 = 0`` (the source itself) is never encoded.  All-zero offsets
  is the intra-chunk array; non-zero offsets hold records whose other endpoints sit
  that far away.  ``link_width == 1`` (parent→child metanode refs) has no other endpoint
  and so carries an empty ``offsets`` list, written at the ``self`` segment.

  ``has_perm`` is the record-width discriminator.  Rows are ``L`` ints normally; when
  ``has_perm`` is true they are ``1 + L`` (``[perm_idx, vi_0, ..., vi_{L-1}]``), where
  ``perm_idx`` is the Lehmer code of the canonical→input-order permutation, letting
  readers recover the original endpoint order (mesh-face winding, directed-edge direction).  It
  is true exactly when a non-identity placement is possible — non-intra offsets AND
  ``level_delta == 0`` AND (``store="duplicate"`` OR not ``directed``) — so an intra-chunk
  link never pays the extra 8 bytes for a value that is always 0.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slot_usage:
  zv_array:
    name: zv_array
    required: true
    equals_string: links
attributes:
  zv_array:
    name: zv_array
    description: 'Discriminator slot identifying the kind of per-array ``.zattrs``
      block.  Each writer in ``core/arrays.py`` stamps the corresponding token from
      :class:`ZvArrayTag`.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LinksMeta
    domain_of:
    - VerticesMeta
    - LinksFamilyMeta
    - LinksMeta
    - AttributeMeta
    - FragmentAttributeMeta
    - ObjectIndexMeta
    - ObjectAttributeMeta
    - GroupingsMeta
    - GroupingsAttributeMeta
    - LinkAttributeFamilyMeta
    - LinkAttributeMeta
    range: ZvArrayTag
    required: true
    equals_string: links
  dtype:
    name: dtype
    description: Numpy dtype string of the array's value type (e.g. "float32").
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LinksMeta
    domain_of:
    - VerticesMeta
    - LinksMeta
    - AttributeMeta
    - FragmentAttributeMeta
    - ObjectAttributeMeta
    - GroupingsAttributeMeta
    - LinkAttributeMeta
    range: string
    required: true
  offsets:
    name: offsets
    description: 'The ``link_width - 1`` relative offsets naming this array, each
      of ``sid_ndim`` signed components, as a list of integer lists. Offsets are relative
      to the record''s SOURCE chunk: ``vi_k`` is local to ``src + o_k``.  The implicit
      ``o_0 = 0`` (the source itself) is never encoded, so ``link_width == 1`` carries
      an empty list.  Redundant with the path segment that encodes the same offsets,
      but stamping it lets readers decode without knowing ``sid_ndim``.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LinksMeta
    domain_of:
    - LinksMeta
    - LinkAttributeMeta
    range: integer
    required: true
    array:
      dimensions:
      - alias: offset
      - alias: component
  has_perm:
    name: has_perm
    description: 'Whether each row of this links array carries a leading ``perm_idx``
      column — i.e. whether rows are ``1 + link_width`` ints rather than ``link_width``.  True
      exactly when a non-identity endpoint placement is possible: non-intra ``offsets``
      AND ``level_delta == 0`` AND (``store="duplicate"`` OR not ``directed``).  Readers
      must take the record width from this stamp rather than inferring it: decoding
      physical bytes at the logical width does not reliably raise, it silently fabricates
      rows.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LinksMeta
    domain_of:
    - LinksMeta
    range: boolean
    required: true
  link_width:
    name: link_width
    description: 'Width of a links row (1 for parent→child metanode references, 2
      for edges, 3 for triangle faces, 4 for quads).

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LinksMeta
    domain_of:
    - LinksFamilyMeta
    - LinksMeta
    range: integer
    required: true
    minimum_value: 1
  level_delta:
    name: level_delta
    description: 'Pyramid-level delta between the source side (the level that owns
      this array) and the target side of the edges.  ``0`` for intra-level arrays
      (the only kind written pre-0.4), ``+N`` for edges from this level to ``this_level
      + N`` (coarser), ``-N`` for edges to ``this_level - N`` (finer).

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: LinksMeta
    domain_of:
    - LinksFamilyMeta
    - LinksMeta
    - LinkAttributeFamilyMeta
    - LinkAttributeMeta
    range: integer
    required: true

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: max_corner 


_Per-axis maxima.  Length must equal ``len(spatial_index_dims)``._



<div data-search-exclude markdown="1">



URI: [zv:max_corner](https://w3id.org/zarr-vectors/schema/0.5/max_corner)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [BoundingBox](BoundingBox.md) | Two parallel ``ndim``-length arrays representing the global ``(min_corner, ma... |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Float](Float.md) |
| Domain Of | [BoundingBox](BoundingBox.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
| Multivalued | Yes |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:max_corner |
| native | zv:max_corner |




## LinkML Source

<details>
```yaml
name: max_corner
description: Per-axis maxima.  Length must equal ``len(spatial_index_dims)``.
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- BoundingBox
range: float
required: true
multivalued: true

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: min_corner 


_Per-axis minima.  Length must equal ``len(spatial_index_dims)``._



<div data-search-exclude markdown="1">



URI: [zv:min_corner](https://w3id.org/zarr-vectors/schema/0.5/min_corner)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [BoundingBox](BoundingBox.md) | Two parallel ``ndim``-length arrays representing the global ``(min_corner, ma... |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Float](Float.md) |
| Domain Of | [BoundingBox](BoundingBox.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
| Multivalued | Yes |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:min_corner |
| native | zv:min_corner |




## LinkML Source

<details>
```yaml
name: min_corner
description: Per-axis minima.  Length must equal ``len(spatial_index_dims)``.
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- BoundingBox
range: float
required: true
multivalued: true

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: name 


_NGFF axis or attribute name (e.g. "x", "intensity")._



<div data-search-exclude markdown="1">



URI: [schema:name](http://schema.org/name)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [Axis](Axis.md) | One axis of the spatial index |  no  |
| [AttributeMeta](AttributeMeta.md) | `` |  no  |
| [FragmentAttributeMeta](FragmentAttributeMeta.md) | `` |  no  |
| [ObjectAttributeMeta](ObjectAttributeMeta.md) | `` |  no  |
| [GroupingsAttributeMeta](GroupingsAttributeMeta.md) | `` |  no  |
| [LinkAttributeFamilyMeta](LinkAttributeFamilyMeta.md) | `` |  no  |
| [LinkAttributeMeta](LinkAttributeMeta.md) | `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [Axis](Axis.md), [AttributeMeta](AttributeMeta.md), [FragmentAttributeMeta](FragmentAttributeMeta.md), [ObjectAttributeMeta](ObjectAttributeMeta.md), [GroupingsAttributeMeta](GroupingsAttributeMeta.md), [LinkAttributeFamilyMeta](LinkAttributeFamilyMeta.md), [LinkAttributeMeta](LinkAttributeMeta.md) |
| Slot URI | [schema:name](http://schema.org/name) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | schema:name |
| native | zv:name |




## LinkML Source

<details>
```yaml
name: name
description: NGFF axis or attribute name (e.g. "x", "intensity").
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slot_uri: schema:name
domain_of:
- Axis
- AttributeMeta
- FragmentAttributeMeta
- ObjectAttributeMeta
- GroupingsAttributeMeta
- LinkAttributeFamilyMeta
- LinkAttributeMeta
range: string
required: true

```
</details></div>


---

---
search:
  boost: 1.0
---# Type: Ncname 




_Prefix part of CURIE_



<div data-search-exclude markdown="1">

URI: [xsd:string](http://www.w3.org/2001/XMLSchema#string)

## Type Properties

| Property | Value |
| --- | --- |
| Base | `NCName` |
| Type URI | [xsd:string](http://www.w3.org/2001/XMLSchema#string) |
| Representation | `str` |






## Notes

* If you are authoring schemas in LinkML YAML, the type is referenced with the lower case "ncname".



## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | xsd:string |
| native | zv:ncname |




</div>


---

---
search:
  boost: 1.0
---# Type: Nodeidentifier 




_A URI, CURIE or BNODE that represents a node in a model._



<div data-search-exclude markdown="1">

URI: [shex:nonLiteral](http://www.w3.org/ns/shex#nonLiteral)

## Type Properties

| Property | Value |
| --- | --- |
| Base | `NodeIdentifier` |
| Type URI | [shex:nonLiteral](http://www.w3.org/ns/shex#nonLiteral) |
| Representation | `str` |






## Notes

* If you are authoring schemas in LinkML YAML, the type is referenced with the lower case "nodeidentifier".



## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | shex:nonLiteral |
| native | zv:nodeidentifier |




</div>


---

---
search:
  boost: 5.0
---

# Slot: num_groups 


_Total grouping count._



<div data-search-exclude markdown="1">



URI: [zv:num_groups](https://w3id.org/zarr-vectors/schema/0.5/num_groups)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [GroupingsMeta](GroupingsMeta.md) | `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Integer](Integer.md) |
| Domain Of | [GroupingsMeta](GroupingsMeta.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
### Value Constraints

| Property | Value |
| --- | --- |
| Minimum Value | 0 |












## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:num_groups |
| native | zv:num_groups |




## LinkML Source

<details>
```yaml
name: num_groups
description: Total grouping count.
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- GroupingsMeta
range: integer
required: true
minimum_value: 0

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: num_links 


_Family-wide *logical* link count — one per input record, however many physical copies ``store`` filed.  Absent on a ``links/<delta>/`` group until ``finalize_links`` has run, and on a ``link_attributes/<name>/<delta>/`` group created without data._

__



<div data-search-exclude markdown="1">



URI: [zv:num_links](https://w3id.org/zarr-vectors/schema/0.5/num_links)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LinksFamilyMeta](LinksFamilyMeta.md) | `` |  no  |
| [LinkAttributeFamilyMeta](LinkAttributeFamilyMeta.md) | `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Integer](Integer.md) |
| Domain Of | [LinksFamilyMeta](LinksFamilyMeta.md), [LinkAttributeFamilyMeta](LinkAttributeFamilyMeta.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
### Value Constraints

| Property | Value |
| --- | --- |
| Minimum Value | 0 |












## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:num_links |
| native | zv:num_links |




## LinkML Source

<details>
```yaml
name: num_links
description: 'Family-wide *logical* link count — one per input record, however many
  physical copies ``store`` filed.  Absent on a ``links/<delta>/`` group until ``finalize_links``
  has run, and on a ``link_attributes/<name>/<delta>/`` group created without data.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- LinksFamilyMeta
- LinkAttributeFamilyMeta
range: integer
required: false
minimum_value: 0

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: num_objects 


_Total object count this array carries._



<div data-search-exclude markdown="1">



URI: [zv:num_objects](https://w3id.org/zarr-vectors/schema/0.5/num_objects)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [ObjectIndexMeta](ObjectIndexMeta.md) | `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Integer](Integer.md) |
| Domain Of | [ObjectIndexMeta](ObjectIndexMeta.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
### Value Constraints

| Property | Value |
| --- | --- |
| Minimum Value | 0 |












## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:num_objects |
| native | zv:num_objects |




## LinkML Source

<details>
```yaml
name: num_objects
description: Total object count this array carries.
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- ObjectIndexMeta
range: integer
required: true
minimum_value: 0

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: num_physical_records 


_On-disk record row count across every offsets array and cell of a ``links/<delta>/`` family.  Equals ``num_links`` for ``store=canonical``; larger when ``store=duplicate`` replicates a record across its incident-chunk cells.  Stamped by ``finalize_links``._

__



<div data-search-exclude markdown="1">



URI: [zv:num_physical_records](https://w3id.org/zarr-vectors/schema/0.5/num_physical_records)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LinksFamilyMeta](LinksFamilyMeta.md) | `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Integer](Integer.md) |
| Domain Of | [LinksFamilyMeta](LinksFamilyMeta.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
### Value Constraints

| Property | Value |
| --- | --- |
| Minimum Value | 0 |












## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:num_physical_records |
| native | zv:num_physical_records |




## LinkML Source

<details>
```yaml
name: num_physical_records
description: 'On-disk record row count across every offsets array and cell of a ``links/<delta>/``
  family.  Equals ``num_links`` for ``store=canonical``; larger when ``store=duplicate``
  replicates a record across its incident-chunk cells.  Stamped by ``finalize_links``.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- LinksFamilyMeta
range: integer
required: false
minimum_value: 0

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: object_index_convention 

<div data-search-exclude markdown="1">



URI: [zv:object_index_convention](https://w3id.org/zarr-vectors/schema/0.5/object_index_convention)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [RootMetadata](RootMetadata.md) | Root-level `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [ObjectIndexConvention](ObjectIndexConvention.md) |
| Domain Of | [RootMetadata](RootMetadata.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:object_index_convention |
| native | zv:object_index_convention |




## LinkML Source

<details>
```yaml
name: object_index_convention
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- RootMetadata
range: ObjectIndexConvention

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: object_sparsity 


_Fraction of objects retained at this level._



<div data-search-exclude markdown="1">



URI: [zv:object_sparsity](https://w3id.org/zarr-vectors/schema/0.5/object_sparsity)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LevelMetadata](LevelMetadata.md) | Per-resolution-level `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Float](Float.md) |
| Domain Of | [LevelMetadata](LevelMetadata.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
### Value Constraints

| Property | Value |
| --- | --- |
| Minimum Value | 0 |
| Maximum Value | 1 |












## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:object_sparsity |
| native | zv:object_sparsity |




## LinkML Source

<details>
```yaml
name: object_sparsity
description: Fraction of objects retained at this level.
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- LevelMetadata
range: float
minimum_value: 0.0
maximum_value: 1.0

```
</details></div>


---

---
search:
  boost: 10.0
---

# Class: ObjectAttributeMeta 


_``.zattrs`` for each ``object_attributes/<name>/`` array._



<div data-search-exclude markdown="1">



URI: [zv:ObjectAttributeMeta](https://w3id.org/zarr-vectors/schema/0.5/ObjectAttributeMeta)





```mermaid
 classDiagram
    class ObjectAttributeMeta
    click ObjectAttributeMeta href "../ObjectAttributeMeta/"
      ObjectAttributeMeta : dtype
        
      ObjectAttributeMeta : name
        
      ObjectAttributeMeta : shape
        
      ObjectAttributeMeta : zv_array
        
          
    
        
        
        ObjectAttributeMeta --> "1" ZvArrayTag : zv_array
        click ZvArrayTag href "../ZvArrayTag/"
    

        
      
```




<!-- no inheritance hierarchy -->

## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [zv_array](zv_array.md) | 1 <br/> [ZvArrayTag](ZvArrayTag.md) | Discriminator slot identifying the kind of per-array `` | direct |
| [name](name.md) | 1 <br/> [String](String.md) | NGFF axis or attribute name (e | direct |
| [dtype](dtype.md) | 1 <br/> [String](String.md) | Numpy dtype string of the array's value type (e | direct |
| [shape](shape.md) | 1..* <br/> [Integer](Integer.md) | Shape of a dense per-object/per-group array | direct |















## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:ObjectAttributeMeta |
| native | zv:ObjectAttributeMeta |






## LinkML Source

### Direct

<details>
```yaml
name: ObjectAttributeMeta
description: '``.zattrs`` for each ``object_attributes/<name>/`` array.'
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slots:
- zv_array
- name
- dtype
- shape
slot_usage:
  zv_array:
    name: zv_array
    required: true
    equals_string: object_attribute

```
</details>

### Induced

<details>
```yaml
name: ObjectAttributeMeta
description: '``.zattrs`` for each ``object_attributes/<name>/`` array.'
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slot_usage:
  zv_array:
    name: zv_array
    required: true
    equals_string: object_attribute
attributes:
  zv_array:
    name: zv_array
    description: 'Discriminator slot identifying the kind of per-array ``.zattrs``
      block.  Each writer in ``core/arrays.py`` stamps the corresponding token from
      :class:`ZvArrayTag`.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: ObjectAttributeMeta
    domain_of:
    - VerticesMeta
    - LinksFamilyMeta
    - LinksMeta
    - AttributeMeta
    - FragmentAttributeMeta
    - ObjectIndexMeta
    - ObjectAttributeMeta
    - GroupingsMeta
    - GroupingsAttributeMeta
    - LinkAttributeFamilyMeta
    - LinkAttributeMeta
    range: ZvArrayTag
    required: true
    equals_string: object_attribute
  name:
    name: name
    description: NGFF axis or attribute name (e.g. "x", "intensity").
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    slot_uri: schema:name
    owner: ObjectAttributeMeta
    domain_of:
    - Axis
    - AttributeMeta
    - FragmentAttributeMeta
    - ObjectAttributeMeta
    - GroupingsAttributeMeta
    - LinkAttributeFamilyMeta
    - LinkAttributeMeta
    range: string
    required: true
  dtype:
    name: dtype
    description: Numpy dtype string of the array's value type (e.g. "float32").
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: ObjectAttributeMeta
    domain_of:
    - VerticesMeta
    - LinksMeta
    - AttributeMeta
    - FragmentAttributeMeta
    - ObjectAttributeMeta
    - GroupingsAttributeMeta
    - LinkAttributeMeta
    range: string
    required: true
  shape:
    name: shape
    description: Shape of a dense per-object/per-group array.
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: ObjectAttributeMeta
    domain_of:
    - ObjectAttributeMeta
    - GroupingsAttributeMeta
    range: integer
    required: true
    multivalued: true

```
</details></div>


---

---
search:
  boost: 1.0
---# Type: Objectidentifier 




_A URI or CURIE that represents an object in the model._



<div data-search-exclude markdown="1">

URI: [shex:iri](http://www.w3.org/ns/shex#iri)

## Type Properties

| Property | Value |
| --- | --- |
| Base | `ElementIdentifier` |
| Type URI | [shex:iri](http://www.w3.org/ns/shex#iri) |
| Representation | `str` |






## Comments

* Used for inheritance and type checking

## Notes

* If you are authoring schemas in LinkML YAML, the type is referenced with the lower case "objectidentifier".



## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | shex:iri |
| native | zv:objectidentifier |




</div>


---

---
search:
  boost: 2.0
---


# Enum: ObjectIndexConvention 




_How the object_id → fragment mapping is encoded._



<div data-search-exclude markdown="1">

URI: [zv:ObjectIndexConvention](https://w3id.org/zarr-vectors/schema/0.5/ObjectIndexConvention)

## Permissible Values
| Value | Meaning | Description |
| --- | --- | --- |
| standard | None | Full ``object_index`` array maps each object to its fragment refs |
| identity | None | One object per vertex; only valid in single-chunk stores where the object_ind... |




## Slots

| Name | Description |
| ---  | --- |
| [object_index_convention](object_index_convention.md) |  |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5






## LinkML Source

<details>
```yaml
name: ObjectIndexConvention
description: How the object_id → fragment mapping is encoded.
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
permissible_values:
  standard:
    text: standard
    description: Full ``object_index`` array maps each object to its fragment refs.
  identity:
    text: identity
    description: 'One object per vertex; only valid in single-chunk stores where the
      object_index would be redundant.

      '

```
</details>

</div>


---

---
search:
  boost: 10.0
---

# Class: ObjectIndexMeta 


_``.zattrs`` for ``object_index/``._



<div data-search-exclude markdown="1">



URI: [zv:ObjectIndexMeta](https://w3id.org/zarr-vectors/schema/0.5/ObjectIndexMeta)





```mermaid
 classDiagram
    class ObjectIndexMeta
    click ObjectIndexMeta href "../ObjectIndexMeta/"
      ObjectIndexMeta : num_objects
        
      ObjectIndexMeta : sid_ndim
        
      ObjectIndexMeta : zv_array
        
          
    
        
        
        ObjectIndexMeta --> "1" ZvArrayTag : zv_array
        click ZvArrayTag href "../ZvArrayTag/"
    

        
      
```




<!-- no inheritance hierarchy -->

## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [zv_array](zv_array.md) | 1 <br/> [ZvArrayTag](ZvArrayTag.md) | Discriminator slot identifying the kind of per-array `` | direct |
| [num_objects](num_objects.md) | 1 <br/> [Integer](Integer.md) | Total object count this array carries | direct |
| [sid_ndim](sid_ndim.md) | 1 <br/> [Integer](Integer.md) | Number of spatial-index dimensions encoded in chunk keys | direct |















## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:ObjectIndexMeta |
| native | zv:ObjectIndexMeta |






## LinkML Source

### Direct

<details>
```yaml
name: ObjectIndexMeta
description: '``.zattrs`` for ``object_index/``.'
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slots:
- zv_array
- num_objects
- sid_ndim
slot_usage:
  zv_array:
    name: zv_array
    required: true
    equals_string: object_index

```
</details>

### Induced

<details>
```yaml
name: ObjectIndexMeta
description: '``.zattrs`` for ``object_index/``.'
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slot_usage:
  zv_array:
    name: zv_array
    required: true
    equals_string: object_index
attributes:
  zv_array:
    name: zv_array
    description: 'Discriminator slot identifying the kind of per-array ``.zattrs``
      block.  Each writer in ``core/arrays.py`` stamps the corresponding token from
      :class:`ZvArrayTag`.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: ObjectIndexMeta
    domain_of:
    - VerticesMeta
    - LinksFamilyMeta
    - LinksMeta
    - AttributeMeta
    - FragmentAttributeMeta
    - ObjectIndexMeta
    - ObjectAttributeMeta
    - GroupingsMeta
    - GroupingsAttributeMeta
    - LinkAttributeFamilyMeta
    - LinkAttributeMeta
    range: ZvArrayTag
    required: true
    equals_string: object_index
  num_objects:
    name: num_objects
    description: Total object count this array carries.
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: ObjectIndexMeta
    domain_of:
    - ObjectIndexMeta
    range: integer
    required: true
    minimum_value: 0
  sid_ndim:
    name: sid_ndim
    description: Number of spatial-index dimensions encoded in chunk keys.
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: ObjectIndexMeta
    domain_of:
    - LinksFamilyMeta
    - ObjectIndexMeta
    range: integer
    required: true
    minimum_value: 1

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: offsets 


_The ``link_width - 1`` relative offsets naming this array, each of ``sid_ndim`` signed components, as a list of integer lists. Offsets are relative to the record's SOURCE chunk: ``vi_k`` is local to ``src + o_k``.  The implicit ``o_0 = 0`` (the source itself) is never encoded, so ``link_width == 1`` carries an empty list.  Redundant with the path segment that encodes the same offsets, but stamping it lets readers decode without knowing ``sid_ndim``._

__



<div data-search-exclude markdown="1">



URI: [zv:offsets](https://w3id.org/zarr-vectors/schema/0.5/offsets)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LinksMeta](LinksMeta.md) | `` |  no  |
| [LinkAttributeMeta](LinkAttributeMeta.md) | `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Integer](Integer.md) |
| Domain Of | [LinksMeta](LinksMeta.md), [LinkAttributeMeta](LinkAttributeMeta.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
<details>
<summary>Advanced Properties</summary>
**Array Configuration:**

- **Dimensions:** DimensionExpression({'alias': 'offset'}) x DimensionExpression({'alias': 'component'})

</details>











## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:offsets |
| native | zv:offsets |




## LinkML Source

<details>
```yaml
name: offsets
description: 'The ``link_width - 1`` relative offsets naming this array, each of ``sid_ndim``
  signed components, as a list of integer lists. Offsets are relative to the record''s
  SOURCE chunk: ``vi_k`` is local to ``src + o_k``.  The implicit ``o_0 = 0`` (the
  source itself) is never encoded, so ``link_width == 1`` carries an empty list.  Redundant
  with the path segment that encodes the same offsets, but stamping it lets readers
  decode without knowing ``sid_ndim``.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- LinksMeta
- LinkAttributeMeta
range: integer
required: true
array:
  dimensions:
  - alias: offset
  - alias: component

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: parent_level 


_Source level index (None for level 0)._



<div data-search-exclude markdown="1">



URI: [zv:parent_level](https://w3id.org/zarr-vectors/schema/0.5/parent_level)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LevelMetadata](LevelMetadata.md) | Per-resolution-level `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Integer](Integer.md) |
| Domain Of | [LevelMetadata](LevelMetadata.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
### Value Constraints

| Property | Value |
| --- | --- |
| Minimum Value | 0 |












## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:parent_level |
| native | zv:parent_level |




## LinkML Source

<details>
```yaml
name: parent_level
description: Source level index (None for level 0).
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- LevelMetadata
range: integer
minimum_value: 0

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: preserves_object_ids 


_True for levels written by the per-object pyramid regime.  When set, ``num_objects`` and ``object_attributes`` row count inherit from the parent level's OID space; dropped objects leave empty manifest slots and zero ``present_mask`` bytes.  ``parent_level`` is load-bearing under this flag._

__



<div data-search-exclude markdown="1">



URI: [zv:preserves_object_ids](https://w3id.org/zarr-vectors/schema/0.5/preserves_object_ids)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LevelMetadata](LevelMetadata.md) | Per-resolution-level `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Boolean](Boolean.md) |
| Domain Of | [LevelMetadata](LevelMetadata.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:preserves_object_ids |
| native | zv:preserves_object_ids |




## LinkML Source

<details>
```yaml
name: preserves_object_ids
description: 'True for levels written by the per-object pyramid regime.  When set,
  ``num_objects`` and ``object_attributes`` row count inherit from the parent level''s
  OID space; dropped objects leave empty manifest slots and zero ``present_mask``
  bytes.  ``parent_level`` is load-bearing under this flag.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- LevelMetadata
range: boolean

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: reduction_factor 


_Multi-resolution coarsening factor (≥ 2)._



<div data-search-exclude markdown="1">



URI: [zv:reduction_factor](https://w3id.org/zarr-vectors/schema/0.5/reduction_factor)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [RootMetadata](RootMetadata.md) | Root-level `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Integer](Integer.md) |
| Domain Of | [RootMetadata](RootMetadata.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
### Value Constraints

| Property | Value |
| --- | --- |
| Minimum Value | 2 |












## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:reduction_factor |
| native | zv:reduction_factor |




## LinkML Source

<details>
```yaml
name: reduction_factor
description: Multi-resolution coarsening factor (≥ 2).
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- RootMetadata
range: integer
minimum_value: 2

```
</details></div>


---

---
search:
  boost: 10.0
---

# Class: RootMetadata 


_Root-level ``.zattrs`` payload, persisted under the key ``zarr_vectors``.  Validates the runtime :class:`zarr_vectors.core.metadata.RootMetadata` dataclass. Note: the canonical axis list lives in ``multiscales[0].axes`` at root level (NGFF-style), NOT under ``zarr_vectors`` — see ``MultiscalesMetadata``._

__



<div data-search-exclude markdown="1">



URI: [zv:RootMetadata](https://w3id.org/zarr-vectors/schema/0.5/RootMetadata)





```mermaid
 classDiagram
    class RootMetadata
    click RootMetadata href "../RootMetadata/"
      RootMetadata : base_bin_shape
        
      RootMetadata : bounds
        
          
    
        
        
        RootMetadata --> "1" BoundingBox : bounds
        click BoundingBox href "../BoundingBox/"
    

        
      RootMetadata : chunk_shape
        
      RootMetadata : cross_chunk_strategy
        
          
    
        
        
        RootMetadata --> "0..1" CrossChunkStrategy : cross_chunk_strategy
        click CrossChunkStrategy href "../CrossChunkStrategy/"
    

        
      RootMetadata : cross_level_depth
        
      RootMetadata : cross_level_storage
        
          
    
        
        
        RootMetadata --> "0..1" CrossLevelStorage : cross_level_storage
        click CrossLevelStorage href "../CrossLevelStorage/"
    

        
      RootMetadata : crs
        
          
    
        
        
        RootMetadata --> "0..1" CRS : crs
        click CRS href "../CRS/"
    

        
      RootMetadata : format_capabilities
        
          
    
        
        
        RootMetadata --> "*" FormatCapability : format_capabilities
        click FormatCapability href "../FormatCapability/"
    

        
      RootMetadata : geometry_types
        
          
    
        
        
        RootMetadata --> "1..*" GeometryType : geometry_types
        click GeometryType href "../GeometryType/"
    

        
      RootMetadata : links_convention
        
          
    
        
        
        RootMetadata --> "0..1" LinksConvention : links_convention
        click LinksConvention href "../LinksConvention/"
    

        
      RootMetadata : object_index_convention
        
          
    
        
        
        RootMetadata --> "0..1" ObjectIndexConvention : object_index_convention
        click ObjectIndexConvention href "../ObjectIndexConvention/"
    

        
      RootMetadata : reduction_factor
        
      RootMetadata : zv_version
        
      
```




<!-- no inheritance hierarchy -->

## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [zv_version](zv_version.md) | 1 <br/> [String](String.md) | ZV spec version this store was written against (e | direct |
| [chunk_shape](chunk_shape.md) | 1..* <br/> [Float](Float.md) | Physical spatial chunk size per axis (all values > 0) | direct |
| [bounds](bounds.md) | 1 <br/> [BoundingBox](BoundingBox.md) | Global vertex bounding box | direct |
| [geometry_types](geometry_types.md) | 1..* <br/> [GeometryType](GeometryType.md) | One or more geometry kinds present in the store | direct |
| [crs](crs.md) | 0..1 <br/> [CRS](CRS.md) | Optional coordinate reference system metadata (free-form dict matching whatev... | direct |
| [links_convention](links_convention.md) | 0..1 <br/> [LinksConvention](LinksConvention.md) |  | direct |
| [object_index_convention](object_index_convention.md) | 0..1 <br/> [ObjectIndexConvention](ObjectIndexConvention.md) |  | direct |
| [cross_chunk_strategy](cross_chunk_strategy.md) | 0..1 <br/> [CrossChunkStrategy](CrossChunkStrategy.md) |  | direct |
| [cross_level_depth](cross_level_depth.md) | 0..1 <br/> [Integer](Integer.md) | Maximum absolute level delta for which cross-pyramid-level link arrays are ma... | direct |
| [cross_level_storage](cross_level_storage.md) | 0..1 <br/> [CrossLevelStorage](CrossLevelStorage.md) | Whether cross-level link arrays are written in both directions (``explicit``:... | direct |
| [reduction_factor](reduction_factor.md) | 0..1 <br/> [Integer](Integer.md) | Multi-resolution coarsening factor (≥ 2) | direct |
| [base_bin_shape](base_bin_shape.md) | * <br/> [Float](Float.md) | Supervoxel bin edge lengths at level 0 | direct |
| [format_capabilities](format_capabilities.md) | * <br/> [FormatCapability](FormatCapability.md) | Optional 0 | direct |















## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:RootMetadata |
| native | zv:RootMetadata |






## LinkML Source

### Direct

<details>
```yaml
name: RootMetadata
description: 'Root-level ``.zattrs`` payload, persisted under the key ``zarr_vectors``.  Validates
  the runtime :class:`zarr_vectors.core.metadata.RootMetadata` dataclass. Note: the
  canonical axis list lives in ``multiscales[0].axes`` at root level (NGFF-style),
  NOT under ``zarr_vectors`` — see ``MultiscalesMetadata``.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slots:
- zv_version
- chunk_shape
- bounds
- geometry_types
- crs
- links_convention
- object_index_convention
- cross_chunk_strategy
- cross_level_depth
- cross_level_storage
- reduction_factor
- base_bin_shape
- format_capabilities

```
</details>

### Induced

<details>
```yaml
name: RootMetadata
description: 'Root-level ``.zattrs`` payload, persisted under the key ``zarr_vectors``.  Validates
  the runtime :class:`zarr_vectors.core.metadata.RootMetadata` dataclass. Note: the
  canonical axis list lives in ``multiscales[0].axes`` at root level (NGFF-style),
  NOT under ``zarr_vectors`` — see ``MultiscalesMetadata``.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
attributes:
  zv_version:
    name: zv_version
    description: 'ZV spec version this store was written against (e.g. "0.5.0"). Renamed
      from ``format_version`` in 0.5.0 to disambiguate from Zarr v3''s ``zarr_format``
      field.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    slot_uri: schema:version
    owner: RootMetadata
    domain_of:
    - RootMetadata
    range: string
    required: true
    pattern: ^\d+\.\d+(\.\d+)?$
  chunk_shape:
    name: chunk_shape
    description: Physical spatial chunk size per axis (all values > 0).
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: RootMetadata
    domain_of:
    - RootMetadata
    - LevelMetadata
    range: float
    required: true
    multivalued: true
  bounds:
    name: bounds
    description: Global vertex bounding box.
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: RootMetadata
    domain_of:
    - RootMetadata
    range: BoundingBox
    required: true
  geometry_types:
    name: geometry_types
    description: One or more geometry kinds present in the store.
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: RootMetadata
    domain_of:
    - RootMetadata
    range: GeometryType
    required: true
    multivalued: true
    minimum_cardinality: 1
  crs:
    name: crs
    description: 'Optional coordinate reference system metadata (free-form dict matching
      whatever CRS vocabulary the store uses, e.g. WKT, PROJ4, EPSG, CF conventions).

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    slot_uri: schema:coordinateReferenceSystem
    owner: RootMetadata
    domain_of:
    - RootMetadata
    range: CRS
    inlined: true
  links_convention:
    name: links_convention
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: RootMetadata
    domain_of:
    - RootMetadata
    range: LinksConvention
  object_index_convention:
    name: object_index_convention
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: RootMetadata
    domain_of:
    - RootMetadata
    range: ObjectIndexConvention
  cross_chunk_strategy:
    name: cross_chunk_strategy
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: RootMetadata
    domain_of:
    - RootMetadata
    range: CrossChunkStrategy
  cross_level_depth:
    name: cross_level_depth
    description: 'Maximum absolute level delta for which cross-pyramid-level link
      arrays are materialized.  ``0`` = none (no ``+N`` or ``-N`` arrays), ``N`` =
      generate up to ``±N`` (or ``+N`` only when ``cross_level_storage="implicit"``),
      ``-1`` = all available pyramid levels.  Default ``1``.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: RootMetadata
    domain_of:
    - RootMetadata
    range: integer
    minimum_value: -1
  cross_level_storage:
    name: cross_level_storage
    description: 'Whether cross-level link arrays are written in both directions (``explicit``:
      ``+N`` at the finer level AND ``-N`` at the coarser level) or only positive
      deltas (``implicit``: only ``+N``, with ``-N`` reconstructed on read).  Default
      ``explicit``.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: RootMetadata
    domain_of:
    - RootMetadata
    range: CrossLevelStorage
  reduction_factor:
    name: reduction_factor
    description: Multi-resolution coarsening factor (≥ 2).
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: RootMetadata
    domain_of:
    - RootMetadata
    range: integer
    minimum_value: 2
  base_bin_shape:
    name: base_bin_shape
    description: 'Supervoxel bin edge lengths at level 0.  When set, every value must
      be > 0 and ``chunk_shape`` must be an integer multiple along every axis (enforced
      runtime-side in ``RootMetadata.validate``).

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: RootMetadata
    domain_of:
    - RootMetadata
    range: float
    multivalued: true
  format_capabilities:
    name: format_capabilities
    description: Optional 0.3+ feature tokens advertised by this store.
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: RootMetadata
    domain_of:
    - RootMetadata
    range: FormatCapability
    multivalued: true

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: row_shape 


_Tail dimensions of one attribute row (``[]`` for a 1-D per-link attribute), letting a reader reconstruct shape from a bare byte blob.  Stamped by the writers that lay down attribute bytes; absent on an array created empty._

__



<div data-search-exclude markdown="1">



URI: [zv:row_shape](https://w3id.org/zarr-vectors/schema/0.5/row_shape)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LinkAttributeMeta](LinkAttributeMeta.md) | `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Integer](Integer.md) |
| Domain Of | [LinkAttributeMeta](LinkAttributeMeta.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Multivalued | Yes |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:row_shape |
| native | zv:row_shape |




## LinkML Source

<details>
```yaml
name: row_shape
description: 'Tail dimensions of one attribute row (``[]`` for a 1-D per-link attribute),
  letting a reader reconstruct shape from a bare byte blob.  Stamped by the writers
  that lay down attribute bytes; absent on an array created empty.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- LinkAttributeMeta
range: integer
required: false
multivalued: true

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: shape 


_Shape of a dense per-object/per-group array._



<div data-search-exclude markdown="1">



URI: [zv:shape](https://w3id.org/zarr-vectors/schema/0.5/shape)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [ObjectAttributeMeta](ObjectAttributeMeta.md) | `` |  no  |
| [GroupingsAttributeMeta](GroupingsAttributeMeta.md) | `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Integer](Integer.md) |
| Domain Of | [ObjectAttributeMeta](ObjectAttributeMeta.md), [GroupingsAttributeMeta](GroupingsAttributeMeta.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
| Multivalued | Yes |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:shape |
| native | zv:shape |




## LinkML Source

<details>
```yaml
name: shape
description: Shape of a dense per-object/per-group array.
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- ObjectAttributeMeta
- GroupingsAttributeMeta
range: integer
required: true
multivalued: true

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: shared_fragments 


_True when per-chunk fragments may be referenced by multiple objects' manifests (shared metavertices in the per-object pyramid regime).  v0.6 successor to ``shared_vertex_groups``.  Readers MAY use this to short-circuit dedup work._

__



<div data-search-exclude markdown="1">



URI: [zv:shared_fragments](https://w3id.org/zarr-vectors/schema/0.5/shared_fragments)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LevelMetadata](LevelMetadata.md) | Per-resolution-level `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Boolean](Boolean.md) |
| Domain Of | [LevelMetadata](LevelMetadata.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:shared_fragments |
| native | zv:shared_fragments |




## LinkML Source

<details>
```yaml
name: shared_fragments
description: 'True when per-chunk fragments may be referenced by multiple objects''
  manifests (shared metavertices in the per-object pyramid regime).  v0.6 successor
  to ``shared_vertex_groups``.  Readers MAY use this to short-circuit dedup work.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- LevelMetadata
range: boolean

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: sid_ndim 


_Number of spatial-index dimensions encoded in chunk keys._



<div data-search-exclude markdown="1">



URI: [zv:sid_ndim](https://w3id.org/zarr-vectors/schema/0.5/sid_ndim)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LinksFamilyMeta](LinksFamilyMeta.md) | `` |  yes  |
| [ObjectIndexMeta](ObjectIndexMeta.md) | `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Integer](Integer.md) |
| Domain Of | [LinksFamilyMeta](LinksFamilyMeta.md), [ObjectIndexMeta](ObjectIndexMeta.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
### Value Constraints

| Property | Value |
| --- | --- |
| Minimum Value | 1 |












## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:sid_ndim |
| native | zv:sid_ndim |




## LinkML Source

<details>
```yaml
name: sid_ndim
description: Number of spatial-index dimensions encoded in chunk keys.
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- LinksFamilyMeta
- ObjectIndexMeta
range: integer
required: true
minimum_value: 1

```
</details></div>


---

---
search:
  boost: 1.0
---# Type: Sparqlpath 




_A string encoding a SPARQL Property Path. The value of the string MUST conform to SPARQL syntax and SHOULD dereference to zero or more valid objects within the current instance document when encoded as RDF._



<div data-search-exclude markdown="1">

URI: [xsd:string](http://www.w3.org/2001/XMLSchema#string)

## Type Properties

| Property | Value |
| --- | --- |
| Base | `str` |
| Type URI | [xsd:string](http://www.w3.org/2001/XMLSchema#string) |
| Representation | `str` |






## Notes

* If you are authoring schemas in LinkML YAML, the type is referenced with the lower case "sparqlpath".



## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | xsd:string |
| native | zv:sparqlpath |




</div>


---

---
search:
  boost: 5.0
---

# Slot: store 


_Cell-duplication policy for a links family; see CrossChunkStore. Family-wide; defaults to ``canonical``._

__



<div data-search-exclude markdown="1">



URI: [zv:store](https://w3id.org/zarr-vectors/schema/0.5/store)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LinksFamilyMeta](LinksFamilyMeta.md) | `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [CrossChunkStore](CrossChunkStore.md) |
| Domain Of | [LinksFamilyMeta](LinksFamilyMeta.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:store |
| native | zv:store |




## LinkML Source

<details>
```yaml
name: store
description: 'Cell-duplication policy for a links family; see CrossChunkStore. Family-wide;
  defaults to ``canonical``.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- LinksFamilyMeta
range: CrossChunkStore
required: false

```
</details></div>


---

---
search:
  boost: 1.0
---# Type: String 




_A character string_



<div data-search-exclude markdown="1">

URI: [xsd:string](http://www.w3.org/2001/XMLSchema#string)

## Type Properties

| Property | Value |
| --- | --- |
| Base | `str` |
| Type URI | [xsd:string](http://www.w3.org/2001/XMLSchema#string) |






## Notes

* In RDF serializations, a slot with range of string is treated as a literal or type xsd:string. If you are authoring schemas in LinkML YAML, the type is referenced with the lower case "string".



## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | xsd:string |
| native | zv:string |
| exact | schema:Text |




</div>


---

---
search:
  boost: 1.0
---# Type: Time 




_A time object represents a (local) time of day, independent of any particular day_



<div data-search-exclude markdown="1">

URI: [xsd:time](http://www.w3.org/2001/XMLSchema#time)

## Type Properties

| Property | Value |
| --- | --- |
| Base | `XSDTime` |
| Type URI | [xsd:time](http://www.w3.org/2001/XMLSchema#time) |
| Representation | `str` |






## Notes

* URI is dateTime because OWL reasoners do not work with straight date or time
* If you are authoring schemas in LinkML YAML, the type is referenced with the lower case "time".



## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | xsd:time |
| native | zv:time |
| exact | schema:Time |




</div>


---

---
search:
  boost: 5.0
---

# Slot: type 


_NGFF axis type — "space", "time", or "channel"._



<div data-search-exclude markdown="1">



URI: [zv:type](https://w3id.org/zarr-vectors/schema/0.5/type)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [Axis](Axis.md) | One axis of the spatial index |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [Axis](Axis.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:type |
| native | zv:type |




## LinkML Source

<details>
```yaml
name: type
description: NGFF axis type — "space", "time", or "channel".
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- Axis
range: string
required: true

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: unit 


_NGFF unit string (e.g. "um", "nanometer", "second")._



<div data-search-exclude markdown="1">



URI: [zv:unit](https://w3id.org/zarr-vectors/schema/0.5/unit)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [Axis](Axis.md) | One axis of the spatial index |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [Axis](Axis.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:unit |
| native | zv:unit |




## LinkML Source

<details>
```yaml
name: unit
description: NGFF unit string (e.g. "um", "nanometer", "second").
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- Axis
range: string

```
</details></div>


---

---
search:
  boost: 1.0
---# Type: Uri 




_a complete URI_



<div data-search-exclude markdown="1">

URI: [xsd:anyURI](http://www.w3.org/2001/XMLSchema#anyURI)

## Type Properties

| Property | Value |
| --- | --- |
| Base | `URI` |
| Type URI | [xsd:anyURI](http://www.w3.org/2001/XMLSchema#anyURI) |
| Representation | `str` |






## Comments

* in RDF serializations a slot with range of uri is treated as a literal or type xsd:anyURI unless it is an identifier or a reference to an identifier, in which case it is translated directly to a node

## Notes

* If you are authoring schemas in LinkML YAML, the type is referenced with the lower case "uri".



## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | xsd:anyURI |
| native | zv:uri |
| close | schema:URL |




</div>


---

---
search:
  boost: 1.0
---# Type: Uriorcurie 




_a URI or a CURIE_



<div data-search-exclude markdown="1">

URI: [xsd:anyURI](http://www.w3.org/2001/XMLSchema#anyURI)

## Type Properties

| Property | Value |
| --- | --- |
| Base | `URIorCURIE` |
| Type URI | [xsd:anyURI](http://www.w3.org/2001/XMLSchema#anyURI) |
| Representation | `str` |






## Notes

* If you are authoring schemas in LinkML YAML, the type is referenced with the lower case "uriorcurie".



## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | xsd:anyURI |
| native | zv:uriorcurie |




</div>


---

---
search:
  boost: 5.0
---

# Slot: vertex_count 


_Total number of vertices at this level._



<div data-search-exclude markdown="1">



URI: [zv:vertex_count](https://w3id.org/zarr-vectors/schema/0.5/vertex_count)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LevelMetadata](LevelMetadata.md) | Per-resolution-level `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Integer](Integer.md) |
| Domain Of | [LevelMetadata](LevelMetadata.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
### Value Constraints

| Property | Value |
| --- | --- |
| Minimum Value | 0 |












## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:vertex_count |
| native | zv:vertex_count |




## LinkML Source

<details>
```yaml
name: vertex_count
description: Total number of vertices at this level.
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- LevelMetadata
range: integer
required: true
minimum_value: 0

```
</details></div>


---

---
search:
  boost: 10.0
---

# Class: VerticesMeta 


_``.zattrs`` for the ``vertices/`` array._



<div data-search-exclude markdown="1">



URI: [zv:VerticesMeta](https://w3id.org/zarr-vectors/schema/0.5/VerticesMeta)





```mermaid
 classDiagram
    class VerticesMeta
    click VerticesMeta href "../VerticesMeta/"
      VerticesMeta : dtype
        
      VerticesMeta : encoding
        
          
    
        
        
        VerticesMeta --> "0..1" Encoding : encoding
        click Encoding href "../Encoding/"
    

        
      VerticesMeta : zv_array
        
          
    
        
        
        VerticesMeta --> "1" ZvArrayTag : zv_array
        click ZvArrayTag href "../ZvArrayTag/"
    

        
      
```




<!-- no inheritance hierarchy -->

## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [zv_array](zv_array.md) | 1 <br/> [ZvArrayTag](ZvArrayTag.md) | Discriminator slot identifying the kind of per-array `` | direct |
| [dtype](dtype.md) | 1 <br/> [String](String.md) | Numpy dtype string of the array's value type (e | direct |
| [encoding](encoding.md) | 0..1 <br/> [Encoding](Encoding.md) | How the chunk bytes are encoded | direct |















## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:VerticesMeta |
| native | zv:VerticesMeta |






## LinkML Source

### Direct

<details>
```yaml
name: VerticesMeta
description: '``.zattrs`` for the ``vertices/`` array.'
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slots:
- zv_array
- dtype
- encoding
slot_usage:
  zv_array:
    name: zv_array
    required: true
    equals_string: vertices

```
</details>

### Induced

<details>
```yaml
name: VerticesMeta
description: '``.zattrs`` for the ``vertices/`` array.'
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slot_usage:
  zv_array:
    name: zv_array
    required: true
    equals_string: vertices
attributes:
  zv_array:
    name: zv_array
    description: 'Discriminator slot identifying the kind of per-array ``.zattrs``
      block.  Each writer in ``core/arrays.py`` stamps the corresponding token from
      :class:`ZvArrayTag`.

      '
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: VerticesMeta
    domain_of:
    - VerticesMeta
    - LinksFamilyMeta
    - LinksMeta
    - AttributeMeta
    - FragmentAttributeMeta
    - ObjectIndexMeta
    - ObjectAttributeMeta
    - GroupingsMeta
    - GroupingsAttributeMeta
    - LinkAttributeFamilyMeta
    - LinkAttributeMeta
    range: ZvArrayTag
    required: true
    equals_string: vertices
  dtype:
    name: dtype
    description: Numpy dtype string of the array's value type (e.g. "float32").
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: VerticesMeta
    domain_of:
    - VerticesMeta
    - LinksMeta
    - AttributeMeta
    - FragmentAttributeMeta
    - ObjectAttributeMeta
    - GroupingsAttributeMeta
    - LinkAttributeMeta
    range: string
    required: true
  encoding:
    name: encoding
    description: How the chunk bytes are encoded.
    from_schema: https://w3id.org/zarr-vectors/schema/0.5
    rank: 1000
    owner: VerticesMeta
    domain_of:
    - VerticesMeta
    range: Encoding

```
</details></div>


---

# zarr_vectors 

Reference schema for the Zarr Vectors (ZV) on-disk metadata (0.5.0). Covers the root-level ``.zattrs`` block (``zarr_vectors`` envelope), the NGFF ``multiscales`` block (which now carries the canonical axes list — ZV no longer duplicates them under ``spatial_index_dims``), the per-resolution-level ``.zattrs`` block (``zarr_vectors_level`` envelope), and the per-array ``.zattrs`` shapes emitted by the writers in ``zarr_vectors/core/arrays.py`` (discriminator slot ``zv_array``).


URI: https://w3id.org/zarr-vectors/schema/0.5


---

---
search:
  boost: 5.0
---

# Slot: zv_array 


_Discriminator slot identifying the kind of per-array ``.zattrs`` block.  Each writer in ``core/arrays.py`` stamps the corresponding token from :class:`ZvArrayTag`._

__



<div data-search-exclude markdown="1">



URI: [zv:zv_array](https://w3id.org/zarr-vectors/schema/0.5/zv_array)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [VerticesMeta](VerticesMeta.md) | `` |  yes  |
| [LinksFamilyMeta](LinksFamilyMeta.md) | `` |  yes  |
| [LinksMeta](LinksMeta.md) | `` |  yes  |
| [AttributeMeta](AttributeMeta.md) | `` |  yes  |
| [FragmentAttributeMeta](FragmentAttributeMeta.md) | `` |  yes  |
| [ObjectIndexMeta](ObjectIndexMeta.md) | `` |  yes  |
| [ObjectAttributeMeta](ObjectAttributeMeta.md) | `` |  yes  |
| [GroupingsMeta](GroupingsMeta.md) | `` |  yes  |
| [GroupingsAttributeMeta](GroupingsAttributeMeta.md) | `` |  yes  |
| [LinkAttributeFamilyMeta](LinkAttributeFamilyMeta.md) | `` |  yes  |
| [LinkAttributeMeta](LinkAttributeMeta.md) | `` |  yes  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [ZvArrayTag](ZvArrayTag.md) |
| Domain Of | [VerticesMeta](VerticesMeta.md), [LinksFamilyMeta](LinksFamilyMeta.md), [LinksMeta](LinksMeta.md), [AttributeMeta](AttributeMeta.md), [FragmentAttributeMeta](FragmentAttributeMeta.md), [ObjectIndexMeta](ObjectIndexMeta.md), [ObjectAttributeMeta](ObjectAttributeMeta.md), [GroupingsMeta](GroupingsMeta.md), [GroupingsAttributeMeta](GroupingsAttributeMeta.md), [LinkAttributeFamilyMeta](LinkAttributeFamilyMeta.md), [LinkAttributeMeta](LinkAttributeMeta.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | zv:zv_array |
| native | zv:zv_array |




## LinkML Source

<details>
```yaml
name: zv_array
description: 'Discriminator slot identifying the kind of per-array ``.zattrs`` block.  Each
  writer in ``core/arrays.py`` stamps the corresponding token from :class:`ZvArrayTag`.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
domain_of:
- VerticesMeta
- LinksFamilyMeta
- LinksMeta
- AttributeMeta
- FragmentAttributeMeta
- ObjectIndexMeta
- ObjectAttributeMeta
- GroupingsMeta
- GroupingsAttributeMeta
- LinkAttributeFamilyMeta
- LinkAttributeMeta
range: ZvArrayTag
required: true

```
</details></div>


---

---
search:
  boost: 5.0
---

# Slot: zv_version 


_ZV spec version this store was written against (e.g. "0.5.0"). Renamed from ``format_version`` in 0.5.0 to disambiguate from Zarr v3's ``zarr_format`` field._

__



<div data-search-exclude markdown="1">



URI: [schema:version](http://schema.org/version)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [RootMetadata](RootMetadata.md) | Root-level `` |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [RootMetadata](RootMetadata.md) |
| Slot URI | [schema:version](http://schema.org/version) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
### Value Constraints

| Property | Value |
| --- | --- |
| Regex Pattern | `^\d+\.\d+(\.\d+)?$` |












## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | schema:version |
| native | zv:zv_version |




## LinkML Source

<details>
```yaml
name: zv_version
description: 'ZV spec version this store was written against (e.g. "0.5.0"). Renamed
  from ``format_version`` in 0.5.0 to disambiguate from Zarr v3''s ``zarr_format``
  field.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
slot_uri: schema:version
domain_of:
- RootMetadata
range: string
required: true
pattern: ^\d+\.\d+(\.\d+)?$

```
</details></div>


---

---
search:
  boost: 2.0
---


# Enum: ZvArrayTag 




_Discriminator value for per-array ``.zattrs`` blocks (slot ``zv_array``).  Each writer in ``zarr_vectors/core/arrays.py`` stamps the corresponding tag._

__



<div data-search-exclude markdown="1">

URI: [zv:ZvArrayTag](https://w3id.org/zarr-vectors/schema/0.5/ZvArrayTag)

## Permissible Values
| Value | Meaning | Description |
| --- | --- | --- |
| vertices | None |  |
| vertex_fragments | None | Per-chunk fragment-index group for ``vertices/<chunk>`` (v0 |
| link_fragments | None | Per-chunk fragment-index group for the intra-chunk link array at ``links/0/<a... |
| links_family | None | The ``links/<delta>/`` group carrying family-wide policy for every offsets ar... |
| links | None | One ``links/<delta>/<offsets>/`` array — a rank-D vlen array whose cell is th... |
| attribute | None |  |
| fragment_attribute | None | Per-fragment attribute array (``fragment_attributes/<name>/<chunk>``) |
| object_index | None |  |
| object_attribute | None |  |
| groupings | None |  |
| groupings_attribute | None |  |
| link_attribute_family | None | The ``link_attributes/<name>/<delta>/`` group |
| link_attribute | None | One ``link_attributes/<name>/<delta>/<offsets>/`` array, mirroring the matchi... |




## Slots

| Name | Description |
| ---  | --- |
| [zv_array](zv_array.md) | Discriminator slot identifying the kind of per-array `` |










## Identifier and Mapping Information





### Schema Source


* from schema: https://w3id.org/zarr-vectors/schema/0.5






## LinkML Source

<details>
```yaml
name: ZvArrayTag
description: 'Discriminator value for per-array ``.zattrs`` blocks (slot ``zv_array``).  Each
  writer in ``zarr_vectors/core/arrays.py`` stamps the corresponding tag.

  '
from_schema: https://w3id.org/zarr-vectors/schema/0.5
rank: 1000
permissible_values:
  vertices:
    text: vertices
  vertex_fragments:
    text: vertex_fragments
    description: 'Per-chunk fragment-index group for ``vertices/<chunk>`` (v0.6+).
      Replaces the v0.5 ``vertex_group_offsets``.

      '
  link_fragments:
    text: link_fragments
    description: 'Per-chunk fragment-index group for the intra-chunk link array at
      ``links/0/<all-zero offsets>/<chunk>``.  Keyed by chunk ALONE — it carries no
      delta and no offsets segment — so exactly one array may write it.  Every other
      offsets segment, and every ``delta != 0`` array, uses an inline self-describing
      blob and has no sidecar.

      '
  links_family:
    text: links_family
    description: 'The ``links/<delta>/`` group carrying family-wide policy for every
      offsets array beneath it.  See ``LinksFamilyMeta``.

      '
  links:
    text: links
    description: 'One ``links/<delta>/<offsets>/`` array — a rank-D vlen array whose
      cell is the record''s SOURCE chunk.  See ``LinksMeta``.

      '
  attribute:
    text: attribute
  fragment_attribute:
    text: fragment_attribute
    description: 'Per-fragment attribute array (``fragment_attributes/<name>/<chunk>``).
      Dense per-chunk byte blob with one row per fragment in that chunk; row count
      is derived from ``vertex_fragments/<chunk>``. Optional; common opt-in use case
      is materializing parent-IDs (e.g. the OID owning each fragment) as a fragment
      attribute.

      '
  object_index:
    text: object_index
  object_attribute:
    text: object_attribute
  groupings:
    text: groupings
  groupings_attribute:
    text: groupings_attribute
  link_attribute_family:
    text: link_attribute_family
    description: 'The ``link_attributes/<name>/<delta>/`` group.  See ``LinkAttributeFamilyMeta``.

      '
  link_attribute:
    text: link_attribute
    description: 'One ``link_attributes/<name>/<delta>/<offsets>/`` array, mirroring
      the matching links array cell-for-cell.  See ``LinkAttributeMeta``.

      '

```
</details>

</div>