"""Custom exceptions for the zarr-vectors package.

All exceptions inherit from :class:`ZVError` so callers can catch broadly
or narrowly as needed.

``ZVFError`` is kept as a deprecated alias for one release for code that
imported the old name.  Subclasses still inherit from :class:`ZVError`
directly.
"""


class ZVError(Exception):
    """Base exception for all zarr-vectors errors."""


class StoreError(ZVError):
    """Raised when a ZV store cannot be created, opened, or is structurally invalid."""


class MetadataError(ZVError):
    """Raised when metadata is missing, malformed, or fails schema validation."""


class ArrayError(ZVError):
    """Raised when an array has wrong shape, dtype, or cannot be read/written."""


class ChunkingError(ZVError):
    """Raised when spatial chunk assignment fails or produces invalid results."""


class ConventionError(ZVError):
    """Raised when convention flags are inconsistent with the store contents.

    For example, using object_index_convention='identity' with multiple
    spatial chunks, or links_convention='implicit_sequential' on a mesh.
    """


class ValidationError(ZVError):
    """Raised when a store fails conformance or consistency validation."""


class IngestError(ZVError):
    """Raised when reading from an external format fails during ingest."""


class ExportError(ZVError):
    """Raised when writing to an external format fails during export."""


class CoarseningError(ZVError):
    """Raised when multi-resolution pyramid construction fails."""


class DracoError(ZVError):
    """Raised when Draco encoding or decoding fails."""


class EditError(ZVError):
    """Raised when an edit cannot be applied: unresolved reference,
    invalid option combination, or a constraint violation in the
    change-set engine (e.g. ``refresh_pyramid`` outside the accepted
    set ``{True, False, "batch"}``).
    """


class ConcurrentEditError(ZVError):
    """Raised when the edit API cannot guarantee single-writer
    semantics on the current backend.  Edit on shared datasets is
    supported only under icechunk; non-transactional backends
    (``LocalStore``, plain S3) raise this when a session is opened with
    ``concurrent_writers=True``.
    """


# Deprecated alias.  Kept for one release for code that imported the old
# name; remove in 0.4.  Aliasing the class object means ``isinstance``
# checks against either name keep working.
ZVFError = ZVError
