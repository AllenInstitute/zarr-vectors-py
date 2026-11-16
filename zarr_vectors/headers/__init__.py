"""Format-specific header preservation for zarr vectors stores.

Headers are stored in the attributes of ``/headers/<format>/`` as raw
JSON-compatible dicts.  The registry provides opaque round-trip
storage; typed (de)serialisation of header dicts lives in the
format package.
"""

from zarr_vectors.headers.registry import HeaderRegistry

__all__ = ["HeaderRegistry"]
