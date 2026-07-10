"""Storage-backend selection for Zarr v3 stores.

Every backend zarr-vectors supports returns a plain
``zarr.abc.store.Store`` built by its own native constructor — there is
no custom byte-level abstraction layer.  This package only decides
*which* backend a URL should use; the store objects themselves are
built in :mod:`zarr_vectors.core.store` (LocalStore / FsspecStore /
obstore-backed ObjectStore) and :mod:`icechunk_backend` (icechunk).

Selection order (``icechunk`` is always explicit, never auto-detected):

1. Explicit ``backend=`` kwarg on the public API.
2. ``ZARR_VECTORS_BACKEND`` environment variable.
3. URL-scheme auto-detect:
   - no scheme or ``file://`` → ``local``
   - cloud schemes (``s3``, ``gs``, ``gcs``, ``az``, ``azure``, ``abfs``,
     ``http``, ``https``) → ``obstore`` if installed, else ``fsspec``,
     else a :class:`~zarr_vectors.exceptions.StoreError` with an install
     hint.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from zarr_vectors.exceptions import StoreError

SCHEMES_LOCAL = frozenset({"", "file"})
SCHEMES_OBJECT_STORE = frozenset(
    {"s3", "gs", "gcs", "az", "azure", "abfs", "http", "https"}
)

_ENV_VAR = "ZARR_VECTORS_BACKEND"

__all__ = [
    "SCHEMES_LOCAL",
    "SCHEMES_OBJECT_STORE",
    "detect_scheme",
    "resolve_backend_name",
]


def detect_scheme(url: str | Path) -> str:
    """Return the URL scheme of ``url``, lowercased; empty string if none.

    A bare Windows drive letter (``C:\\foo``) is treated as local
    (returns ``""``), not as the scheme ``c``.
    """
    if isinstance(url, Path):
        return ""
    if not isinstance(url, str):
        return ""
    # urlparse misreads ``C:\foo`` as scheme=='c'.  Reject single-letter
    # schemes — no real scheme is a single character.
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if len(scheme) <= 1:
        return ""
    return scheme


def resolve_backend_name(
    url: str | Path,
    explicit: str | None = None,
    *,
    env_override: str | None = None,
) -> str:
    """Decide which backend to use for ``url``.

    Args:
        url: The store URL or path.
        explicit: User-supplied ``backend=`` kwarg.  Wins if set.
        env_override: Override for the ``ZARR_VECTORS_BACKEND`` env var
            (for testing).  Pass ``""`` to ignore the env var entirely.

    Returns:
        One of ``"local"``, ``"obstore"``, ``"fsspec"``.

    Raises:
        StoreError: If a cloud scheme is given but no compatible backend
            is installed.
    """
    if explicit:
        return explicit.lower()
    env_val = env_override if env_override is not None else os.environ.get(_ENV_VAR)
    if env_val:
        return env_val.lower()

    scheme = detect_scheme(url)
    if scheme in SCHEMES_LOCAL:
        return "local"
    if scheme in SCHEMES_OBJECT_STORE:
        if _have("obstore"):
            return "obstore"
        if _have("fsspec"):
            return "fsspec"
        raise StoreError(
            f"URL {url!r} has scheme {scheme!r} which requires a cloud "
            f"backend, but neither 'obstore' nor 'fsspec' is installed. "
            f"Install with: pip install zarr-vectors[obstore]"
        )
    # Unknown scheme — let local handle it; if it's broken, the backend
    # constructor will raise something more specific.
    return "local"


def _have(module: str) -> bool:
    """Return True if ``module`` is importable.

    Honours ``sys.modules`` overrides used in tests — a sentinel of
    ``None`` indicates "not available", and a stub module object
    (regardless of whether it has a real ``__spec__``) indicates
    "available".
    """
    import importlib.util
    import sys

    if module in sys.modules:
        return sys.modules[module] is not None
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False
