"""Compatibility import for the shared source-cache helper.

The implementation lives at ``code/behavioural-simulations/source_cache.py``
so standalone profile builders can use it without importing the heavy
``profiles`` package initializer. Load that file by path so this shim remains
safe even when ``profiles/`` is first on ``sys.path``.
"""

import importlib.util
from pathlib import Path

_shared_path = Path(__file__).resolve().parents[1] / "source_cache.py"
_spec = importlib.util.spec_from_file_location(
    "_behavioural_source_cache", _shared_path
)
if _spec is None or _spec.loader is None:  # pragma: no cover - broken checkout
    raise ImportError(f"cannot load shared source cache from {_shared_path}")
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)

cache_path = _shared.cache_path
cached_file = _shared.cached_file

__all__ = ["cache_path", "cached_file"]
