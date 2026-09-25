"""Compatibility import for the shared policy-corpus helpers.

The implementation lives at ``code/behavioural-simulations/policy_store.py``
so standalone profile builders can import it without loading the heavy
``profiles`` package initializer. Load that file by path rather than using an
absolute import: this shim can itself be first on ``sys.path``.
"""

import importlib.util
from pathlib import Path

_shared_path = Path(__file__).resolve().parents[1] / "policy_store.py"
_spec = importlib.util.spec_from_file_location(
    "_behavioural_policy_store", _shared_path
)
if _spec is None or _spec.loader is None:  # pragma: no cover - broken checkout
    raise ImportError(f"cannot load shared policy store from {_shared_path}")
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)

DEFAULT_SEARCH_FIELDS = _shared.DEFAULT_SEARCH_FIELDS
DEFAULT_STOPWORDS = _shared.DEFAULT_STOPWORDS
ENTRY_KEYS = _shared.ENTRY_KEYS
chunk_units = _shared.chunk_units
load_policy = _shared.load_policy
search_entries = _shared.search_entries
write_json = _shared.write_json

__all__ = [
    "DEFAULT_SEARCH_FIELDS",
    "DEFAULT_STOPWORDS",
    "ENTRY_KEYS",
    "chunk_units",
    "load_policy",
    "search_entries",
    "write_json",
]
