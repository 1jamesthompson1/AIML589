"""OpenRouter app attribution shared by all repository workflows.

OpenRouter groups usage and cost by the optional ``HTTP-Referer`` and
``X-OpenRouter-Title`` headers. Keep workflow-specific titles here so all
OpenRouter clients use the same, auditable attribution.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from inspect_ai.model import GenerateConfig


def openrouter_headers(app_name: str, referer: str) -> dict[str, str]:
    """Build attribution headers for one named OpenRouter app."""
    return {
        "HTTP-Referer": referer,
        "X-OpenRouter-Title": app_name,
    }


SIMULATIONS_HEADERS = openrouter_headers(
    "Simulations", "https://nz-llm.sjhl.nz/agents/"
)
COMPARISONS_HEADERS = openrouter_headers(
    "Simulations Comparisons", "https://nz-llm.sjhl.nz/results-viewer/"
)
WVS_VALUE_MAP_HEADERS = openrouter_headers(
    "WVS Value Map", "https://nz-llm.sjhl.nz/results-viewer/?tab=evals"
)


def headers_for_provider(
    provider: str | None, headers: Mapping[str, str]
) -> dict[str, str] | None:
    """Return headers for OpenRouter, or ``None`` for another provider."""
    return dict(headers) if provider and provider.casefold() == "openrouter" else None


def is_openrouter_model(model: object) -> bool:
    """Whether a model id or Inspect model is backed by OpenRouter.

    Inspect strips the ``openrouter/`` prefix from ``Model.name``, so checking
    only the display name misses actual OpenRouter models. The underlying API
    service and base URL provide a reliable fallback.
    """
    if isinstance(model, str):
        return model.casefold().startswith("openrouter/")

    name = str(getattr(model, "name", ""))
    if name.casefold().startswith("openrouter/"):
        return True

    api = getattr(model, "api", None)
    if str(getattr(api, "service", "")).casefold() == "openrouter":
        return True
    hostname = urlsplit(str(getattr(api, "base_url", ""))).hostname
    return bool(hostname and hostname.casefold() == "openrouter.ai")


def attributed_config(
    model: object,
    config: GenerateConfig,
    headers: Mapping[str, str],
) -> GenerateConfig:
    """Attach app headers to OpenRouter generation, leaving other providers."""
    if not is_openrouter_model(model):
        return config
    merged_headers = dict(config.extra_headers or {})
    merged_headers.update(headers)
    return config.merge({"extra_headers": merged_headers})
