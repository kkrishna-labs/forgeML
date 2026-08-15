"""Model registry: promotion, aliasing and champion lookup."""

from __future__ import annotations

from forgeml.registry.model_registry import (
    RegisteredChampion,
    get_champion_uri,
    promote_alias,
    register_champion,
)

__all__ = [
    "RegisteredChampion",
    "get_champion_uri",
    "promote_alias",
    "register_champion",
]
