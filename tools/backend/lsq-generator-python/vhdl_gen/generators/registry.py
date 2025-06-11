"""
Global registry for VHDL generator "instances."

1. Build the VHDL generator class.
2. Call register(gen) to store it.
3. Later, call get("generator's suffix") wherever you need the same instance
   (e.g. when emitting instantiation code).
"""

from __future__ import annotations

from typing import Dict
from vhdl_gen.generators.base import BaseVHDLGenerator


# Internal singleton map: suffix → generator instance
_generators: Dict[str, BaseVHDLGenerator] = {}


def register_registry(gen: BaseVHDLGenerator) -> None:
    """
    Register a fully configured generator instance.

    Raises ValueError if another generator with the same "suffix" is already present.
    """
    key = gen.suffix
    if key in _generators:
        raise ValueError(f"Generator '{key}' is already registered")
    _generators[key] = gen


def get_registry(key: str) -> BaseVHDLGenerator:
    """
    Retrieve the generator instance by its "suffix" attribute.

    Returns:
        BaseVHDLGenerator, the previously registered instance.
    """
    return _generators[key]


def unregister_registry(key: str) -> None:
    """
    Remove a generator from the registry.
    """
    _generators.pop(key, None)

