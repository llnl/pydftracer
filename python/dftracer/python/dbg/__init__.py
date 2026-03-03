import importlib as _importlib

from dftracer.python.dbg.ai import *
from dftracer.python.dbg.logger import *

_DYNAMO_EXPORTS = {"Dynamo", "dynamo", "create_backend"}


def __getattr__(name: str) -> object:
    if name in _DYNAMO_EXPORTS:
        dynamo_module = _importlib.import_module("dftracer.python.dynamo")
        value = getattr(dynamo_module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [name for name in globals() if not name.startswith("_")]
for _name in _DYNAMO_EXPORTS:
    if _name not in __all__:
        __all__.append(_name)
