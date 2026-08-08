from importlib import import_module as _import_module

_riichi = _import_module("riichi")
for _name in dir(_riichi):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_riichi, _name)
__all__ = [n for n in globals() if not n.startswith("_")]
