import importlib


def import_str(string: str):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)
