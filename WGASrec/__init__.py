# wgasrec/__init__.py
__version__ = "0.1.0"

from .utils.registry import register, get

def build_model(name="WGASRec", **cfg):
    return get("model", name)(**cfg)

def build_dataset(name, *args, **kwargs):
    return get("dataset", name)(*args, **kwargs)

__all__ = ["register", "get", "build_model", "build_dataset", "__version__"]
