REGISTRY = {}
def register(kind, name):
    def deco(fn):
        REGISTRY.setdefault(kind, {})[name.lower()] = fn
        return fn
    return deco
def get(kind, name):
    return REGISTRY[kind][name.lower()]
