
transform_registry = {}


def register_transform(name):
    def decorator(cls):
        if name in transform_registry:
            raise ValueError(f'{name} already registered!')
        transform_registry[name] = cls
        cls._registry_name = name
        return cls
    return decorator


def available_transforms():
    return sorted(transform_registry.keys())


def create_transform(name: str, **kwargs):
    try:
        return transform_registry[name](**kwargs)
    except KeyError:
        raise KeyError(
            f"Transform '{name}' not found. "
            f"Available: {', '.join(available_transforms()) or '<empty>'}"
        )
