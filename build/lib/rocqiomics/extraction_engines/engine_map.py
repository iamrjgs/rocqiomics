
ALLOWED_ENGINES = [
    'pyradiomics',
    'fastrad'
]

def import_error_message(name):
    return (
        f"{name} engine requested but not available.\n"
        f"Please install it (e.g., `pip install {name}`) to use this engine."
    )

def MAP_ENGINE(name):
    name = name.lower()

    if name not in ALLOWED_ENGINES:
        raise ValueError(
                f"Engine must be one of: {ALLOWED_ENGINES}. "
                f"Value provided: {name}"
            )

    if name == "pyradiomics":
        try:
            from .pyradiomics import PyradiomicsExtractor
            return PyradiomicsExtractor
        except ImportError as e:
            raise ImportError(import_error_message(name)) from e

    if name == "fastrad":
        try:
            from .fastrad import FastradExtractor
            return FastradExtractor
        except ImportError as e:
            raise ImportError(import_error_message(name)) from e