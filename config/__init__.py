# config/__init__.py

# for another profile change here
from .default import Config as _Config

cfg = _Config()

__all__ = [
    "cfg",
]

