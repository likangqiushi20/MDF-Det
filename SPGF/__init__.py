"""Leakage-controlled, native-resolution SPGF scene-prior module."""

from .model import GroupNormalization, build_spgf_v4

__all__ = ["GroupNormalization", "build_spgf_v4"]
