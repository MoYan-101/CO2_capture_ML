from __future__ import annotations


def build_layered_datasets(*args, **kwargs):
    from .prepare_co2_capture_datasets import build_layered_datasets as _build_layered_datasets

    return _build_layered_datasets(*args, **kwargs)


__all__ = ["build_layered_datasets"]
