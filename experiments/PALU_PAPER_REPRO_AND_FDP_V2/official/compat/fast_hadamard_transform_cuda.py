"""Pure-PyTorch compatibility shim for the optional PaLu CUDA extension.

The official PaLu import path requires this module even when the selected
decomposition never calls a Hadamard kernel.  The formal reproduction uses
the official whiten decomposition, so this shim is only an import/runtime
fallback; it does not alter the rank search or decomposition path.
"""

from __future__ import annotations

import torch


def _hadamard(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    original_shape = x.shape
    n = original_shape[-1]
    padded_n = 1 << (n - 1).bit_length()
    y = x.reshape(-1, n)
    if padded_n != n:
        y = torch.nn.functional.pad(y, (0, padded_n - n))
    h = y
    width = 1
    while width < padded_n:
        h = h.reshape(-1, padded_n // (2 * width), 2, width)
        a = h[:, :, 0, :].clone()
        b = h[:, :, 1, :].clone()
        h[:, :, 0, :] = a + b
        h[:, :, 1, :] = a - b
        h = h.reshape(-1, padded_n)
        width *= 2
    return (h[:, :n] * scale).reshape(original_shape)


def fast_hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _hadamard(x, scale)


def fast_hadamard_transform_12N(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _hadamard(x, scale)


def fast_hadamard_transform_20N(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _hadamard(x, scale)


def fast_hadamard_transform_28N(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _hadamard(x, scale)


def fast_hadamard_transform_40N(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _hadamard(x, scale)
