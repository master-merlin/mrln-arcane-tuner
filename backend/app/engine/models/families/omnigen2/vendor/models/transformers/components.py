# Vendored from VectorSpaceLab/OmniGen2 @ 18e6f9d5271b517fcb32e999f10df943ae9b8f20 (2026-07-13)
# Source: omnigen2/models/transformers/components.py
# Apache-2.0 — vendored for the omnigen2 family (local diffusers 0.39.0).
import torch.nn.functional as F


def swiglu(x, y):
    return F.silu(x.float(), inplace=False).to(x.dtype) * y
