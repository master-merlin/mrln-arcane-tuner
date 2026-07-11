# Vendored from boogu-project/Boogu-Image @ ac9e40c1350fd60c502137a678ad1001d51e2ae7 (2026-07-10)
# Source: boogu/models/transformers/components.py
# vendored for boogu_image family — local diffusers 0.39.0

import torch.nn.functional as F


def swiglu(x, y):
    return F.silu(x.float(), inplace=False).to(x.dtype) * y
