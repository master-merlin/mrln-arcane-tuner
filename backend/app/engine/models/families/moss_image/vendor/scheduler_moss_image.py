# Copyright 2026 OpenMOSS Team. Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MRLN-VENDOR (LANE-165): copied from the HuggingFace repository OpenMOSS-Team/MOSS-Image-Base at
# revision 1ceede80badddd6a7812069619edaa5633587192, file `pipeline_moss_image.py` lines 264-287
# (Apache-2.0, model card). Byte-identical in OpenMOSS-Team/MOSS-Image-SFT at
# fc5de0d56fc910cbca6f6625a70f5f582b26deb1. The installed diffusers release (0.41.0) ships no
# MOSS-Image scheduler (its shift-then-snap-to-the-1000-step-training-grid schedule is MOSS's own).
# What changed from upstream: only `MOSSImageScheduler` is kept, and the import block is reduced to
# the two names it uses. The class body is byte-for-byte upstream.
# Retire path: delete this file when a diffusers release ships a MOSS-Image scheduler
# (row in families/VENDOR_INVENTORY.md).
from __future__ import annotations

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.configuration_utils import register_to_config


class MOSSImageScheduler(FlowMatchEulerDiscreteScheduler):
    """Euler integration using the training project's V2, nearest-grid schedule."""
    @register_to_config
    def __init__(self, num_train_timesteps=1000, shift=5.0, training_shift=1.0):
        super().__init__(num_train_timesteps=num_train_timesteps, shift=1.0)
        self.register_to_config(shift=shift, training_shift=training_shift)

    def set_timesteps(self, num_inference_steps=50, device=None):
        if num_inference_steps < 1 or self.config.shift <= 0 or self.config.training_shift <= 0:
            raise ValueError('Steps and shifts must be positive.')
        total = self.config.num_train_timesteps
        def grid(count, shift):
            lower = shift / (total + shift - 1)
            values = torch.linspace(1, lower, count)
            return shift * values / (1 + (shift - 1) * values)
        sigmas = grid(num_inference_steps, self.config.shift)
        training = grid(total, self.config.training_shift).flip(0)
        upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
        lower = (upper - 1).clamp(min=0)
        nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
        sigmas = torch.unique_consecutive(training[nearest])
        # Our sigmas have already been shifted and snapped. Avoid a second shift.
        self._shift = 1.0
        super().set_timesteps(device=device, sigmas=sigmas.tolist())
