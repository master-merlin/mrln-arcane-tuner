"""SDXL model family registration."""

from app.engine.core.definitions import ModelFamily
from .trainer import SDXLTrainer


class SDXLFamily(ModelFamily):
    """SDXL implementation of the ModelFamily logic provider."""

    family_name = "sdxl"
    archetype = "latent_diffusion"
    # supports_min_snr: trainer.py compute_loss_weight reads min_snr_gamma (the one family that does);
    # supports_flow_timestep_sampling False: trainer.py sample_timesteps draws a discrete uniform
    # DDPM timestep and reads none of the flow-matching timestep fields (the user's decision, LANE-168).
    capability_overrides = {
        "supports_train_te": True,
        "supports_min_snr": True,
        "supports_flow_timestep_sampling": False,
    }

    def get_trainer_class(self):
        return SDXLTrainer
