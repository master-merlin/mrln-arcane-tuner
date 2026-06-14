"""LTX 2.3 (Lightricks) video-LoRA family.

Joint audio + video latent-diffusion DiT (``LTX2VideoTransformer3DModel``)
with a Gemma3 text encoder routed through ``LTX2TextConnectors``.  T2V and
I2V are a single model: image-to-video is expressed via per-frame conditioning
(a conditioned first frame at timestep ~0), not a separate checkpoint.

Audio is OPTIONAL.  With ``train_audio=False`` the audio VAE / vocoder are
never loaded and the audio LoRA modules are excluded.  With ``train_audio=True``
clips that lack audio contribute a ZERO audio-loss term (mask=0) so the model
is never trained on silence-as-target.
"""
