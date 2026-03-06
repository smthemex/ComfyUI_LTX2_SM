"""Latent upsampler model components."""

from ..upsampler.model import LatentUpsampler, upsample_video
from ..upsampler.model_configurator import LatentUpsamplerConfigurator

__all__ = [
    "LatentUpsampler",
    "LatentUpsamplerConfigurator",
    "upsample_video",
]
