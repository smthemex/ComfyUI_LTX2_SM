
from comfy_api.latest import ComfyExtension,io
from typing_extensions import override

from .LTX2_node import LTX2_SM_Model, LTX2_SM_VAE, LTX2_SM_Clip, LTX2_SM_AUDIO_VAE, LTX2_DECO_VIDEO, LTX2_DECO_AUDIO, LTX2_LATENTS, LTX2_SM_ENCODER, LTX2_SM_KSampler
class  LTX2_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LTX2_SM_Model,
            LTX2_SM_VAE,
            LTX2_SM_Clip,
            LTX2_SM_AUDIO_VAE,
            LTX2_DECO_VIDEO,
            LTX2_DECO_AUDIO,
            LTX2_LATENTS,
            LTX2_SM_ENCODER,
            LTX2_SM_KSampler,
        ]
async def comfy_entrypoint() -> LTX2_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return LTX2_SM_Extension()