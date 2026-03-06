import logging
from collections.abc import Iterator

import torch

from ..ltx_core.components.diffusion_steps import EulerDiffusionStep
from ..ltx_core.components.guiders import CFGGuider
from ..ltx_core.components.noisers import GaussianNoiser
from ..ltx_core.components.protocols import DiffusionStepProtocol
from ..ltx_core.components.schedulers import LTX2Scheduler
from ..ltx_core.loader import LoraPathStrengthAndSDOps
from ..ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ..ltx_core.model.video_vae import decode_video as vae_decode_video
from ..ltx_core.text_encoders.gemma import encode_text
from ..ltx_core.types import LatentState, VideoPixelShape
from ..ltx_pipelines.utils import ModelLedger
from ..ltx_pipelines.utils.args import default_1_stage_arg_parser
from ..ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
from ..ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    denoise_audio_video,
    euler_denoising_loop,
    generate_enhanced_prompt,
    get_device,
    guider_denoising_func,
    image_conditionings_by_replacing_latent,
)
from ..ltx_pipelines.utils.media_io import encode_video
from ..ltx_pipelines.utils.types import PipelineComponents

#device = get_device()


class TI2VidOneStagePipeline:
    """
    Single-stage text/image-to-video generation pipeline.
    Generates video at the target resolution in a single diffusion pass with
    classifier-free guidance (CFG). Supports optional image conditioning via
    the images parameter.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device = "cpu",
        fp8transformer: bool = False,
        gguf_dit: bool = False,
    ):
        self.dtype = torch.bfloat16
        self.device = device
        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            loras=loras,
            fp8transformer=fp8transformer,
            gguf_dit=gguf_dit,
            load_mode= "dit",
        )
        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )

    def load_dit(self,load=True,):
        if load:
            self.transformer=self.model_ledger.transformer()
        else:
            self.transformer=None
            del self.transformer
            cleanup_memory()

    def __call__(  # noqa: PLR0913
        self,
        prompt,
        negative_prompt,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        cfg_guidance_scale: float,
        images,#: list[tuple[str, int, float]],
        enhance_prompt: bool = False,
        context_p=None,
        context_n=None,
        offload=True,
       
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        assert_resolution(height=height, width=width, is_two_stage=False)
        cur_device = self.device if not offload else "cuda"
        generator = torch.Generator(device=cur_device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        if context_n is None:
            cfg_guidance_scale=1.0
        cfg_guider = CFGGuider(cfg_guidance_scale)
        dtype = torch.bfloat16

        # text_encoder = self.model_ledger.text_encoder()
        # if enhance_prompt:
        #     prompt = generate_enhanced_prompt(
        #         text_encoder, prompt, images[0][0] if len(images) > 0 else None, seed=seed
        #     )
        # context_p, context_n = encode_text(text_encoder, prompts=[prompt, negative_prompt])
        v_context_p, a_context_p = context_p
        if context_n is None:
            v_context_n = None
            a_context_n = None
        else:
            v_context_n, a_context_n = context_n

        torch.cuda.synchronize()
      
        cleanup_memory()

        # Stage 1: Initial low resolution video generation.
       
        #transformer = self.model_ledger.transformer()

        if offload:
            from ..ltx_core.model.transformer.model import BlockGPUManager         
            gpu_manager = BlockGPUManager(device="cuda", )
            gpu_manager.setup_for_inference(self.transformer.velocity_model)
        else:
            gpu_manager = None
        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=cur_device)

        def first_stage_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=guider_denoising_func(
                    cfg_guider,
                    v_context_p,
                    v_context_n,
                    a_context_p,
                    a_context_n,
                    transformer=self.transformer,
                    gpu_manager=gpu_manager  # noqa: F821
                ),
                gpu_manager=gpu_manager,
            )

        stage_1_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        # stage_1_conditionings = image_conditionings_by_replacing_latent(
        #     images=images,
        #     height=stage_1_output_shape.height,
        #     width=stage_1_output_shape.width,
        #     video_encoder=video_encoder,
        #     dtype=dtype,
        #     device=cur_device,
        # )

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=images[0] if images else [],
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=first_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=cur_device,
        )

        torch.cuda.synchronize()
        if gpu_manager is not None:
            gpu_manager.unload_blocks_to_cpu()
        #del transformer
        cleanup_memory()


        return video_state.latent, audio_state.latent
    
def load_pipeline_ltx_one_stage(
    checkpoint_path: str,
    gemma_root: str,
    loras: list[str],
    device="cpu" ,
    fp8transformer: bool=False,
    gguf_dit=False,
) -> TI2VidOneStagePipeline:
    
    pipeline = TI2VidOneStagePipeline(
        checkpoint_path=checkpoint_path,
        gemma_root=gemma_root,
        loras=loras,
        device=device,
        fp8transformer=fp8transformer,
        gguf_dit=gguf_dit
    )
    pipeline.load_dit()
    return pipeline



@torch.inference_mode()
def inference_ltx_one_stage(
    pipeline,
    context_p,
    context_n,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: int,
    num_inference_steps: int,
    cfg_guidance_scale: float,
    images,
    offload=True,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    video, audio = pipeline(
        prompt=None,
        negative_prompt=None,
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        num_inference_steps=num_inference_steps,
        cfg_guidance_scale=cfg_guidance_scale,
        images=images,
        context_p=context_p,
        context_n=context_n,
        offload=offload,
    )

    return video, audio


# @torch.inference_mode()
# def main() -> None:
#     logging.getLogger().setLevel(logging.INFO)
#     parser = default_1_stage_arg_parser()
#     args = parser.parse_args()
#     pipeline = TI2VidOneStagePipeline(
#         checkpoint_path=args.checkpoint_path,
#         gemma_root=args.gemma_root,
#         loras=args.lora,
#         fp8transformer=args.enable_fp8,
#     )
#     video, audio = pipeline(
#         prompt=args.prompt,
#         negative_prompt=args.negative_prompt,
#         seed=args.seed,
#         height=args.height,
#         width=args.width,
#         num_frames=args.num_frames,
#         frame_rate=args.frame_rate,
#         num_inference_steps=args.num_inference_steps,
#         cfg_guidance_scale=args.cfg_guidance_scale,
#         images=args.images,
#     )

#     encode_video(
#         video=video,
#         fps=args.frame_rate,
#         audio=audio,
#         audio_sample_rate=AUDIO_SAMPLE_RATE,
#         output_path=args.output_path,
#         video_chunks_number=1,
#     )


# if __name__ == "__main__":
#     main()
