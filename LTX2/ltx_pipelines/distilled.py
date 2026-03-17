import logging
from collections.abc import Iterator
import torch

from ..ltx_core.components.diffusion_steps import EulerDiffusionStep
from ..ltx_core.components.noisers import GaussianNoiser
from ..ltx_core.components.protocols import DiffusionStepProtocol
from ..ltx_core.loader import LoraPathStrengthAndSDOps
from ..ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ..ltx_core.types import LatentState, VideoPixelShape
from ..ltx_pipelines.utils import ModelLedger,euler_denoising_loop
from .utils.args import (
    ImageConditioningInput,
    default_2_stage_distilled_arg_parser,
    detect_checkpoint_path,
)
from ..ltx_core.quantization import QuantizationPolicy
from ..ltx_pipelines.utils.constants import (
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    detect_params,
)
from ..ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    combined_image_conditionings,
    denoise_audio_video,
    encode_prompts,
    get_device,
    simple_denoising_func,
)
from ..ltx_pipelines.utils.types import PipelineComponents

device = get_device()


class DistilledPipeline:
    """
    Two-stage distilled video generation pipeline.
    Stage 1 generates video at the target resolution, then Stage 2 upsamples
    by 2x and refines with additional denoising steps for higher quality output.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        spatial_upsampler_path: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device = "cpu",
        quantization: QuantizationPolicy | None = None,
        gguf_dit: bool = False,
    ):
        self.device = device
        self.dtype = torch.bfloat16

        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root_path=gemma_root,
            loras=loras,
            quantization=quantization,
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

    def __call__(
            self,
            prompt: str,
            seed: int,
            height: int,
            width: int,
            num_frames: int,
            frame_rate: float,
            images: list[tuple[str, int, float]],
            tiling_config: TilingConfig | None = None,
            enhance_prompt: bool = False,
            context_p=None,
            offload=True,
            stage_one=True,
            v_lat=None,
            a_lat=None,
            ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        assert_resolution(height=height, width=width, is_two_stage=True)
        cur_device = self.device if not offload else "cuda"
        
        generator = torch.Generator(device=cur_device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16

        video_context, audio_context = context_p

        cleanup_memory()

        if offload:
            from ..ltx_core.model.transformer.model import BlockGPUManager         
            gpu_manager = BlockGPUManager(device="cuda", )
            gpu_manager.setup_for_inference(self.transformer.velocity_model)
        else:
            gpu_manager = None

        if stage_one:
            stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(cur_device)
            def denoising_loop(
                sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
            ) -> tuple[LatentState, LatentState]:
                return euler_denoising_loop(
                    sigmas=sigmas,
                    video_state=video_state,
                    audio_state=audio_state,
                    stepper=stepper,
                    denoise_fn=simple_denoising_func(
                        video_context=video_context,
                        audio_context=audio_context,
                        transformer=self.transformer,
                        gpu_manager=gpu_manager,  # noqa: F821
                    ),
                )

            # stage_1_output_shape = VideoPixelShape(
            #     batch=1,
            #     frames=num_frames,
            #     width=width // 2,
            #     height=height // 2,
            #     fps=frame_rate,
            # )

            # print(f"Stage 1: {stage_1_output_shape}")
            video_state, audio_state = denoise_audio_video(
                output_shape=images["stage_1_output_shape"],
                conditionings=images["stage_1_conditionings"],
                noiser=noiser,
                sigmas=stage_1_sigmas,
                stepper=stepper,
                denoising_loop_fn=denoising_loop,
                components=self.pipeline_components,
                dtype=dtype,
                device=cur_device,
            )

            return video_state.latent, audio_state.latent
        else:
            stage_2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(device)
            #stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)

            def denoising_loop(
                    sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
                ) -> tuple[LatentState, LatentState]:
                return euler_denoising_loop(
                    sigmas=sigmas,
                    video_state=video_state,
                    audio_state=audio_state,
                    stepper=stepper,
                    denoise_fn=simple_denoising_func(
                        video_context=video_context,
                        audio_context=audio_context,
                        transformer=self.transformer,
                        gpu_manager=gpu_manager,  # noqa: F821
                    ),
                )

            video_state, audio_state = denoise_audio_video(
                output_shape=images["stage_2_output_shape"],
                conditionings=images["stage_2_conditionings"],
                noiser=noiser,
                sigmas=stage_2_sigmas,
                stepper=stepper,
                denoising_loop_fn=denoising_loop,
                components=self.pipeline_components,
                dtype=dtype,
                device=device,
                noise_scale=stage_2_sigmas[0],
                initial_video_latent=v_lat,
                initial_audio_latent=a_lat,
            )

            torch.cuda.synchronize()
            if gpu_manager is not None:
                gpu_manager.unload_blocks_to_cpu()

            cleanup_memory()
            return video_state.latent, audio_state.latent

   
def load_pipeline_ltx_distilled(
    checkpoint_path: str,
    gemma_root: str,
    spatial_upsampler_path: str,
    loras: list[str],
    device="cpu" ,
    quantization: QuantizationPolicy | None = None,
    gguf_dit=False,
) -> DistilledPipeline:

    pipeline = DistilledPipeline(
        checkpoint_path=checkpoint_path,
        gemma_root=gemma_root,
        spatial_upsampler_path=spatial_upsampler_path,
        loras=loras,
        device=device,
        quantization=quantization,
        gguf_dit=gguf_dit
    )
    pipeline.load_dit()
    return pipeline

@torch.inference_mode()
def inference_ltx_distilled(
    pipeline,
    context_p,
    latents,
    seed: int,
    offload=True,
) -> tuple[torch.Tensor, torch.Tensor]:
    tiling_config = TilingConfig.default()
    num_frames = latents["num_frames"]
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
    video, audio = pipeline(
        prompt=None,
        seed=seed,
        height=latents["height"],
        width=latents["width"],
        num_frames=num_frames,
        frame_rate=latents["frame_rate"],
        images=latents,
        tiling_config=tiling_config,
        context_p=context_p,
        offload=offload,
    )
    return video, audio


@torch.inference_mode()
def inference_ltx_distilled_stage2(
    pipeline,
    context_p,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: int,
    images: str,
    offload=True,
) -> tuple[torch.Tensor, torch.Tensor]:
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
    video, audio = pipeline(
        prompt=None,
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        images=images,
        tiling_config=tiling_config,
        context_p=context_p,
        offload=offload,
    )
    return video, audio

