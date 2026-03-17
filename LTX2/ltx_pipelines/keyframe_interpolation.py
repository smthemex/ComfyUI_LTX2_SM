import logging
from collections.abc import Iterator

import torch

from ..ltx_core.components.diffusion_steps import EulerDiffusionStep
from ..ltx_core.components.guiders import CFGGuider
from ..ltx_core.components.noisers import GaussianNoiser
from ..ltx_core.components.protocols import DiffusionStepProtocol
from ..ltx_core.components.schedulers import LTX2Scheduler
from ..ltx_core.loader import LoraPathStrengthAndSDOps
from ..ltx_core.quantization import QuantizationPolicy
from ..ltx_core.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ..ltx_core.model.video_vae import TilingConfig, get_video_chunks_number

from ..ltx_core.types import LatentState, VideoPixelShape
from .utils import ModelLedger
from .utils.args import ImageConditioningInput, default_2_stage_arg_parser, detect_checkpoint_path
from .utils.constants import (
    STAGE_2_DISTILLED_SIGMA_VALUES, detect_params
)
from .utils.helpers import (
    assert_resolution,
    cleanup_memory,
    denoise_audio_video,
    encode_prompts,
    get_device,
    image_conditionings_by_adding_guiding_latent,
    multi_modal_guider_factory_denoising_func,
    simple_denoising_func,
)
from .utils.media_io import encode_video
from .utils.samplers import euler_denoising_loop
from .utils.types import PipelineComponents

device = get_device()


class KeyframeInterpolationPipeline:
    """
    Keyframe-based Two-stage video interpolation pipeline.
    Interpolates between keyframes to generate a video with smoother transitions.
    Stage 1 generates video at the target resolution, then Stage 2 upsamples
    by 2x and refines with additional denoising steps for higher quality output.
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device = "cpu",
        quantization: QuantizationPolicy | None = None,
        gguf_dit=False,
    ):
        self.device = device
        self.dtype = torch.bfloat16
        self.distilled_lora=distilled_lora
        self.stage_1_model_ledger = ModelLedger(
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
        if self.distilled_lora:
            self.stage_2_model_ledger = self.stage_1_model_ledger.with_additional_loras(
                loras=distilled_lora,
            )
        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )

    def load_dit(self,stage1=True,):
        if stage1:
            self.transformer1=self.stage_1_model_ledger.transformer()
        else:
            if  self.distilled_lora:
                self.transformer1=None
                cleanup_memory()
                self.transformer2=self.stage_2_model_ledger.transformer()
            else:
                self.transformer2=self.transformer1

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        audio_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        images: list[tuple[str, int, float]],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        context_p=None,
        context_n=None,
        offload=False,
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

        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = context_n

        torch.cuda.synchronize()
        #del text_encoder
        cleanup_memory()


        if stage_one:
            if offload:
                from ..ltx_core.model.transformer.model import BlockGPUManager         
                gpu_manager = BlockGPUManager(device="cuda",)
                gpu_manager.setup_for_inference(self.transformer1.velocity_model)
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
                    denoise_fn=multi_modal_guider_factory_denoising_func(
                        video_guider_factory=create_multimodal_guider_factory(
                        params=video_guider_params,
                        negative_context=v_context_n,
                    ),
                        audio_guider_factory=create_multimodal_guider_factory(
                            params=audio_guider_params,
                            negative_context=a_context_n,
                        ),
                        v_context=v_context_p,
                        a_context=a_context_p,
                        transformer=self.transformer1,  # noqa: F821
                        gpu_manager=gpu_manager, # noqa: F821
                    ),
                )

            # stage_1_output_shape = VideoPixelShape(
            #     batch=1,
            #     frames=num_frames,
            #     width=width // 2,
            #     height=height // 2,
            #     fps=frame_rate,
            # )

            video_state, audio_state = denoise_audio_video(
                 output_shape=images["stage_1_output_shape"],
                conditionings=images["stage_1_conditionings"],
                noiser=noiser,
                sigmas=sigmas,
                stepper=stepper,
                denoising_loop_fn=first_stage_denoising_loop,
                components=self.pipeline_components,
                dtype=dtype,
                device=cur_device,
            )

            torch.cuda.synchronize()
            cleanup_memory()
            return video_state.latent, audio_state.latent
        else:
            self.load_dit(False)
            if offload:
                from ..ltx_core.model.transformer.model import BlockGPUManager         
                gpu_manager = BlockGPUManager(device="cuda",)
                gpu_manager.setup_for_inference(self.transformer2.velocity_model)
            else:
                gpu_manager = None
            torch.cuda.synchronize()
            cleanup_memory()
            distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(cur_device)

            def second_stage_denoising_loop(
                sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
            ) -> tuple[LatentState, LatentState]:
                return euler_denoising_loop(
                    sigmas=sigmas,
                    video_state=video_state,
                    audio_state=audio_state,
                    stepper=stepper,
                    denoise_fn=simple_denoising_func(
                        video_context=v_context_p,
                        audio_context=a_context_p,
                        transformer=self.transformer2,
                        gpu_manager=gpu_manager,  # noqa: F821
                    ),
                )

            #stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)

            video_state, audio_state = denoise_audio_video(
                output_shape=images["stage_2_output_shape"],
                conditionings=images["stage_2_conditionings"],
                noiser=noiser,
                sigmas=distilled_sigmas,
                stepper=stepper,
                denoising_loop_fn=second_stage_denoising_loop,
                components=self.pipeline_components,
                dtype=dtype,
                device=cur_device,
                noise_scale=distilled_sigmas[0],
                initial_video_latent=v_lat,
                initial_audio_latent=a_lat,
            )

            torch.cuda.synchronize()
            if gpu_manager is not None:
                gpu_manager.unload_blocks_to_cpu()

            cleanup_memory()

            return video_state.latent, audio_state.latent


def load_pipeline_Keyframe_ltx(
    checkpoint_path: str,
    distilled_lora: str,
    spatial_upsampler_path: str,
    gemma_root: str,
    loras: list[str],
    device: str = "cpu",
    quantization: QuantizationPolicy | None = None,
    gguf_dit=False,
) -> KeyframeInterpolationPipeline:
    
    pipeline = KeyframeInterpolationPipeline(
        checkpoint_path=checkpoint_path,
        distilled_lora=distilled_lora,
        spatial_upsampler_path=spatial_upsampler_path,
        gemma_root=gemma_root,
        loras=loras,
        device=device,
        quantization=quantization,
        gguf_dit=gguf_dit,
    )
    pipeline.load_dit() 
    return pipeline

@torch.inference_mode()
def inference_ltx_Keyframe(
    pipeline,
    context_p,
    context_n,
    latents,
    seed: int,
    num_inference_steps: int,
    cfg_guidance_scale: float,
    offload=True,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_frames = latents["num_frames"]
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
    video, audio = pipeline(
        prompt=None,
        negative_prompt=None,
        seed=seed,
        height=latents["height"],
        width=latents["width"],
        num_frames=num_frames,
        frame_rate=latents["frame_rate"],
        num_inference_steps=num_inference_steps,
        cfg_guidance_scale=cfg_guidance_scale,
        images=latents,
        tiling_config=tiling_config,
        context_p=context_p,
        context_n=context_n,
        offload=offload,
    )
    return video, audio


