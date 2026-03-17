import logging
from collections.abc import Iterator

import torch

from ..ltx_core.components.diffusion_steps import EulerDiffusionStep
from ..ltx_core.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ..ltx_core.components.noisers import GaussianNoiser
from ..ltx_core.components.protocols import DiffusionStepProtocol
from ..ltx_core.components.schedulers import LTX2Scheduler
from ..ltx_core.loader import LoraPathStrengthAndSDOps
from ..ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ..ltx_core.model.upsampler import upsample_video
from ..ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ..ltx_core.model.video_vae import decode_video as vae_decode_video
from ..ltx_core.types import LatentState, VideoPixelShape,Audio
from ..ltx_core.quantization import QuantizationPolicy
from ..ltx_pipelines.utils import ModelLedger
from .utils.constants import (
    STAGE_2_DISTILLED_SIGMA_VALUES,detect_params
)
from .utils import (
    ModelLedger,
    assert_resolution,
    cleanup_memory,
    combined_image_conditionings,
    denoise_audio_video,
    encode_prompts,
    euler_denoising_loop,
    get_device,
    multi_modal_guider_factory_denoising_func,
    simple_denoising_func,
)
from .utils.media_io import decode_audio_from_file, encode_video
from .utils.samplers import euler_denoising_loop
from .utils.types import PipelineComponents

#device = get_device()


class TI2VidTwoStagesPipeline:
    """
    Two-stage text/image-to-video generation pipeline.
    Stage 1 generates video at the target resolution with CFG guidance, then
    Stage 2 upsamples by 2x and refines using a distilled LoRA for higher
    quality output. Supports optional image conditioning via the images parameter.
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: str = "cpu",
        quantization: QuantizationPolicy | None = None,
        gguf_dit: bool = False,
    ):
        self.device = device
        self.dtype = torch.bfloat16
        self.distilled_lora=distilled_lora
        self.stage_1_model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=loras,
            quantization=quantization,
            gguf_dit=gguf_dit,
            load_mode= "dit",
        )
        if distilled_lora:
            self.stage_2_model_ledger = self.stage_1_model_ledger.with_additional_loras(
                loras=self.distilled_lora,
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
        prompt,
        negative_prompt,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        audio_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        images: dict,
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

        cleanup_memory()

        if offload:
            from ..ltx_core.model.transformer.model import BlockGPUManager         
            gpu_manager = BlockGPUManager(device="cuda",)
            gpu_manager.setup_for_inference(self.transformer1.velocity_model)
        else:
            gpu_manager = None
        if stage_one:
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
                    gpu_manager=gpu_manager,
                    ),
                )

            # stage_1_output_shape = VideoPixelShape(
            #     batch=1,
            #     frames=num_frames,
            #     width=width // 2,
            #     height=height // 2,
            #     fps=frame_rate,
            # )

            video_state,audio_state = denoise_audio_video(
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

            video_state,audio_state = denoise_audio_video(
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
            # del transformer
            # del video_encoder
            cleanup_memory()

            return video_state.latent, audio_state.latent



def load_pipeline_ltx(
    checkpoint_path: str,
    distilled_lora: str,
    spatial_upsampler_path: str,
    gemma_root: str,
    loras: list[str],
    device: str = "cpu",
    quantization: QuantizationPolicy | None = None,
    gguf_dit=False,
) -> TI2VidTwoStagesPipeline:
    
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=checkpoint_path,
        distilled_lora=distilled_lora,
        spatial_upsampler_path=spatial_upsampler_path,
        gemma_root=gemma_root,
        loras=loras,
        device=device,
        quantization=quantization,
        gguf_dit=gguf_dit,
    )
    pipeline.load_dit() #check it
    return pipeline

@torch.inference_mode()
def inference_ltx_two_stages(
    pipeline,
    context_p,
    context_n,
    latents,
    seed,
    num_inference_steps: int,
    args,
    offload=True,
) -> tuple[torch.Tensor, torch.Tensor]:

    num_frames = latents["num_frames"]
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
    video_guider_params=MultiModalGuiderParams(
        cfg_scale=args["video_cfg_guidance_scale"],
        stg_scale=args["video_stg_guidance_scale"],
        rescale_scale=args["video_rescale_scale"],
        modality_scale=args["a2v_guidance_scale"],
        skip_step=args["video_skip_step"],
        stg_blocks=args["video_stg_blocks"],
        )
    audio_guider_params=MultiModalGuiderParams(
        cfg_scale=args["audio_cfg_guidance_scale"],
        stg_scale=args["audio_stg_guidance_scale"],
        rescale_scale=args["audio_rescale_scale"],
        modality_scale=args["v2a_guidance_scale"],
        skip_step=args["audio_skip_step"],
        stg_blocks=args["audio_stg_blocks"],
    )
    video, audio = pipeline(
        prompt=None,
        negative_prompt=None,
        seed=seed,
        height=latents["height"],
        width=latents["width"],
        num_frames=num_frames,
        frame_rate=latents["frame_rate"],
        num_inference_steps=num_inference_steps,
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params,
        images=latents,
        tiling_config=tiling_config,
        context_p=context_p,
        context_n=context_n,
        offload=offload,
    )
    return video, audio


# @torch.inference_mode()
# def main() -> None:
#     logging.getLogger().setLevel(logging.INFO)
#     parser = default_2_stage_arg_parser()
#     args = parser.parse_args()
#     pipeline = TI2VidTwoStagesPipeline(
#         checkpoint_path=args.checkpoint_path,
#         distilled_lora=args.distilled_lora,
#         spatial_upsampler_path=args.spatial_upsampler_path,
#         gemma_root=args.gemma_root,
#         loras=args.lora,
#         fp8transformer=args.enable_fp8,
#     )
#     tiling_config = TilingConfig.default()
#     video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
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
#         tiling_config=tiling_config,
#     )

#     encode_video(
#         video=video,
#         fps=args.frame_rate,
#         audio=audio,
#         audio_sample_rate=AUDIO_SAMPLE_RATE,
#         output_path=args.output_path,
#         video_chunks_number=video_chunks_number,
#     )


# if __name__ == "__main__":
#     main()
