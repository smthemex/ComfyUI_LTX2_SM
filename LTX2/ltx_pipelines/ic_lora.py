import logging
from collections.abc import Iterator

import torch

from ..ltx_core.components.diffusion_steps import EulerDiffusionStep
from ..ltx_core.components.noisers import GaussianNoiser
from ..ltx_core.components.protocols import DiffusionStepProtocol
from ..ltx_core.conditioning import ConditioningItem, VideoConditionByKeyframeIndex
from ..ltx_core.loader import LoraPathStrengthAndSDOps
from ..ltx_core.model.video_vae import TilingConfig, VideoEncoder, get_video_chunks_number

from ..ltx_core.types import Audio, LatentState, VideoLatentShape, VideoPixelShape
from ..ltx_pipelines.utils import (
    ModelLedger,
    assert_resolution,
    cleanup_memory,
    combined_image_conditionings,
    denoise_audio_video,
    encode_prompts,
    euler_denoising_loop,
    get_device,
    simple_denoising_func,
    )
from ..ltx_pipelines.utils.args import (
    ImageConditioningInput,
    VideoConditioningAction,
    VideoMaskConditioningAction,
    default_2_stage_distilled_arg_parser,
    detect_checkpoint_path,
)

from .utils.args import (
    ImageConditioningInput,
    VideoConditioningAction,
    VideoMaskConditioningAction,
    default_2_stage_distilled_arg_parser,
    detect_checkpoint_path,
)
from .utils.constants import (
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    detect_params,
)
from .utils.helpers import (
    image_conditionings_by_replacing_latent,

)
from .utils.media_io import encode_video, load_video_conditioning
from .utils.types import PipelineComponents

device = get_device()


class ICLoraPipeline:
    """
    Two-stage video generation pipeline with In-Context (IC) LoRA support.
    Allows conditioning the generated video on control signals such as depth maps,
    human pose, or image edges via the video_conditioning parameter.
    The specific IC-LoRA model should be provided via the loras parameter.
    Stage 1 generates video at the target resolution, then Stage 2 upsamples
    by 2x and refines with additional denoising steps for higher quality output.
    """

    def __init__(
        self,
        checkpoint_path: str,
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device = "cpu",
        fp8transformer: bool = False,
        gguf_dit: bool = False,
    ):
        self.dtype = torch.bfloat16
        self.stage_1_model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root_path=gemma_root,
            loras=loras,
            fp8transformer=fp8transformer,
            gguf_dit=gguf_dit,
            load_mode= "dit",
        )
        self.stage_2_model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root_path=gemma_root,
            loras=[],
            fp8transformer=fp8transformer,
            gguf_dit=gguf_dit,
            load_mode= "dit",
        )
        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )
        self.device = device

    def load_dit(self,stage1=True,load=True,):
        if load:
            if stage1:
                self.transformer1=self.stage_1_model_ledger.transformer()
            else:
                self.transformer2=self.stage_2_model_ledger.transformer()
        else:
            if not  stage1:
                self.transformer2=None
                del self.transformer2
            cleanup_memory()
    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[tuple[str, int, float]],
        video_conditioning: list[tuple[str, float]],
        enhance_prompt: bool = False,
        tiling_config: TilingConfig | None = None,
        context_p=None,
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

        # text_encoder = self.stage_1_model_ledger.text_encoder()
        v_context_p, a_context_p = context_p
        # if enhance_prompt:
        #     prompt = generate_enhanced_prompt(
        #         text_encoder, prompt, images[0][0] if len(images) > 0 else None, seed=seed
        #     )
        # video_context, audio_context = encode_text(text_encoder, prompts=[prompt])[0]

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
        
            # Stage 1: Initial low resolution video generation.
            #video_encoder = self.stage_1_model_ledger.video_encoder()
            #transformer = self.stage_1_model_ledger.transformer()
            stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(cur_device)

            def first_stage_denoising_loop(
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
                        transformer=self.transformer1, 
                        gpu_manager=gpu_manager, # noqa: F821
                    ),
                    gpu_manager=gpu_manager,
                )
            

            stage_1_output_shape = VideoPixelShape(
                batch=1,
                frames=num_frames,
                width=width // 2,
                height=height // 2,
                fps=frame_rate,
            )
            # stage_1_conditionings = self._create_conditionings(
            #     images=images,
            #     video_conditioning=video_conditioning,
            #     height=stage_1_output_shape.height,
            #     width=stage_1_output_shape.width,
            #     video_encoder=video_encoder,
            #     num_frames=num_frames,
            #)
            video_state, audio_state = denoise_audio_video(
                output_shape=stage_1_output_shape,
                conditionings=images[1] if images else [],
                noiser=noiser,
                sigmas=stage_1_sigmas,
                stepper=stepper,
                denoising_loop_fn=first_stage_denoising_loop,
                components=self.pipeline_components,
                dtype=dtype,
                device=cur_device,
            )

            torch.cuda.synchronize()
            #del transformer
            cleanup_memory()
            return video_state.latent, audio_state.latent
        else:
            # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
            # upscaled_video_latent = upsample_video(
            #     latent=video_state.latent[:1],
            #     video_encoder=video_encoder,
            #     upsampler=self.stage_2_model_ledger.spatial_upsampler(),
            # )
            self.load_dit(stage1=False,load=True,)
            if offload:
                from ..ltx_core.model.transformer.model import BlockGPUManager         
                gpu_manager = BlockGPUManager(device="cuda",)
                gpu_manager.setup_for_inference(self.transformer2.velocity_model)
            else:
                gpu_manager = None
            torch.cuda.synchronize()
            cleanup_memory()

            #transformer = self.stage_2_model_ledger.transformer()
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
                        gpu_manager=gpu_manager,    # noqa: F821
                    ),
                    gpu_manager=gpu_manager,
                )

            stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
            # stage_2_conditionings = image_conditionings_by_replacing_latent(
            #     images=images,
            #     height=stage_2_output_shape.height,
            #     width=stage_2_output_shape.width,
            #     video_encoder=video_encoder,
            #     dtype=self.dtype,
            #     device=cur_device,
            # )

            video_state, audio_state = denoise_audio_video(
                output_shape=stage_2_output_shape,
                conditionings=images[0] if images else [],
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

            # decoded_video = vae_decode_video(
            #     video_state.latent, self.stage_2_model_ledger.video_decoder(), tiling_config, generator)
            # decoded_audio = vae_decode_audio(
            #     audio_state.latent, self.stage_2_model_ledger.audio_decoder(), self.stage_2_model_ledger.vocoder()
            # )
            return video_state.latent, audio_state.latent

    def _create_conditionings(
        self,
        images: list[tuple[str, int, float]],
        video_conditioning: list[tuple[str, float]],
        height: int,
        width: int,
        num_frames: int,
        video_encoder: VideoEncoder,
    ) -> list[ConditioningItem]:
        conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=height,
            width=width,
            video_encoder=video_encoder,
            dtype=self.dtype,
            device=self.device,
        )

        for video_path, strength in video_conditioning:
            video = load_video_conditioning(
                video_path=video_path,
                height=height,
                width=width,
                frame_cap=num_frames,
                dtype=self.dtype,
                device=self.device,
            )
            encoded_video = video_encoder(video)
            conditionings.append(VideoConditionByKeyframeIndex(keyframes=encoded_video, frame_idx=0, strength=strength))

        return conditionings

def create_conditionings_(
        images: list[tuple[str, int, float]],
        video_conditioning: list[tuple[str, float]],
        height: int,
        width: int,
        num_frames: int,
        video_encoder: VideoEncoder,
        dtype, device,
    ) -> list[ConditioningItem]:
    conditionings = image_conditionings_by_replacing_latent(
        images=images,
        height=height,
        width=width,
        video_encoder=video_encoder,
        dtype=dtype,
        device=device,
    )   
    for video, strength in video_conditioning:
        # video = load_video_conditioning(
        #     video_path=video_path,
        #     height=height,
        #     width=width,
        #     frame_cap=num_frames,
        #     dtype=dtype,
        #     device=device,
        # ) #(1, C, F, height, width)
        encoded_video = video_encoder(video)
        conditionings.append(VideoConditionByKeyframeIndex(keyframes=encoded_video, frame_idx=0, strength=strength))

    # for video_path, strength in video_conditioning:
    #     video = load_video_conditioning(
    #         video_path=video_path,
    #         height=height,
    #         width=width,
    #         frame_cap=num_frames,
    #         dtype=dtype,
    #         device=device,
    #     ) #(1, C, F, height, width)
    #     encoded_video = video_encoder(video)
    #     conditionings.append(VideoConditionByKeyframeIndex(keyframes=encoded_video, frame_idx=0, strength=strength))

    return conditionings

def load_pipeline_ICLora_ltx(
    checkpoint_path: str,
    distilled_lora: str,
    spatial_upsampler_path: str,
    gemma_root: str,
    loras: list[str],
    device: str = "cpu",
    fp8transformer: bool=False,
    gguf_dit=False,
) -> ICLoraPipeline:  
    pipeline = ICLoraPipeline(
        checkpoint_path=checkpoint_path,
        distilled_lora=distilled_lora,
        spatial_upsampler_path=spatial_upsampler_path,
        gemma_root=gemma_root,
        loras=loras,
        device=device,
        fp8transformer=fp8transformer,
        gguf_dit=gguf_dit,
    )
    pipeline.load_dit() #check it
    return pipeline

@torch.inference_mode()
def inference_ltx_ICLora(
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
    images: str,
    offload=True,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
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
        tiling_config=tiling_config,
        context_p=context_p,
        offload=offload,
    )
    return video, audio


# @torch.inference_mode()
# def main() -> None:
#     logging.getLogger().setLevel(logging.INFO)
#     parser = default_2_stage_distilled_arg_parser()
#     parser.add_argument(
#         "--video-conditioning",
#         action=VideoConditioningAction,
#         nargs=2,
#         metavar=("PATH", "STRENGTH"),
#         required=True,
#     )
#     args = parser.parse_args()
#     pipeline = ICLoraPipeline(
#         checkpoint_path=args.checkpoint_path,
#         spatial_upsampler_path=args.spatial_upsampler_path,
#         gemma_root=args.gemma_root,
#         loras=args.lora,
#         fp8transformer=args.enable_fp8,
#     )
#     tiling_config = TilingConfig.default()
#     video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
#     video, audio = pipeline(
#         prompt=args.prompt,
#         seed=args.seed,
#         height=args.height,
#         width=args.width,
#         num_frames=args.num_frames,
#         frame_rate=args.frame_rate,
#         images=args.images,
#         video_conditioning=args.video_conditioning,
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
