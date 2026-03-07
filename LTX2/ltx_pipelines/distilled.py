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

        # text_encoder = self.model_ledger.text_encoder()
        # if enhance_prompt:
        #     prompt = generate_enhanced_prompt(text_encoder, prompt, images[0][0] if len(images) > 0 else None)
        # context_p = encode_text(text_encoder, prompts=[prompt])[0]
        video_context, audio_context = context_p

        # torch.cuda.synchronize()
        # del text_encoder
        cleanup_memory()

        # Stage 1: Initial low resolution video generation.
        #video_encoder = self.model_ledger.video_encoder()
       
        #transformer = self.model_ledger.transformer()

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
                    gpu_manager=gpu_manager,
                )

            stage_1_output_shape = VideoPixelShape(
                batch=1,
                frames=num_frames,
                width=width // 2,
                height=height // 2,
                fps=frame_rate,
            )
            # stage_1_conditionings = image_conditionings_by_replacing_latent(
            #     images=images,
            #     height=stage_1_output_shape.height,
            #     width=stage_1_output_shape.width,
            #     video_encoder=video_encoder,
            #     dtype=dtype,
            #     device=cur_device,
            # )
            print(f"Stage 1: {stage_1_output_shape}")
            video_state, audio_state = denoise_audio_video(
                output_shape=stage_1_output_shape,
                conditionings=images[1] if images else [],
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
            stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
            # stage_2_conditionings = image_conditionings_by_replacing_latent(
            #     images=images,
            #     height=stage_2_output_shape.height,
            #     width=stage_2_output_shape.width,
            #     video_encoder=video_encoder,
            #     dtype=dtype,
            #     device=cur_device,
            # )

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
                    gpu_manager=gpu_manager,
                )

            video_state, audio_state = denoise_audio_video(
                output_shape=stage_2_output_shape,
                conditionings=images[0] if images else [],
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
            #del transformer
            #del video_encoder
            cleanup_memory()
            return video_state.latent, audio_state.latent
        # # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
        # upscaled_video_latent = upsample_video(
        #     latent=video_state.latent[:1], video_encoder=video_encoder, upsampler=self.model_ledger.spatial_upsampler()
        # )

        # torch.cuda.synchronize()
        # cleanup_memory()

        # stage_2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(cur_device)
        # stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        # stage_2_conditionings = image_conditionings_by_replacing_latent(
        #     images=images,
        #     height=stage_2_output_shape.height,
        #     width=stage_2_output_shape.width,
        #     video_encoder=video_encoder,
        #     dtype=dtype,
        # #     device=cur_device,
        # # )
        # video_state, audio_state = denoise_audio_video(
        #     output_shape=stage_2_output_shape,
        #     conditionings=images[0] if images else [],
        #     noiser=noiser,
        #     sigmas=stage_2_sigmas,
        #     stepper=stepper,
        #     denoising_loop_fn=denoising_loop,
        #     components=self.pipeline_components,
        #     dtype=dtype,
        #     device=cur_device,
        #     noise_scale=stage_2_sigmas[0],
        #     initial_video_latent=upscaled_video_latent,
        #     initial_audio_latent=audio_state.latent,
        # )

        # torch.cuda.synchronize()
        # if gpu_manager is not None:
        #     gpu_manager.unload_blocks_to_cpu()
        # del transformer
        # #del video_encoder
        # cleanup_memory()

        # # if offload:
        # #     video_decoder=self.model_ledger.video_decoder().to(cur_device)
        # #     audio_decoder=self.model_ledger.audio_decoder().to(cur_device)
        # #     vocoder=self.model_ledger.vocoder().to(cur_device)
        # # else:
        # #     video_decoder=self.model_ledger.video_decoder()
        # #     audio_decoder=self.model_ledger.audio_decoder()
        # #     vocoder=self.model_ledger.vocoder()
        
        # # decoded_video = vae_decode_video(
        # #   video_state.latent, self.model_ledger.video_decoder(), tiling_config, generator)

        # # decoded_audio = vae_decode_audio(audio_state.latent, audio_decoder, vocoder)   
        # return video_state.latent, audio_state.latent

   
def load_pipeline_ltx_distilled(
    checkpoint_path: str,
    gemma_root: str,
    spatial_upsampler_path: str,
    loras: list[str],
    device="cpu" ,
    fp8transformer: bool=False,
    gguf_dit=False,
) -> DistilledPipeline:

    pipeline = DistilledPipeline(
        checkpoint_path=checkpoint_path,
        gemma_root=gemma_root,
        spatial_upsampler_path=spatial_upsampler_path,
        loras=loras,
        device=device,
        fp8transformer=fp8transformer,
        gguf_dit=gguf_dit
    )
    pipeline.load_dit()
    return pipeline



@torch.inference_mode()
def inference_ltx_distilled(
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
# @torch.inference_mode()
# def main() -> None:
#     logging.getLogger().setLevel(logging.INFO)
#     parser = default_2_stage_distilled_arg_parser()
#     args = parser.parse_args()
#     pipeline = DistilledPipeline(
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
#         tiling_config=tiling_config,
#         enhance_prompt=args.enhance_prompt,
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
