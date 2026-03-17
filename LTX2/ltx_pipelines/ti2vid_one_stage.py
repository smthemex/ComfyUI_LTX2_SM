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
from ..ltx_core.model.video_vae import decode_video as vae_decode_video
from ..ltx_core.quantization import QuantizationPolicy
from ..ltx_core.types import LatentState, VideoPixelShape
from ..ltx_pipelines.utils import (
    ModelLedger,
    assert_resolution,
    cleanup_memory,
    combined_image_conditionings,
    denoise_audio_video,
    encode_prompts,
    euler_denoising_loop,
    get_device,
    multi_modal_guider_factory_denoising_func,)
from ..ltx_pipelines.utils.args import ImageConditioningInput, default_1_stage_arg_parser, detect_checkpoint_path
from ..ltx_pipelines.utils.constants import detect_params

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
        quantization: QuantizationPolicy | None = None,
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
        # if context_n is None:
        #     cfg_guidance_scale=1.0
        # cfg_guider = CFGGuider(cfg_guidance_scale)
        dtype = torch.bfloat16

        v_context_p, a_context_p = context_p
        if context_n is None:
            v_context_n = None
            a_context_n = None
        else:
            v_context_n, a_context_n = context_n

        torch.cuda.synchronize()
      
        cleanup_memory()


        if offload:
            from ..ltx_core.model.transformer.model import BlockGPUManager         
            gpu_manager = BlockGPUManager(device="cuda", )
            gpu_manager.setup_for_inference(self.transformer.velocity_model)
        else:
            gpu_manager = None
        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=cur_device)
        
        video_guider_factory = create_multimodal_guider_factory(
            params=video_guider_params,
            negative_context=v_context_n,
        )
        audio_guider_factory = create_multimodal_guider_factory(
            params=audio_guider_params,
            negative_context=a_context_n,
        )
        def first_stage_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=multi_modal_guider_factory_denoising_func(
                    video_guider_factory=video_guider_factory,
                    audio_guider_factory=audio_guider_factory,
                    v_context=v_context_p,
                    a_context=a_context_p,
                    transformer=self.transformer,
                    gpu_manager=gpu_manager
                ),
            )


        #stage_1_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)

        video_state, audio_state = denoise_audio_video(
            output_shape=images["stage_2_output_shape"],
            conditionings=images["stage_2_conditionings"],
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
    quantization: QuantizationPolicy | None = None,
    gguf_dit=False,
) -> TI2VidOneStagePipeline:
    
    pipeline = TI2VidOneStagePipeline(
        checkpoint_path=checkpoint_path,
        gemma_root=gemma_root,
        loras=loras,
        device=device,
        quantization=quantization,
        gguf_dit=gguf_dit
    )
    pipeline.load_dit()
    return pipeline



@torch.inference_mode()
def inference_ltx_one_stage(
    pipeline,
    context_p,
    context_n,
    latents,
    seed: int,
    num_inference_steps: int,
    args,
    offload=True,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_frames = latents["num_frames"]
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
        context_p=context_p,
        context_n=context_n,
        offload=offload,
    )

    return video, audio


