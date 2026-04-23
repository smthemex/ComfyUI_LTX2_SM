import logging
from collections.abc import Iterator
from einops import rearrange
import torch

from ..ltx_core.components.diffusion_steps import Res2sDiffusionStep
from ..ltx_core.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    MultiModalGuider,
    create_multimodal_guider_factory,
)
from ..ltx_core.components.noisers import GaussianNoiser
from ..ltx_core.components.schedulers import LTX2Scheduler
from ..ltx_core.loader import LoraPathStrengthAndSDOps
from ..ltx_core.quantization import QuantizationPolicy
from ..ltx_core.types import VideoLatentShape
from ..ltx_core.conditioning.types.noise_mask_cond import TemporalRegionMask
from .utils.blocks import DiffusionStage
from .utils.constants import STAGE_2_DISTILLED_SIGMAS,DISTILLED_SIGMAS
from .utils.denoisers import FactoryGuidedDenoiser, SimpleDenoiser,GuidedDenoiser
from .utils.types import ModalitySpec
from .utils.samplers import res2s_audio_video_denoising_loop

# pipeline components
class OmniPipeline:
    """
    simple pipeline component
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        loras: list[LoraPathStrengthAndSDOps],
        device: str = "cpu",
        quantization: QuantizationPolicy | None = None,
        infer_mode="distilled", #["distilled","one_stage","two_stages","keyframe","ic_lora","audio2v","retake","twostages_hq"]
        torch_compile: bool = False,
        offload: bool = False,
    ):
        self.device = device
        self.infer_mode=infer_mode
        self.distilled =True if "distill" in checkpoint_path.lower() or len(distilled_lora)>=1 else False
        self.gguf_dit=True if checkpoint_path.endswith(".gguf") else False
        self.dtype = torch.bfloat16
        self.distilled_lora=distilled_lora
        self._scheduler = LTX2Scheduler()
        self.offload = offload
        self.stage_2  = None
        stage_1_loras = tuple(loras)
        stage_2_loras = (*tuple(loras), *tuple(distilled_lora)) if not  self.infer_mode=="ic_lora" else ()

        if self.infer_mode=="twostages_hq" and len(distilled_lora)>=1:
            distilled_lora_stage_1 = LoraPathStrengthAndSDOps(
                        path=distilled_lora[0].path,
                        strength=0.25,
                        sd_ops=distilled_lora[0].sd_ops,
                    )
            distilled_lora_stage_2 = LoraPathStrengthAndSDOps(
                        path=distilled_lora[0].path,
                        strength=0.8,
                        sd_ops=distilled_lora[0].sd_ops,
                    )
            stage_1_loras=(*loras,distilled_lora_stage_1,)
            stage_2_loras=(*loras,distilled_lora_stage_2,)

        self.stage_1 = DiffusionStage(
                checkpoint_path=checkpoint_path,
                dtype=self.dtype,
                device=device,
                loras=stage_1_loras,
                quantization=quantization,
                torch_compile=torch_compile,
                offload=self.offload,
            )
         
        if self.infer_mode in ["ic_lora","twostages_hq","two_stages","keyframe"]:
            self.stage_2= DiffusionStage(
                checkpoint_path=checkpoint_path,
                dtype=self.dtype,
                device=device,
                loras=stage_2_loras,
                quantization=quantization,
                torch_compile=torch_compile,
                offload=self.offload,
            )

        self.upsampler=None

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        seed: int,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        audio_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        images: dict,
        context_p=None,
        context_n=None,
        streaming_prefetch_count=None, # OFFLOAD BLOCK
        max_batch_size=1,
        regenerate_audio=False,
        regenerate_video=False,
        upsampler=None,


    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        self.upsampler=upsampler

        #assert_resolution(height=height, width=width, is_two_stage=True)
        cur_device = self.device if not self.offload else "cuda"
        if self.infer_mode=="retake":
            seed = torch.randint(0, 2**31, (1,), device=self.device).item() if seed < 0 else seed
        generator = torch.Generator(device=cur_device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)

        dtype = torch.bfloat16
        
        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = context_n
         

        stage_1_output_shape= images["stage_1_output_shape"]
        stage_2_output_shape= images["stage_2_output_shape"]
        if self.infer_mode =="twostages_hq":
            stepper = Res2sDiffusionStep()
            empty_latent = torch.empty(VideoLatentShape.from_pixel_shape(stage_1_output_shape).to_torch_shape())
            stage_1_sigmas = self._scheduler.execute(latent=empty_latent, steps=num_inference_steps)
            sigmas = stage_1_sigmas.to(dtype=torch.float32, device=cur_device)
            video_state, audio_state = self.stage_1(
                denoiser=GuidedDenoiser(
                    v_context=v_context_p,
                    a_context=a_context_p,
                    video_guider=MultiModalGuider(
                        params=video_guider_params,
                        negative_context=v_context_n,
                    ),
                    audio_guider=MultiModalGuider(
                        params=audio_guider_params,
                        negative_context=a_context_n,
                    ),
                ),
                sigmas=sigmas,
                noiser=noiser,
                stepper=stepper,
                width=stage_1_output_shape.width,
                height=stage_1_output_shape.height,
                frames=stage_1_output_shape.frames,
                fps=stage_1_output_shape.fps,
                video=ModalitySpec(context=v_context_p, conditionings=images["stage_1_conditionings"]),
                audio=ModalitySpec(context=a_context_p),
                loop=res2s_audio_video_denoising_loop,
                streaming_prefetch_count=streaming_prefetch_count,
                max_batch_size=max_batch_size,
            )
           
        elif self.infer_mode=="keyframe":
            stage_1_sigmas=None
            sigmas = (
            stage_1_sigmas if stage_1_sigmas is not None else self._scheduler.execute(steps=num_inference_steps)
                ).to(dtype=torch.float32, device=cur_device)
            
            video_guider_factory = create_multimodal_guider_factory(
            params=video_guider_params,
            negative_context=v_context_n,
                )
            audio_guider_factory = create_multimodal_guider_factory(
                params=audio_guider_params,
                negative_context=a_context_n,
                )
            video_state, audio_state = self.stage_1(
                denoiser=FactoryGuidedDenoiser(
                    v_context=v_context_p,
                    a_context=a_context_p,
                    video_guider_factory=video_guider_factory,
                    audio_guider_factory=audio_guider_factory,
                ),
                sigmas=sigmas,
                noiser=noiser,
                width=stage_1_output_shape.width,
                height=stage_1_output_shape.height,
                frames=stage_1_output_shape.frames,
                fps=stage_1_output_shape.fps,
                video=ModalitySpec(
                    context=v_context_p,
                    conditionings=images["stage_1_conditionings"],
                ),
                audio=ModalitySpec(
                    context=a_context_p,
                ),
                streaming_prefetch_count=streaming_prefetch_count,
                max_batch_size=max_batch_size,
                
            )

        elif self.infer_mode=="ic_lora":
            stage_1_sigmas=DISTILLED_SIGMAS
            stage_1_sigmas = stage_1_sigmas.to(dtype=torch.float32, device=cur_device)
            video_state, audio_state = self.stage_1(
                denoiser=SimpleDenoiser(v_context_p, a_context_p),
                sigmas=stage_1_sigmas,
                noiser=noiser,
                width=stage_1_output_shape.width,
                height=stage_1_output_shape.height,
                frames=stage_1_output_shape.frames,
                fps=stage_1_output_shape.fps,
                video=ModalitySpec(
                    context=v_context_p,
                    conditionings=images["stage_1_conditionings"],
                ),
                audio=ModalitySpec(
                    context=a_context_p,
                ),
                streaming_prefetch_count=streaming_prefetch_count,
            )     
            
        elif self.infer_mode=="distilled":
            stage_1_sigmas=DISTILLED_SIGMAS
            stage_1_sigmas = stage_1_sigmas.to(dtype=torch.float32, device=cur_device)
            video_state, audio_state = self.stage_1(
                denoiser=SimpleDenoiser(v_context_p, a_context_p),
                sigmas=stage_1_sigmas,
                noiser=noiser,
                width=stage_1_output_shape.width,
                height=stage_1_output_shape.height,
                frames=stage_1_output_shape.frames,
                fps=stage_1_output_shape.fps,
                video=ModalitySpec(context=v_context_p, conditionings=images["stage_1_conditionings"]),
                audio=ModalitySpec(context=a_context_p),
                streaming_prefetch_count=streaming_prefetch_count,
                
            )
    
        elif self.infer_mode=="audio2v":
            stage_1_sigmas=None
            sigmas = (
                stage_1_sigmas if stage_1_sigmas is not None else self._scheduler.execute(steps=num_inference_steps)
            ).to(dtype=torch.float32, device=cur_device)
            video_state, _ = self.stage_1(
                denoiser=GuidedDenoiser(
                    v_context=v_context_p,
                    a_context=a_context_p,
                    video_guider=MultiModalGuider(
                        params=video_guider_params,
                        negative_context=v_context_n,
                    ),
                    audio_guider=MultiModalGuider(
                        params=MultiModalGuiderParams(),
                    ),
                ),
                sigmas=sigmas,
                noiser=noiser,
                width=stage_1_output_shape.width,
                height=stage_1_output_shape.height,
                frames=stage_1_output_shape.frames,
                fps=stage_1_output_shape.fps,
                video=ModalitySpec(
                    context=v_context_p,
                    conditionings=images["stage_1_conditionings"],
                ),
                audio=ModalitySpec(
                    context=a_context_p,
                    frozen=True,
                    noise_scale=0.0,
                    initial_latent=images["encoded_audio_latent"],
                ),
                streaming_prefetch_count=streaming_prefetch_count,
                max_batch_size=max_batch_size,
               
            )
        elif self.infer_mode=="retake":
            initial_audio_latent=images["encoded_audio_latent"]
            initial_video_latent=images["encoded_video_latent"]
            start_time=images["start_time"]
            end_time=images["end_time"]
            video_modality_spec = ModalitySpec(
                context=v_context_p,
                conditionings=[TemporalRegionMask(start_time=start_time, end_time=end_time, fps=stage_1_output_shape.fps)]
                if regenerate_video
                else [],
                initial_latent=initial_video_latent,
                frozen=not regenerate_video,
            )
            audio_modality_spec = ModalitySpec(
                context=a_context_p,
                conditionings=[TemporalRegionMask(start_time=start_time, end_time=end_time, fps=stage_1_output_shape.fps)]
                if (initial_audio_latent is not None and regenerate_audio)
                else [],
                initial_latent=initial_audio_latent,
                frozen=initial_audio_latent is not None and not regenerate_audio,
            )
            sigmas = DISTILLED_SIGMAS if self.distilled else self._scheduler.execute(steps=num_inference_steps)
            sigmas = sigmas.to(dtype=torch.float32, device=cur_device)
            if self.distilled:
                denoiser = SimpleDenoiser(
                    v_context=v_context_p,
                    a_context=a_context_p,
                )
            else:
                #v_context_n, a_context_n = contexts[1].video_encoding, contexts[1].audio_encoding
                video_guider = MultiModalGuider(
                    params=video_guider_params,
                    negative_context=v_context_n,
                )
                audio_guider = MultiModalGuider(
                    params=audio_guider_params,
                    negative_context=a_context_n,
                )
                denoiser = GuidedDenoiser(
                    v_context=v_context_p,
                    a_context=a_context_p,
                    video_guider=video_guider,
                    audio_guider=audio_guider,
                )
            # Run diffusion stage
            video_state, audio_state = self.stage_1(
                denoiser=denoiser,
                sigmas=sigmas,
                noiser=noiser,
                width=stage_1_output_shape.width,
                height=stage_1_output_shape.height,
                frames=stage_1_output_shape.frames,
                fps=stage_1_output_shape.fps,
                video=video_modality_spec,
                audio=audio_modality_spec,
                streaming_prefetch_count=streaming_prefetch_count,
                max_batch_size=max_batch_size,
               
            )

        elif self.infer_mode=="one_stage":

            sigmas = self._scheduler.execute(steps=num_inference_steps).to(dtype=torch.float32, device=cur_device)
            video_guider_factory = create_multimodal_guider_factory(
                params=video_guider_params,
                negative_context=v_context_n,
                )
            audio_guider_factory = create_multimodal_guider_factory(
                    params=audio_guider_params,
                    negative_context=a_context_n,
                )
            video_state, audio_state = self.stage_1(
                denoiser=FactoryGuidedDenoiser(
                    v_context=v_context_p,
                    a_context=a_context_p,
                    video_guider_factory=video_guider_factory,
                    audio_guider_factory=audio_guider_factory,
                    
                ),
                    sigmas=sigmas,
                    noiser=noiser,
                    width=stage_1_output_shape.width,
                    height=stage_1_output_shape.height,
                    frames=stage_1_output_shape.frames,
                    fps=stage_1_output_shape.fps,
                    video=ModalitySpec(
                        context=v_context_p,
                        conditionings=images["stage_1_conditionings"],
                    ),
                    audio=ModalitySpec(
                        context=a_context_p,
                    ),
                    streaming_prefetch_count=streaming_prefetch_count,
                    max_batch_size=max_batch_size,
                   
                )
        else: #two_stages
            stage_1_sigmas= None
            sigmas = (
                stage_1_sigmas if stage_1_sigmas is not None else self._scheduler.execute(steps=num_inference_steps)
            ).to(dtype=torch.float32, device=cur_device)
            video_state, audio_state = self.stage_1(
                denoiser=FactoryGuidedDenoiser(
                    v_context=v_context_p,
                    a_context=a_context_p,
                    video_guider_factory=create_multimodal_guider_factory(
                        params=video_guider_params,
                        negative_context=v_context_n,
                    ),
                    audio_guider_factory=create_multimodal_guider_factory(
                        params=audio_guider_params,
                        negative_context=a_context_n,
                    ),
                ),
                sigmas=sigmas,
                noiser=noiser,
                width=stage_1_output_shape.width,
                height=stage_1_output_shape.height,
                frames=stage_1_output_shape.frames,
                fps=stage_1_output_shape.fps,
                video=ModalitySpec(context=v_context_p, conditionings=images["stage_1_conditionings"]),
                audio=ModalitySpec(context=a_context_p),
                streaming_prefetch_count=streaming_prefetch_count,
                max_batch_size=max_batch_size,
            )

        if self.upsampler is not None:
            logging.info(f"{self.infer_mode} mode ,start inferS Stage 2")     
            upscaled_video_latent = self.upsampler(video_state.latent[:1])
            
            stage_2_sigmas=STAGE_2_DISTILLED_SIGMAS
            stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=cur_device)
            if self.stage_2 is None: 
                self.stage_2=self.stage_1
            state_outputs = self.stage_2(
                denoiser=SimpleDenoiser(v_context_p, a_context_p),
                sigmas=stage_2_sigmas,
                noiser=noiser,
                width=stage_2_output_shape.width,
                height=stage_2_output_shape.height,
                frames=stage_2_output_shape.frames,
                fps=stage_2_output_shape.fps,
                video=ModalitySpec(
                    context=v_context_p,
                    conditionings=images["stage_2_conditionings"],
                    noise_scale=stage_2_sigmas[0].item(),
                    initial_latent=upscaled_video_latent,
                ),
                audio=ModalitySpec(
                    context=a_context_p,
                    noise_scale=stage_2_sigmas[0].item() if not self.infer_mode=="audio2v" else 0.0,
                    frozen=False if not self.infer_mode=="audio2v" else True,
                    initial_latent=audio_state.latent if not self.infer_mode=="audio2v" else images["encoded_audio_latent"] ,
                ),
                loop=None if not self.infer_mode=="twostages_hq" else res2s_audio_video_denoising_loop,
                streaming_prefetch_count=streaming_prefetch_count,
            )

            if self.infer_mode=="audio2v":
                video_state,audio_state=state_outputs,images["encoded_audio_latent"]
            else:
                video_state,audio_state=state_outputs 
        else:
            logging.info(f"No upsampler model,{self.infer_mode} Skipping Stage 2")
              
        return video_state.latent, audio_state.latent if self.infer_mode!="audio2v" else audio_state
            

        
    @staticmethod
    def _downsample_mask_to_latent(
        mask: torch.Tensor,
        target_latent_shape: VideoLatentShape,
    ) -> torch.Tensor:
        """
        Downsample a pixel-space mask to latent space using VAE scale factors.
        Handles causal temporal downsampling: the first frame is kept separately
        (temporal scale factor = 1 for the first frame), while the remaining
        frames are downsampled by the VAE's temporal scale factor.
        Args:
            mask: Pixel-space mask of shape (B, 1, F_pixel, H_pixel, W_pixel).
                Values in [0, 1].
            target_latent_shape: Expected latent shape after VAE encoding.
                Used to determine the target (F_latent, H_latent, W_latent).
        Returns:
            Flattened latent-space mask of shape (B, F_lat * H_lat * W_lat),
            matching the patchifier's token ordering (f, h, w).
        """
        b = mask.shape[0]
        f_lat = target_latent_shape.frames
        h_lat = target_latent_shape.height
        w_lat = target_latent_shape.width

        # Step 1: Spatial downsampling (area interpolation per frame)
        f_pix = mask.shape[2]
        spatial_down = torch.nn.functional.interpolate(
            rearrange(mask, "b 1 f h w -> (b f) 1 h w"),
            size=(h_lat, w_lat),
            mode="area",
        )
        spatial_down = rearrange(spatial_down, "(b f) 1 h w -> b 1 f h w", b=b)

        # Step 2: Causal temporal downsampling
        # First frame: kept as-is (causal VAE encodes first frame independently)
        first_frame = spatial_down[:, :, :1, :, :]  # (B, 1, 1, H_lat, W_lat)

        if f_pix > 1 and f_lat > 1:
            # Remaining frames: downsample by temporal factor via group-mean
            t = (f_pix - 1) // (f_lat - 1)  # temporal downscale factor
            assert (f_pix - 1) % (f_lat - 1) == 0, (
                f"Pixel frames ({f_pix}) not compatible with latent frames ({f_lat}): "
                f"(f_pix - 1) must be divisible by (f_lat - 1)"
            )
            rest = rearrange(spatial_down[:, :, 1:, :, :], "b 1 (f t) h w -> b 1 f t h w", t=t)
            rest = rest.mean(dim=3)  # (B, 1, F_lat-1, H_lat, W_lat)
            latent_mask = torch.cat([first_frame, rest], dim=2)  # (B, 1, F_lat, H_lat, W_lat)
        else:
            latent_mask = first_frame

        # Flatten to (B, F_lat * H_lat * W_lat) matching patchifier token order (f, h, w)
        return rearrange(latent_mask, "b 1 f h w -> b (f h w)")


def load_pipeline_ltx(
    checkpoint_path: str,
    distilled_lora: str,
    loras: list[str],
    device: str = "cpu",
    quantization: QuantizationPolicy | None = None,
    infer_mode="distilled",
    offload=False,
    
) -> OmniPipeline:
    
    pipeline = OmniPipeline(
        checkpoint_path=checkpoint_path,
        distilled_lora=distilled_lora,
        loras=loras,
        device=device,
        quantization=quantization,
        infer_mode=infer_mode,
        offload=offload,
    )
    return pipeline



def load_mask_video(
    mask_path: str,
    height: int,
    width: int,
    num_frames: int,
    device,
) -> torch.Tensor:
    """Load a mask video and return a pixel-space tensor of shape (1, 1, F, H, W).
    The mask video is loaded, resized to (height, width), converted to
    grayscale, and normalised to [0, 1].
    Args:
        mask_path: Path to the mask video file.
        height: Target height in pixels.
        width: Target width in pixels.
        num_frames: Maximum number of frames to load.
    Returns:
        Tensor of shape ``(1, 1, F, H, W)`` with values in ``[0, 1]``.
    """
    if isinstance(mask_path, str):
        mask_video = load_video_conditioning(
            video_path=mask_path,
            height=height,
            width=width,
            frame_cap=num_frames,
            dtype=torch.bfloat16,
            device=device,
        )
        mask = mask_video.mean(dim=1, keepdim=True)  # (1, 1, F, H, W)
        # mask_video shape: (1, C, F, H, W) — take mean over channels for grayscale
    else:
        mask = mask_path.unsqueeze(0).unsqueeze(0) # FHW-->BFCHW
    
    # Normalise to [0, 1] — load_video_conditioning applies normalize_latent,
    # so undo that: values are in [-1, 1], remap to [0, 1]
    mask = (mask + 1.0) / 2.0
    return mask.clamp(0.0, 1.0)

