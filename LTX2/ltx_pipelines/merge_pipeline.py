import logging
from collections.abc import Iterator
from einops import rearrange
import torch
from ..ltx_core.components.diffusion_steps import EulerDiffusionStep
from ..ltx_core.components.guiders import (
    MultiModalGuiderFactory,MultiModalGuider,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ..ltx_core.conditioning import (
    ConditioningItemAttentionStrengthWrapper,
    VideoConditionByReferenceLatent,
)
from ..ltx_core.components.patchifiers import get_pixel_coords
from ..ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ..ltx_core.components.diffusion_steps import Res2sDiffusionStep
from ..ltx_core.components.noisers import GaussianNoiser
from ..ltx_core.components.protocols import DiffusionStepProtocol
from ..ltx_core.components.schedulers import LTX2Scheduler
from ..ltx_core.loader import LoraPathStrengthAndSDOps
from ..ltx_core.tools import VideoLatentShape
from ..ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ..ltx_core.model.video_vae import decode_video as vae_decode_video
from ..ltx_core.types import LatentState, VideoPixelShape,Audio
from ..ltx_core.quantization import QuantizationPolicy
from ..ltx_pipelines.utils import ModelLedger
from .utils.constants import (
    STAGE_2_DISTILLED_SIGMA_VALUES,DISTILLED_SIGMA_VALUES
)
from ..ltx_core.tools import LatentTools
from dataclasses import dataclass
from ..ltx_core.types import (
    Audio,
    AudioLatentShape,
    LatentState,
    SpatioTemporalScaleFactors,
    VideoPixelShape,
)
from .utils.args import (
    ImageConditioningInput,
)
from ..ltx_pipelines.utils.helpers import multi_modal_guider_denoising_func,denoise_video_only,noise_audio_state,noise_video_state
from .utils import (
    ModelLedger,
    assert_resolution,
    cleanup_memory,
    combined_image_conditionings,
    denoise_audio_video,
    euler_denoising_loop,
    res2s_audio_video_denoising_loop,
    multi_modal_guider_factory_denoising_func,
    simple_denoising_func,
)
from .utils.media_io import load_video_conditioning
from .utils.samplers import euler_denoising_loop
from .utils.types import PipelineComponents

#device = get_device()
def _encode_video_for_retake(
    video_encoder: torch.nn.Module,
    video_path,# str or torch.Tensor
    output_shape: VideoPixelShape,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Load video and encode to latents."""
    if isinstance(video_path, str):
        pixel_video = load_video_conditioning(
            video_path=video_path,
            height=output_shape.height,
            width=output_shape.width,
            frame_cap=output_shape.frames,
            dtype=dtype,
            device=device,
        )  # (1, C, F, H, W)
    else:
        pixel_video = video_path.permute(3, 0, 1, 2).unsqueeze(0) # FHWC-->BCFHW
    return video_encoder(pixel_video)


def _encode_audio_for_retake(
    audio_encoder: torch.nn.Module,
    waveform: torch.Tensor,
    waveform_sr: int,
    output_shape: VideoPixelShape,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode audio to latents and trim/pad to match output_shape."""
    waveform_batch = waveform.unsqueeze(0) if waveform.dim() == 2 else waveform
    initial_audio_latent = vae_encode_audio(
        Audio(waveform=waveform_batch.to(dtype), sampling_rate=waveform_sr), audio_encoder, None
    )
    expected_audio_shape = AudioLatentShape.from_video_pixel_shape(output_shape)
    expected_frames = expected_audio_shape.frames
    actual_frames = initial_audio_latent.shape[2]
    if actual_frames > expected_frames:
        initial_audio_latent = initial_audio_latent[:, :, :expected_frames, :]
    elif actual_frames < expected_frames:
        pad = torch.zeros(
            initial_audio_latent.shape[0],
            initial_audio_latent.shape[1],
            expected_frames - actual_frames,
            initial_audio_latent.shape[3],
            device=initial_audio_latent.device,
            dtype=initial_audio_latent.dtype,
        )
        initial_audio_latent = torch.cat([initial_audio_latent, pad], dim=2)
    return initial_audio_latent

@dataclass(frozen=True)
class TemporalRegionMask:
    """Conditioning item that sets ``denoise_mask = 0`` outside a time range
    and ``1`` inside, so only the specified temporal region is regenerated.
    Uses ``start_time`` and ``end_time`` in seconds. Works in *patchified*
    (token) space using the patchifier's ``get_patch_grid_bounds``: for video
    coords are latent frame indices (converted from seconds via ``fps``), for
    audio coords are already in seconds.
    """

    start_time: float  # seconds, inclusive
    end_time: float  # seconds, exclusive
    fps: float

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        coords = latent_tools.patchifier.get_patch_grid_bounds(
            latent_tools.target_shape, device=latent_state.denoise_mask.device
        )
        # coords: [B, 3, N, 2] (video) or [B, 1, N, 2] (audio); temporal dim is index 0
        if coords.shape[1] == 1:
            # Audio: patchifier returns seconds
            t_start = coords[:, 0, :, 0]  # [B, N]
            t_end = coords[:, 0, :, 1]  # [B, N]
            in_region = (t_end > self.start_time) & (t_start < self.end_time)
        else:
            # Video: get pixel bounds per patch, find patches for start/end frame, read latent from coords.
            scale_factors = getattr(latent_tools, "scale_factors", SpatioTemporalScaleFactors.default())
            pixel_bounds = get_pixel_coords(coords, scale_factors, causal_fix=getattr(latent_tools, "causal_fix", True))
            timestamp_bounds = pixel_bounds[0, 0] / self.fps
            t_start, t_end = timestamp_bounds.unbind(dim=-1)
            in_region = (t_end > self.start_time) & (t_start < self.end_time)
        state = latent_state.clone()
        mask_val = in_region.to(state.denoise_mask.dtype)
        if state.denoise_mask.dim() == 3:
            mask_val = mask_val.unsqueeze(-1)
        state.denoise_mask.copy_(mask_val)
        return state


class OmniPipeline:
    """
    simple pipeline
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
        infer_mode="distilled", #["distilled","one_stage","two_stages","keyframe","ic_lora","audio2v","retake","twostages_hq"]
    ):
        self.device = device
        self.infer_mode=infer_mode
        self.isdistilled =True if "distill" in checkpoint_path.lower() or len(distilled_lora)>=1 else False
        self.dtype = torch.bfloat16
        self.distilled_lora=distilled_lora
        if self.infer_mode=="ic_lora":
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
            self.stage_2_model_ledger = ModelLedger(
                dtype=self.dtype,
                device=device,
                checkpoint_path=checkpoint_path,
                spatial_upsampler_path=spatial_upsampler_path,
                gemma_root_path=gemma_root,
                loras=[],
                quantization=quantization,
                gguf_dit=gguf_dit,
                load_mode= "dit",
            )
        else:
            if self.infer_mode=="twostages_hq":       
                if  len(distilled_lora)>=1:
                    distilled_lora_stage_1 = LoraPathStrengthAndSDOps(
                        path=distilled_lora[0].path,
                        strength=0.8,
                        sd_ops=distilled_lora[0].sd_ops,
                    )
                    loras=(*loras, distilled_lora_stage_1)
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
        if distilled_lora and not self.infer_mode=="ic_lora":
            if self.infer_mode=="twostages_hq":
                if  len(distilled_lora)>=1:
                    distilled_lora_stage_2 = LoraPathStrengthAndSDOps(
                        path=distilled_lora[0].path,
                        strength=0.8,
                        sd_ops=distilled_lora[0].sd_ops,
                    )
                    self.stage_2_model_ledger = self.stage_1_model_ledger.with_loras(
                        loras=(*loras, distilled_lora_stage_2),
                    )
            else:    
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
            if  (self.distilled_lora and not self.infer_mode=="ic_lora") or self.infer_mode=="ic_lora":
                self.transformer1=None
                cleanup_memory()
                self.transformer2=self.stage_2_model_ledger.transformer()
            else:
                self.transformer2=self.transformer1
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
        offload=False,
        stage_one=True,
        v_lat=None,
        a_lat=None,
        video_conditioning: list[tuple[str, float]]=[],
        conditioning_attention_strength: float = 1.0,
        conditioning_attention_mask: torch.Tensor | None = None,
        block_group_size=2,

    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        #assert_resolution(height=height, width=width, is_two_stage=True)
        cur_device = self.device if not offload else "cuda"
        if self.infer_mode=="retake":
            seed = torch.randint(0, 2**31, (1,), device=self.device).item() if seed < 0 else seed
        generator = torch.Generator(device=cur_device).manual_seed(seed)

        noiser = GaussianNoiser(generator=generator)
        if self.infer_mode=="twostages_hq":
            stepper = Res2sDiffusionStep()
        else:
            stepper = EulerDiffusionStep()
        
        dtype = torch.bfloat16
        
        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = context_n

        torch.cuda.synchronize()
        cleanup_memory()
        
        if offload:
            from ..ltx_core.model.transformer.model import BlockGPUManager         
            gpu_manager = BlockGPUManager(device="cuda",block_group_size=block_group_size)
            gpu_manager.setup_for_inference(self.transformer1.velocity_model)
        else:
            gpu_manager = None
        def denoising_loop(
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
                        gpu_manager=gpu_manager,  # noqa: F821
                    ),
                )
        def denoising_loop2(
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
        def first_stage_denoising_loop_v(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=multi_modal_guider_denoising_func(
                    video_guider=MultiModalGuider(
                        params=video_guider_params,
                        negative_context=v_context_n,
                    ),
                    audio_guider=MultiModalGuider(
                        params=MultiModalGuiderParams(),
                    ),
                    v_context=v_context_p,
                    a_context=a_context_p,
                    transformer=self.transformer1,
                    gpu_manager=gpu_manager,  # noqa: F821
                ),
            )
        if self.isdistilled:
            denoise_fn_r = simple_denoising_func(
                video_context=v_context_p,
                audio_context=a_context_p,
                transformer=self.transformer1,
                gpu_manager=gpu_manager,
            )
        else:
            video_guider = MultiModalGuider(
                params=video_guider_params,
                negative_context=v_context_n,
            )
            audio_guider = MultiModalGuider(
                params=audio_guider_params,
                negative_context=a_context_n,
            )
            denoise_fn_r = multi_modal_guider_denoising_func(
                video_guider,
                audio_guider,
                v_context=v_context_p,
                a_context=a_context_p,
                transformer=self.transformer1,
                gpu_manager=gpu_manager,
            )
        def denoising_loop_r(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=denoise_fn_r,
            )
        
        if self.infer_mode=="twostages_hq":
            denoising_loop_fn=res2s_audio_video_denoising_loop
        else:
            denoising_loop_fn=euler_denoising_loop
        def first_stage_denoising_loop(
                sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
            ) -> tuple[LatentState, LatentState]:
            return denoising_loop_fn(
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
        def second_stage_denoising_loop(
                sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
            ) -> tuple[LatentState, LatentState]:
            return denoising_loop_fn(
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
        if not self.infer_mode=="distilled":
            if self.infer_mode=="twostages_hq":
                empty_latent = torch.empty(VideoLatentShape.from_pixel_shape(images["stage_1_output_shape"]).to_torch_shape())
                stage_1_sigmas = (
                    LTX2Scheduler()
                    .execute(latent=empty_latent, steps=num_inference_steps)
                    .to(dtype=torch.float32, device=cur_device)
                )
            elif self.infer_mode=="retake":
                stage_1_sigmas = (
                    torch.tensor(DISTILLED_SIGMA_VALUES) if self.isdistilled else LTX2Scheduler().execute(steps=num_inference_steps)
                ).to(dtype=torch.float32, device=cur_device)

            else:
                stage_1_sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=cur_device)
        else:
            stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(cur_device)

        stage_2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(cur_device)
        
        if stage_one:
            if self.infer_mode=="distilled" or self.infer_mode=="ic_lora":
                denoise_fn=denoising_loop
            else:
                denoise_fn=first_stage_denoising_loop
            if self.infer_mode=="audio2v":
                video_state = denoise_video_only(
                    output_shape=images["stage_1_output_shape"],
                    conditionings=images["stage_1_conditionings"],
                    noiser=noiser,
                    sigmas=stage_1_sigmas,
                    stepper=stepper,
                    denoising_loop_fn=first_stage_denoising_loop_v,
                    components=self.pipeline_components,
                    dtype=dtype,
                    device=cur_device,
                    initial_audio_latent=images["encoded_audio_latent"],
                )

                return video_state.latent, images["encoded_audio_latent"] 
            elif self.infer_mode=="retake":
                video_state, video_tools = noise_video_state(
                    output_shape=images["stage_1_output_shape"],
                    noiser=noiser,
                    conditionings=images["stage_2_conditionings"],
                    components=self.pipeline_components,
                    dtype=dtype,
                    device=cur_device,
                    initial_latent=images["stage_1_conditionings"],
                )
                audio_state, audio_tools = noise_audio_state(
                    output_shape=images["stage_1_output_shape"],
                    noiser=noiser,
                    conditionings=images["decoded_audio"],
                    components=self.pipeline_components,
                    dtype=dtype,
                    device=cur_device,
                    initial_latent=images["encoded_audio_latent"],
                )
                video_state, audio_state = denoising_loop_r(stage_1_sigmas, video_state, audio_state, stepper)
                video_state = video_tools.clear_conditioning(video_state)
                video_state = video_tools.unpatchify(video_state)
                audio_state = audio_tools.clear_conditioning(audio_state)
                audio_state = audio_tools.unpatchify(audio_state)

            else:
                video_state,audio_state = denoise_audio_video(
                    output_shape=images["stage_1_output_shape"],
                    conditionings=images["stage_1_conditionings"],
                    noiser=noiser,
                    sigmas=stage_1_sigmas,
                    stepper=stepper,
                    denoising_loop_fn=denoise_fn,
                    components=self.pipeline_components,
                    dtype=dtype,
                    device=cur_device,
                )
            torch.cuda.synchronize()
            if gpu_manager is not None and self.infer_mode=="one_stage":
                gpu_manager.unload_blocks_to_cpu()
            cleanup_memory()
            return video_state.latent, audio_state.latent
        else:
            self.load_dit(False)
            if not self.infer_mode=="distilled" :
                if offload:
                    from ..ltx_core.model.transformer.model import BlockGPUManager         
                    gpu_manager = BlockGPUManager(device="cuda",block_group_size=block_group_size)
                    gpu_manager.setup_for_inference(self.transformer2.velocity_model)
                else:
                    gpu_manager = None
                denoise2_fn=second_stage_denoising_loop
            else:
                denoise2_fn=denoising_loop2
            if self.infer_mode=="audio2v":
                sate_loop_fn=denoise_video_only
            else:
                sate_loop_fn=denoise_audio_video

            state_outputs = sate_loop_fn(
                output_shape=images["stage_2_output_shape"],
                conditionings=images["stage_2_conditionings"],
                noiser=noiser,
                sigmas=stage_2_sigmas,
                stepper=stepper,
                denoising_loop_fn=denoise2_fn,
                components=self.pipeline_components,
                dtype=dtype,
                device=cur_device,
                noise_scale=stage_2_sigmas[0],
                initial_video_latent=v_lat,
                initial_audio_latent=a_lat,
            )
            if self.infer_mode=="audio2v":
                video_state,audio_state=state_outputs,images["encoded_audio_latent"]
            else:
                video_state,audio_state=state_outputs 
            torch.cuda.synchronize()
            if gpu_manager is not None:
                gpu_manager.unload_blocks_to_cpu()
            # del transformer
            # del video_encoder
            cleanup_memory()
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
    spatial_upsampler_path: str,
    gemma_root: str,
    loras: list[str],
    device: str = "cpu",
    quantization: QuantizationPolicy | None = None,
    gguf_dit=False,
    infer_mode="distilled",
    
) -> OmniPipeline:
    
    pipeline = OmniPipeline(
        checkpoint_path=checkpoint_path,
        distilled_lora=distilled_lora,
        spatial_upsampler_path=spatial_upsampler_path,
        gemma_root=gemma_root,
        loras=loras,
        device=device,
        quantization=quantization,
        gguf_dit=gguf_dit,
        infer_mode=infer_mode,
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


def downsample_mask_to_latent(
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

def ic_create_conditionings(
    images: list[ImageConditioningInput],
    video_conditioning: list[tuple[str, float]],
    height: int,
    width: int,
    num_frames: int,
    video_encoder,
    conditioning_attention_strength: float = 1.0,
    conditioning_attention_mask: torch.Tensor | None = None, 
    dtype=torch.bfloat16,
    device: torch.device = torch.device("cuda"),
    ) :
    """
    Create conditioning items for video generation.
    Args:
        conditioning_attention_strength: Scalar attention weight in [0, 1].
            If conditioning_attention_mask is also provided, the downsampled mask
            is multiplied by this strength. Otherwise this scalar is passed
            directly as the attention mask.
        conditioning_attention_mask: Optional pixel-space attention mask with shape
            (B, 1, F_pixel, H_pixel, W_pixel) matching the reference video's
            pixel dimensions. Downsampled to latent space with causal temporal
            handling, then multiplied by conditioning_attention_strength.
    Returns:
        List of conditioning items. IC-LoRA conditionings are appended last.
    """
    conditionings = combined_image_conditionings(
        images=images,
        height=height,
        width=width,
        video_encoder=video_encoder,
        dtype=dtype,
        device=device,
    )

    # Calculate scaled dimensions for reference video conditioning.
    # IC-LoRAs trained with downscaled reference videos expect the same ratio at inference.
    scale = 1 # TODO  Some ic lora scale is not supported
    # if scale != 1 and (height % scale != 0 or width % scale != 0):
    #     raise ValueError(
    #         f"Output dimensions ({height}x{width}) must be divisible by reference_downscale_factor ({scale})"
    #     )
    ref_height = height // scale
    ref_width = width // scale

    for video_path, strength in video_conditioning:
        # Load video at scaled-down resolution (if scale > 1)
        if isinstance(video_path, str): 
            video = load_video_conditioning(
                video_path=video_path,
                height=ref_height,
                width=ref_width,
                frame_cap=num_frames,
                dtype=dtype,
                device=device,
            )
        else:
            video = video_path.permute(3, 0, 1, 2).unsqueeze(0) # FHWC-->BCFHW
        encoded_video = video_encoder(video)
        reference_video_shape = VideoLatentShape.from_torch_shape(encoded_video.shape)

        # Build attention_mask for ConditioningItemAttentionStrengthWrapper
        if conditioning_attention_mask is not None:
            # Downsample pixel-space mask to latent space, then scale by strength
            latent_mask = downsample_mask_to_latent(
                mask=conditioning_attention_mask,
                target_latent_shape=reference_video_shape,
            )
            attn_mask = latent_mask * conditioning_attention_strength
        elif conditioning_attention_strength < 1.0:
            # Use scalar strength only
            attn_mask = conditioning_attention_strength
        else:
            attn_mask = None

        cond = VideoConditionByReferenceLatent(
            latent=encoded_video,
            downscale_factor=scale,
            strength=strength,
        )
        if attn_mask is not None:
            cond = ConditioningItemAttentionStrengthWrapper(cond, attention_mask=attn_mask)
        conditionings.append(cond)

    if video_conditioning:
        logging.info(f"[IC-LoRA] Added {len(video_conditioning)} video conditioning(s)")

    return conditionings

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