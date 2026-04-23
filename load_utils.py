 # !/usr/bin/env python
# -*- coding: UTF-8 -*-
import time
import torch
import os
import gc
from einops import rearrange
from dataclasses import dataclass

import folder_paths
import comfy.model_management as mm
from .model_loader_utils import tensor2pillist_upscale,tensor_upscale

from .LTX2.ltx_pipelines.utils.blocks import (
    AudioDecoder,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
    AudioConditioner,
)
from .LTX2.ltx_core.types import Audio, VideoLatentShape,SpatioTemporalScaleFactors, VideoPixelShape
from .LTX2.ltx_core.loader.registry import DummyRegistry
from .LTX2.ltx_pipelines.merge_pipeline import load_pipeline_ltx,load_mask_video
from .LTX2.ltx_core.conditioning.types.noise_mask_cond import TemporalRegionMask
from .LTX2.ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP,LoraPathStrengthAndSDOps
from .LTX2.ltx_core.quantization import QuantizationPolicy
from .LTX2.ltx_pipelines.utils.helpers import combined_image_conditionings,cleanup_memory
from .LTX2.ltx_pipelines.utils.args import ImageConditioningInput
from .LTX2.ltx_core.components.guiders import MultiModalGuiderParams
from .LTX2.ltx_core.conditioning import ConditioningItem,ConditioningItemAttentionStrengthWrapper,VideoConditionByReferenceLatent
from .LTX2.ltx_pipelines.utils.helpers import audio_latent_from_file,video_latent_from_file
from .LTX2.ltx_pipelines.utils.media_io import apply_start_time_max_duration,get_videostream_metadata
from .LTX2.ltx_core.types import Audio, AudioLatentShape
from .LTX2.ltx_core.model.video_vae import TilingConfig
from .LTX2.ltx_core.model.audio_vae import encode_audio as vae_encode_audio

node_cr_path_ = os.path.dirname(os.path.abspath(__file__))

def clear_comfyui_cache():
    cf_models=mm.loaded_models()
    try:
        for pipe in cf_models:
            pipe.unpatch_model(device_to=torch.device("cpu"))
            #print(f"Unpatching models.{pipe}")
    except: pass
    mm.soft_empty_cache()
    torch.cuda.empty_cache()
    max_gpu_memory = torch.cuda.max_memory_allocated()
    print(f"After Max GPU memory allocated: {max_gpu_memory / 1000 ** 3:.2f} GB")

def load_model( dit, gguf, lora, distilled_lora, sampling_mode,offload,device): 
    dit_path=folder_paths.get_full_path("diffusion_models", dit) if dit != "none" else None
    gguf_path=folder_paths.get_full_path("gguf", gguf) if gguf != "none" else None 
    lora_path=folder_paths.get_full_path("loras", lora) if lora != "none" else None
    distilled_lora_path=folder_paths.get_full_path("loras", distilled_lora) if distilled_lora != "none" else None

    quantization=None
    if dit_path is not None:
        quantization=QuantizationPolicy.fp8_cast()  if "fp8" in dit_path.lower() else None
    checkpoint_path=dit_path or gguf_path
    lora_ops=[LoraPathStrengthAndSDOps(lora_path,1.0,None)] if lora_path is not None else []
    lora_ops_distilled=[LoraPathStrengthAndSDOps(distilled_lora_path,0.6,LTXV_LORA_COMFY_RENAMING_MAP)] if distilled_lora_path is not None else []

    model=load_pipeline_ltx(checkpoint_path, lora_ops_distilled,lora_ops,device,quantization,sampling_mode,offload)
    return model
    

def load_clip(clip,connector,node_cr_path,infer_device):
    gemma_path=folder_paths.get_full_path("gguf", clip) if clip != "none" else None
    connector_path=folder_paths.get_full_path("checkpoints", connector) if connector != "none" else None
    gemma_root=os.path.join(node_cr_path,"LTX2/gemma")
    assert connector_path is not None and gemma_path is not None,"Please provide a gemma_path and connector_path"
    prompt_encoder=PromptEncoder(gemma_path, gemma_root, torch.bfloat16,  infer_device, registry=DummyRegistry(),connector_path=connector_path)    
    return prompt_encoder

def encoder_text(prompt_encoder,prompt,negative_prompt,images,enhance_prompt,save_emb,streaming_prefetch_count):
    prompts = [p for p in [prompt, negative_prompt] if p.strip()] 
    with torch.no_grad():
        context = prompt_encoder(
            prompts,
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images if images is not None else None,
            enhance_prompt_seed=42,
            streaming_prefetch_count=streaming_prefetch_count,
        )

    torch.cuda.empty_cache()
    gc.collect()
    prompt_embeds=torch.concat([context[0].video_encoding,context[0].audio_encoding], dim=-1)
    #print(prompt_embeds.shape) #torch.Size([1, 1024, 6144])
    if len(context)>1:
        negative_prompt_embeds=torch.concat([context[1].video_encoding,context[1].audio_encoding], dim=-1)
        #print(negative_prompt_embeds.shape) #torch.Size([1, 1024, 6144])
    else:
        negative_prompt_embeds=None #distilled does not have negative prompts
    if save_emb:
        save_lat_emb(prompt_embeds,negative_prompt_embeds)
    positive=[[prompt_embeds,{"pooled_output": None}]]
    negative=[[negative_prompt_embeds,{"pooled_output": None}]]

    return positive, negative

def load_vae(vae,device):
    vae_path=folder_paths.get_full_path("vae", vae) if vae != "none" else None
    assert vae_path is not None,"Please provide a vae_path"
    decoder=VideoDecoder(vae_path, torch.bfloat16, device, registry=DummyRegistry())
    encoder=ImageConditioner(vae_path, torch.bfloat16, device, registry=DummyRegistry())
    return decoder,encoder

def load_audio_vae(audio_vae,device):
    vae_path=folder_paths.get_full_path("vae", audio_vae) if audio_vae != "none" else None
    assert vae_path is not None,"Please provide a vae_path"
    decoder=AudioDecoder(vae_path, torch.bfloat16, device, registry=DummyRegistry())
    encoder=AudioConditioner(vae_path, torch.bfloat16, device, registry=DummyRegistry())
    return decoder,encoder

def infer_latent_shape(width,height,num_frames,frame_rate):
    stage_1_output_shape = VideoPixelShape(batch=1,frames=num_frames,width=width,height=height,fps=frame_rate,)
    stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width*2, height=height*2, fps=frame_rate)
    return stage_1_output_shape,stage_2_output_shape


def decoder_video(video_decoder,latent,device,tile=False,):

    lat=latent["samples"]
    seed=latent.get("seed",0)
    generator=torch.manual_seed(seed) 

    if tile:
        video = video_decoder(lat,tiling_config=TilingConfig.default(), generator=generator,) #Decoded chunk [f, h, w, c], uint8 in [0, 255]. 
    else:
        video = video_decoder(lat, None, generator) #Decoded chunk [f, h, w, c], uint8 in [0, 255].       
    video_frames = list(video)
    if len(video_frames) == 1:
        video_result = video_frames[0].unsqueeze(0)  # 添加批次维度 [1, f, h, w, c]
    else:
        video_result = torch.cat(video_frames, dim=0)  # [total_f, h, w, c]
        video_result = video_result.unsqueeze(0) 
    torch.cuda.empty_cache()
    gc.collect()
    #print(video_result.shape) ##torch.Size([1, 81, 512, 768, 3])
    video=video_result.float().cpu()[0]/255.0
    return video


def get_latents(image_conditioner,image,audio_encoder,audio,width,height,device,num_frames,frame_rate,stage_1_output_shape,stage_2_output_shape,strength=0.8,audio_start_time=0,audio_max_duration=0.0,ic_lora_video=None,ic_lora_mask=None):
    images,stage_1_conditionings,stage_2_conditionings,initial_video_latent=[],[],[],None
    regenerate_video = True
    dtype=torch.bfloat16
    start_time=float(audio_start_time) if regenerate_video else 0.0
    end_time=audio_max_duration if regenerate_video else 0.0
    if image is not None and image_conditioner is not None:
        image_list=tensor2pillist_upscale(image, width, height)
        if len(image_list)>5: #retake
            video_scale = SpatioTemporalScaleFactors.default()
            image_list=tensor_upscale(image, width, height).to(device,dtype)
            if (image_list.shape[2] - 1) % video_scale.time != 0:
                snapped = ((num_frames - 1) // video_scale.time) * video_scale.time + 1
                image_list = image_list[:snapped, :, :,:]
            image_list=image_list[:num_frames, :, :,:]if num_frames<image_list.shape[2] else image_list
            video_path=""
            output_shape = get_videostream_metadata(video_path)
            initial_video_latent = image_conditioner(
                lambda enc: video_latent_from_file(
                    video_encoder=enc,
                    file_path=video_path,
                    output_shape=output_shape,
                    dtype=dtype,
                    device=device,
                )
            )
            
            # stage_1_conditionings = _encode_video_for_retake(
            #     video_encoder=video_encoder,
            #     video_path=image_list,
            #     output_shape=stage_1_output_shape,
            #     dtype=dtype,
            #     device=device,
            #     )
            
            # stage_2_conditionings: list[ConditioningItem] = [
            #     TemporalRegionMask(
            #         start_time=float(audio_start_time) if regenerate_video else 0.0,
            #         end_time=audio_max_duration if regenerate_video else 0.0,
            #         fps=frame_rate,
            #     )
            # ]
        else:
            for i,img in enumerate(image_list):
                img_in=ImageConditioningInput(path=img, strength=strength, frame_idx=i, )
                images.append(img_in)
            # if len(image_list)>1: # keyframe
            #     cond_fn=image_conditionings_by_adding_guiding_latent
            # else:
            #     cond_fn=combined_image_conditionings
            torch.cuda.empty_cache()
            cleanup_memory()
            if ic_lora_video is not None: #ic_lora
                conditioning_attention_strength=1.0
                video_conditioning=[(tensor_upscale(image, width, height).to(device,dtype),1.0)]
                conditioning_attention_mask = None
                conditioning_attention_strength = 1.0
                if ic_lora_mask is not None:
                    mask_strength = 1.0
                    conditioning_attention_strength = mask_strength
                    conditioning_attention_mask = load_mask_video(
                        mask_path=ic_lora_mask,
                        height=height,  # Stage 1 operates at half resolution
                        width=width,
                        num_frames=num_frames,
                    )
                stage_1_conditionings = image_conditioner(
                    lambda enc: ic_create_conditionings(
                        images=images,
                        video_conditioning=video_conditioning,
                        height=stage_1_output_shape.height,
                        width=stage_1_output_shape.width,
                        video_encoder=enc,
                        num_frames=num_frames,
                        conditioning_attention_strength=conditioning_attention_strength,
                        conditioning_attention_mask=conditioning_attention_mask,
                        dtype=dtype,
                        device = device,
                    )
                )
            else:
                stage_1_conditionings = image_conditioner(
                    lambda enc: combined_image_conditionings(
                        images=images,
                        height=stage_1_output_shape.height,
                        width=stage_1_output_shape.width,
                        video_encoder=enc,
                        dtype=dtype,
                        device=device,
                    )
                )

            stage_2_conditionings = image_conditioner(
                lambda enc: combined_image_conditionings(
                    images=images,
                    height=stage_2_output_shape.height,
                    width=stage_2_output_shape.width,
                    video_encoder=enc,
                    dtype=dtype,
                    device=device,
                )
            )
       
    output={"stage_1_conditionings":stage_1_conditionings,"stage_2_conditionings":stage_2_conditionings,"encoded_video_latent":initial_video_latent,
            "stage_1_output_shape":stage_1_output_shape,"stage_2_output_shape":stage_2_output_shape,"start_time":start_time,"end_time":end_time,
            "width":width,"height":height,"num_frames":num_frames,"frame_rate":frame_rate}
    
    decoded_audio,encoded_audio_latent=[],None

    if audio is not None and audio_encoder is not None:
        if len(image_list)>5: #retake
            waveform=audio["waveform"].squeeze(0)
            waveform_batch = waveform.unsqueeze(0) if waveform.dim() == 2 else waveform
            sample_rate = audio["sample_rate"]
            audio_in=Audio(waveform=waveform_batch.to(dtype), sampling_rate=sample_rate)
            encoded_audio_latent = audio_encoder(
                lambda enc: audio_latent_from_file(
                    audio_encoder=enc,
                    file_path=audio_in,
                    output_shape=stage_2_output_shape,
                    dtype=dtype,
                    device=device,
                )
            )
            decoded_audio = [
                TemporalRegionMask(
                    start_time=float(audio_start_time) if regenerate_video else 0.0,
                    end_time=audio_max_duration if regenerate_video else 0.0,
                    fps=frame_rate,
                )
                ]
        else:
            audio_max_duration =None if not audio_max_duration > 0 else audio_max_duration
            encoded_audio_latent,decoded_audio=encoder_audio(audio_encoder,audio,num_frames,frame_rate, device,audio_start_time,audio_max_duration)
    audio_output={"decoded_audio":decoded_audio,"encoded_audio_latent":encoded_audio_latent,}
    output.update(audio_output)
    return output


def decoder_audio(audio_decoder,audio_latents,device):
    samples=audio_decoder(audio_latents["samples"])
    torch.cuda.empty_cache()
    gc.collect()
    return {"waveform": samples.waveform.contiguous().reshape(1, -1).cpu().float().unsqueeze(0), "sample_rate": samples.sampling_rate}

def encoder_audio(audio_encoder,audio,num_frames,frame_rate,device,audio_start_time, audio_max_duration):

    waveform = apply_start_time_max_duration(audio["waveform"].squeeze(0), audio["sample_rate"], audio_start_time, audio_max_duration)

    decoded_audio=Audio(waveform=waveform.to(device).unsqueeze(0), sampling_rate=audio["sample_rate"])

    encoded_audio_latent = audio_encoder(lambda enc: vae_encode_audio(decoded_audio, enc, None))

    torch.cuda.empty_cache()
    gc.collect()
    audio_shape = AudioLatentShape.from_duration(batch=1, duration=num_frames / frame_rate, channels=8, mel_bins=16)
    encoded_audio_latent = encoded_audio_latent[:, :, : audio_shape.frames]
    #print(encoded_audio_latent.shape) #torch.Size([1, 8, 126, 16])
    #print(waveform.shape) #torch.Size([2, 1409024])
    audio["waveform"]=waveform.unsqueeze(0)
    
    return encoded_audio_latent,audio


def inference_ltx2(model, positive,negative,latents,seed, steps,cfg,block_group_size,spatial_upsampler,encoder):

    video_features_size = 4096 if positive[0][0].shape[2] == 6144 else 3840
    context_p=[positive[0][0][..., :video_features_size],positive[0][0][..., video_features_size:] ]
    if negative is not None:
        context_n = [negative[0][0][..., :video_features_size],negative[0][0][..., video_features_size:] ]
    else:
        context_n = None
    #num_frames = latents["num_frames"]
    #tiling_config = TilingConfig.default()
    #video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
    video_guider_params=MultiModalGuiderParams(
        cfg_scale=cfg["video_cfg_guidance_scale"],
        stg_scale=cfg["video_stg_guidance_scale"],
        rescale_scale=cfg["video_rescale_scale"],
        modality_scale=cfg["a2v_guidance_scale"],
        skip_step=cfg["video_skip_step"],
        stg_blocks=cfg["video_stg_blocks"],
        )
    audio_guider_params=MultiModalGuiderParams(
        cfg_scale=cfg["audio_cfg_guidance_scale"],
        stg_scale=cfg["audio_stg_guidance_scale"],
        rescale_scale=cfg["audio_rescale_scale"],
        modality_scale=cfg["v2a_guidance_scale"],
        skip_step=cfg["audio_skip_step"],
        stg_blocks=cfg["audio_stg_blocks"],
    )
    spatial_upsampler_path=folder_paths.get_full_path("latent_upscale_models", spatial_upsampler) if spatial_upsampler != "none" else None
    if spatial_upsampler_path is not None and encoder is not None:
        upsampler = VideoUpsampler(
            None, spatial_upsampler_path, torch.bfloat16, torch.device("cuda"), registry=DummyRegistry()
        )
        upsampler.video_encoder=encoder._build_encoder()
    else:
        upsampler = None
    block_group_size= None if block_group_size ==0 else block_group_size
    
    video, audio = model(
        seed=seed,
        num_inference_steps=steps,
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params,
        images=latents,
        context_p=context_p,
        context_n=context_n,
        streaming_prefetch_count=block_group_size,
        upsampler = upsampler,
    )
    
    video_latents={"samples": video,"seed":seed}
    audio_latents={"samples": audio}
    # default_data1_path = os.path.join(folder_paths.get_output_directory(),f"raw_latents_sm.pt")
    # default_data2_path = os.path.join(folder_paths.get_output_directory(),f"raw_audio_latents_sm.pt")
    # torch.save(video_latents,default_data1_path)
    # torch.save(audio_latents,default_data2_path)
    return  video_latents, audio_latents


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
    dtype: torch.dtype = torch.float16,
    device: torch.device = torch.device("cuda"),

    ) -> list[ConditioningItem]:
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
        if scale != 1 and (height % scale != 0 or width % scale != 0):
            raise ValueError(
                f"Output dimensions ({height}x{width}) must be divisible by reference_downscale_factor ({scale})"
            )
        ref_height = height // scale
        ref_width = width // scale

        for video_path, strength in video_conditioning:
            # Load video at scaled-down resolution (if scale > 1)
            # frame_gen = decode_video_by_frame(path=video_path, frame_cap=num_frames, device=self.device)
            # video = video_preprocess(frame_gen, ref_height, ref_width, self.dtype, self.device)
            # if isinstance(video_path, str): 
            #     video = load_video_conditioning(
            #         video_path=video_path,
            #         height=ref_height,
            #         width=ref_width,
            #         frame_cap=num_frames,
            #         dtype=dtype,
            #         device=device,
            #     )
            # else:
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
            print(f"[IC-LoRA] Added {len(video_conditioning)} video conditioning(s)")

        return conditionings
def _encode_video_for_retake(
    video_encoder: torch.nn.Module,
    video_path,# str or torch.Tensor
    output_shape: VideoPixelShape,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Load video and encode to latents."""
    # if isinstance(video_path, str):
    #     pixel_video = load_video_conditioning(
    #         video_path=video_path,
    #         height=output_shape.height,
    #         width=output_shape.width,
    #         frame_cap=output_shape.frames,
    #         dtype=dtype,
    #         device=device,
    #     )  # (1, C, F, H, W)
    # else:
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


def read_lat_emb(device):

    if not  os.path.exists(os.path.join(folder_paths.get_output_directory(),"raw_embeds_sm.pt")):
        raise Exception("No backup prompt embeddings found. Please run LTX2_SM_ENCODER node first.")
    else:
        prompt_embeds=torch.load(os.path.join(folder_paths.get_output_directory(),"raw_embeds_sm.pt"),weights_only=False).to(device,torch.bfloat16)
        if os.path.exists(os.path.join(folder_paths.get_output_directory(),"n_raw_embeds_sm.pt")):
            negative_prompt_embeds=torch.load(os.path.join(folder_paths.get_output_directory(),"n_raw_embeds_sm.pt"),weights_only=False).to(device,torch.bfloat16)
        else:
            negative_prompt_embeds=torch.zeros_like(prompt_embeds)
    positive=[[prompt_embeds,{"pooled_output": None}]]
    negative=[[negative_prompt_embeds,{"pooled_output": None}]]
    return positive,negative

    
def  save_lat_emb(data1,data2,mode=""):
    data1_prefix, data2_prefix = "raw_embeds", "n_raw_embeds"
    default_data1_path = os.path.join(folder_paths.get_output_directory(),f"{data1_prefix}_sm.pt")
    default_data2_path = os.path.join(folder_paths.get_output_directory(),f"{data2_prefix}_sm.pt")
    prefix = mode+str(int(time.time()))
    if os.path.exists(default_data1_path): # use a different path if the file already exists
        default_data1_path=os.path.join(folder_paths.get_output_directory(),f"{data1_prefix}_sm_{prefix}.pt")
    torch.save(data1,default_data1_path)
    if data2 is not None:
        if os.path.exists(default_data2_path):
            default_data2_path=os.path.join(folder_paths.get_output_directory(),f"{data2_prefix}_sm_{prefix}.pt")
        torch.save(data2,default_data2_path)