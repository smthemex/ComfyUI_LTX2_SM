 # !/usr/bin/env python
# -*- coding: UTF-8 -*-
import time
import torch
import os
import gc
import folder_paths
import comfy.model_management as mm
from .model_loader_utils import nomarl_upscale,tensor2pillist_upscale,tensor_upscale
#from .LTX2.ltx_pipelines.ti2vid_two_stages import load_pipeline_ltx,inference_ltx_two_stages,TI2VidTwoStagesPipeline
#from .LTX2.ltx_pipelines.keyframe_interpolation import load_pipeline_Keyframe_ltx,inference_ltx_Keyframe,KeyframeInterpolationPipeline
#from .LTX2.ltx_pipelines.ic_lora import ICLoraPipeline,load_pipeline_ICLora_ltx,inference_ltx_ICLora,create_conditionings_
from .LTX2.ltx_pipelines.merge_pipeline import load_pipeline_ltx,_encode_video_for_retake,_encode_audio_for_retake,TemporalRegionMask,ic_create_conditionings,load_mask_video
from .LTX2.ltx_core.loader.primitives import LoraPathStrengthAndSDOps
from .LTX2.ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP
from .LTX2.ltx_core.quantization import QuantizationPolicy
#from .LTX2.ltx_pipelines.ti2vid_one_stage import load_pipeline_ltx_one_stage,inference_ltx_one_stage
#from .LTX2.ltx_pipelines.distilled import load_pipeline_ltx_distilled,inference_ltx_distilled,DistilledPipeline
from .LTX2.ltx_pipelines.utils import ModelLedger,encode_prompts
from .LTX2.ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from .LTX2.ltx_core.model.video_vae import decode_video as vae_decode_video
from .LTX2.ltx_pipelines.utils.helpers import (
     combined_image_conditionings,image_conditionings_by_adding_guiding_latent,
    cleanup_memory,
)
from .LTX2.ltx_pipelines.utils.args import ImageConditioningInput
from .LTX2.ltx_core.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
)
from .LTX2.ltx_core.conditioning import ConditioningItem
from .LTX2.ltx_core.types import SpatioTemporalScaleFactors, VideoPixelShape
from .LTX2.ltx_pipelines.utils.media_io import apply_start_time_max_duration, encode_video
from .LTX2.ltx_core.types import Audio, AudioLatentShape
from .LTX2.ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from .LTX2.ltx_core.model.upsampler import upsample_video
from .LTX2.ltx_core.model.audio_vae import encode_audio as vae_encode_audio
node_cr_path_ = os.path.dirname(os.path.abspath(__file__))

def clear_comfyui_cache():
    cf_models=mm.loaded_models()
    try:
        for pipe in cf_models:
            pipe.unpatch_model(device_to=torch.device("cpu"))
            print(f"Unpatching models.{pipe}")
    except: pass
    mm.soft_empty_cache()
    torch.cuda.empty_cache()
    max_gpu_memory = torch.cuda.max_memory_allocated()
    print(f"After Max GPU memory allocated: {max_gpu_memory / 1000 ** 3:.2f} GB")


def load_model( dit, gguf, lora, distilled_lora, sampling_mode):
    
    dit_path=folder_paths.get_full_path("diffusion_models", dit) if dit != "none" else None
    gguf_path=folder_paths.get_full_path("gguf", gguf) if gguf != "none" else None 
    lora_path=folder_paths.get_full_path("loras", lora) if lora != "none" else None
    distilled_lora_path=folder_paths.get_full_path("loras", distilled_lora) if distilled_lora != "none" else None
    gguf_dit=False
    quantization=None
    if dit_path is not None:
        checkpoint_path=dit_path
        if "fp8" in checkpoint_path.lower():
            quantization=QuantizationPolicy.fp8_cast()
    elif gguf_path is not None:
        checkpoint_path=gguf_path
        gguf_dit=True
    if lora_path is not None:
        lora_ops=[LoraPathStrengthAndSDOps(lora_path,1.0,None)]
    else:
        lora_ops=[]
    if distilled_lora_path is not None:
        lora_ops_distilled=[LoraPathStrengthAndSDOps(distilled_lora_path,0.6,LTXV_LORA_COMFY_RENAMING_MAP)]
    else:
        lora_ops_distilled=[]
    
    model=load_pipeline_ltx(checkpoint_path, lora_ops_distilled, None,None, loras=lora_ops,device="cpu" ,quantization=quantization,gguf_dit=gguf_dit,infer_mode=sampling_mode)

    return model
    

def load_clip(clip,connector,node_cr_path):
    clip_path=folder_paths.get_full_path("gguf", clip) if clip != "none" else None
    connector_path=folder_paths.get_full_path("checkpoints", connector) if connector != "none" else None
    repo_path=os.path.join(node_cr_path,"LTX2/gemma")
    assert clip_path is not None,"Please provide a clip_path"

    model_ledger = ModelLedger(
        dtype=torch.bfloat16,
        device="cpu",
        checkpoint_path=connector_path,
        gemma_root_path=repo_path,
        loras=[],
        quantization=None,
        gguf_dit=True,
        load_mode="clip",
        clip_path=clip_path,
    )
    #text_encoder=model_ledger.text_encoder()
    return model_ledger

def encoder_text(text_encoder,prompt,negative_prompt,image,enhanced_prompt,save_emb,device):
   
    # if enhanced_prompt:
    #     print("enhanced_prompt")
    #     prompt = generate_enhanced_prompt(text_encoder, prompt, image if image is not None else None)
    prompts = [p for p in [prompt, negative_prompt] if p.strip()] 

    with torch.no_grad():
        context = encode_prompts(prompts,text_encoder,enhance_prompt_image=image if image is not None else None,enhance_prompt_seed=42,enhance_first_prompt=enhanced_prompt,)

    torch.cuda.empty_cache()
    gc.collect()
    prompt_embeds=torch.concat([context[0].video_encoding,context[0].audio_encoding], dim=-1)
    print(prompt_embeds.shape)
    if len(context)>1:
        negative_prompt_embeds=torch.concat([context[1].video_encoding,context[1].audio_encoding], dim=-1)
        print(negative_prompt_embeds.shape)
    else:
        negative_prompt_embeds=None #distilled does not have negative prompts
    if save_emb:
        save_lat_emb("embeds",prompt_embeds,negative_prompt_embeds)
    positive=[[prompt_embeds,{"pooled_output": None}]]
    negative=[[negative_prompt_embeds,{"pooled_output": None}]]

    return positive, negative


def load_vae(vae):
    vae_path=folder_paths.get_full_path("vae", vae) if vae != "none" else None
    assert vae_path is not None,"Please provide a vae_path"
    model_ledger = ModelLedger(
        dtype=torch.bfloat16,
        device="cpu",
        checkpoint_path=vae_path,
        gemma_root_path=None,
        loras=[],
        quantization=None,
        gguf_dit=True,
        load_mode="vae",
        clip_path=None,
    )
      
    return model_ledger

def load_audio_vae(audio_vae,):
    #vocoder_path=folder_paths.get_full_path("vae", vocoder) if vocoder != "none" else None    
    vae_path=folder_paths.get_full_path("vae", audio_vae) if audio_vae != "none" else None
    assert vae_path is not None,"Please provide a vae_path"
    model_ledger = ModelLedger(
        dtype=torch.bfloat16,
        device="cpu",
        checkpoint_path=vae_path,
        gemma_root_path=None,
        loras=[],
        quantization=None,
        gguf_dit=True,
        load_mode="audio",
        clip_path=None,
    )
    return model_ledger

def infer_latent_shape(width,height,num_frames,frame_rate):
    stage_1_output_shape = VideoPixelShape(batch=1,frames=num_frames,width=width,height=height,fps=frame_rate,)
    stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width*2, height=height*2, fps=frame_rate)
    return stage_1_output_shape,stage_2_output_shape


def en_decoder_video(vae,latent,device,spatial_upsampler,tile=False):
    video,out_lat=None,None
    spatial_upsampler_path=folder_paths.get_full_path("latent_upscale_models", spatial_upsampler) if spatial_upsampler != "none" else None
    lat=latent["samples"]
    if  spatial_upsampler_path is None:
        generator=latent.get("samples",None) if latent.get("samples",None) is not None else torch.manual_seed(0)
        video_decoder=vae.video_decoder()
        video_decoder.to(device)
        if tile:
            video = vae_decode_video(lat, video_decoder,tiling_config=TilingConfig.default(), generator=generator,) #Decoded chunk [f, h, w, c], uint8 in [0, 255]. 
        else:
            video = vae_decode_video(lat, video_decoder, generator=generator) #Decoded chunk [f, h, w, c], uint8 in [0, 255].       
        video_frames = list(video)
        if len(video_frames) == 1:
            video_result = video_frames[0].unsqueeze(0)  # 添加批次维度 [1, f, h, w, c]
        else:
            video_result = torch.cat(video_frames, dim=0)  # [total_f, h, w, c]
            video_result = video_result.unsqueeze(0) 

        video_decoder.to(torch.device("cpu"))
        torch.cuda.empty_cache()
        gc.collect()
        #print(video_result.shape) ##torch.Size([1, 81, 512, 768, 3])
        video=video_result.float().cpu()[0]/255.0
    else:
        video_encoder = vae.video_encoder()
        model_ledger = ModelLedger(
            dtype=torch.bfloat16,
            device="cpu",
            checkpoint_path=None,
            gemma_root_path="",
            spatial_upsampler_path=spatial_upsampler_path,
            loras=[],
            quantization=None,
            gguf_dit=True,
            load_mode="spatial",
            clip_path=None,
        )
        upsampler=model_ledger.spatial_upsampler()  
        video_encoder.to(device)
        upsampler.to(device)
        out_lat = upsample_video(latent=lat[:1], video_encoder=video_encoder, upsampler=upsampler)
        #print(out_lat.shape) #torch.Size([1, 128, 11, 32, 48])
        video_encoder.to("cpu")
        upsampler.to("cpu")
        torch.cuda.empty_cache()
        cleanup_memory()

    output={"samples":out_lat,}
    return video,output


def get_latents(vae,image,audio_vae,audio,width,height,device,num_frames,frame_rate,stage_1_output_shape,stage_2_output_shape,strength=0.8,audio_start_time=0,audio_max_duration=0.0,ic_lora_video=None,ic_lora_mask=None):
    images,stage_1_conditionings,stage_2_conditionings=[],[],[]
    regenerate_video = True
    if image is not None and vae is not None:
        video_encoder = vae.video_encoder()
        video_encoder.to(device)
        image_list=tensor2pillist_upscale(image, width, height)
        if len(image_list)>5: #retake
            video_scale = SpatioTemporalScaleFactors.default()
            image_list=tensor_upscale(image, width, height).to(device,torch.bfloat16)
            if (image_list.shape[2] - 1) % video_scale.time != 0:
                snapped = ((num_frames - 1) // video_scale.time) * video_scale.time + 1
                image_list = image_list[:snapped, :, :,:]
            image_list=image_list[:num_frames, :, :,:]if num_frames<image_list.shape[2] else image_list
            
            stage_1_conditionings = _encode_video_for_retake(
                video_encoder=video_encoder,
                video_path=image_list,
                output_shape=stage_1_output_shape,
                dtype=torch.bfloat16,
                device=device,
                )
            
            stage_2_conditionings: list[ConditioningItem] = [
                TemporalRegionMask(
                    start_time=float(audio_start_time) if regenerate_video else 0.0,
                    end_time=audio_max_duration if regenerate_video else 0.0,
                    fps=frame_rate,
                )
            ]
        else:
            for i,img in enumerate(image_list):
                img_in=ImageConditioningInput(path=img, strength=strength, frame_idx=i, )
                images.append(img_in)
            if len(image_list)>1: # keyframe
                cond_fn=image_conditionings_by_adding_guiding_latent
            else:
                cond_fn=combined_image_conditionings
            torch.cuda.empty_cache()
            cleanup_memory()
            if ic_lora_video is not None: #ic_lora
                conditioning_attention_strength=1.0
                video_conditioning=[(tensor_upscale(image, width, height).to(device,torch.bfloat16),1.0)]
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
                stage_1_conditionings = ic_create_conditionings(
                    images=images,
                    video_conditioning=video_conditioning,
                    height=stage_1_output_shape.height,
                    width=stage_1_output_shape.width,
                    video_encoder=video_encoder,
                    num_frames=num_frames,
                    conditioning_attention_strength=conditioning_attention_strength,
                    conditioning_attention_mask=conditioning_attention_mask,
                    dtype=torch.bfloat16,
                    device=device,
                )
            else:
                stage_1_conditionings = cond_fn(
                    images=images,
                    height=stage_1_output_shape.height,
                    width=stage_1_output_shape.width,
                    video_encoder=video_encoder,
                    dtype=torch.bfloat16,
                    device=device,
                )

            stage_2_conditionings = cond_fn(
                images=images,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                video_encoder=video_encoder,
                dtype=torch.bfloat16,
                device=device,
                )
        video_encoder.to("cpu")
    output={"stage_1_conditionings":stage_1_conditionings,"stage_2_conditionings":stage_2_conditionings,"stage_1_output_shape":stage_1_output_shape,"stage_2_output_shape":stage_2_output_shape,
            "width":width,"height":height,"num_frames":num_frames,"frame_rate":frame_rate}
    
    decoded_audio,encoded_audio_latent=[],None

    if audio is not None and audio_vae is not None:
        if len(image_list)>5:
            audio_encoder=audio_vae.audio_encoder().to(device)
            encoded_audio_latent = _encode_audio_for_retake(
                audio_encoder=audio_encoder,
                waveform=audio["waveform"].squeeze(0),
                waveform_sr=audio["sample_rate"],
                output_shape=stage_2_output_shape,
                dtype=torch.bfloat16,
            )
            decoded_audio = [
                TemporalRegionMask(
                    start_time=float(audio_start_time) if regenerate_video else 0.0,
                    end_time=audio_max_duration if regenerate_video else 0.0,
                    fps=frame_rate,
                )
                ]
            audio_encoder.to(torch.device("cpu"))
        else:
            audio_max_duration =None if not audio_max_duration > 0 else audio_max_duration
            encoded_audio_latent,decoded_audio=encoder_audio(audio_vae,audio,num_frames,frame_rate, device,audio_start_time,audio_max_duration)
    audio_output={"decoded_audio":decoded_audio,"encoded_audio_latent":encoded_audio_latent,}
    output.update(audio_output)
    return output


def decoder_audio(audio_vae,audio_latents,device):
    audio_decoder=audio_vae.audio_decoder().to(device)
    vocoder=audio_vae.vocoder().to(device)
    samples = vae_decode_audio(audio_latents["samples"], audio_decoder, vocoder)
    audio_decoder.to(torch.device("cpu"))
    vocoder.to(torch.device("cpu"))
    torch.cuda.empty_cache()
    gc.collect()

    return {"waveform": samples.waveform.contiguous().reshape(1, -1).cpu().float().unsqueeze(0), "sample_rate": samples.sampling_rate}

def encoder_audio(audio_vae,audio,num_frames,frame_rate,device,audio_start_time, audio_max_duration):
    audio_encoder=audio_vae.audio_encoder().to(device)

    waveform = apply_start_time_max_duration(audio["waveform"].squeeze(0), audio["sample_rate"], audio_start_time, audio_max_duration)
    print(waveform.shape) 
    decoded_audio=Audio(waveform=waveform.to(device).unsqueeze(0), sampling_rate=audio["sample_rate"])
    encoded_audio_latent = vae_encode_audio(decoded_audio, audio_encoder)
    audio_encoder.to(torch.device("cpu"))
    torch.cuda.empty_cache()
    gc.collect()
    audio_shape = AudioLatentShape.from_duration(batch=1, duration=num_frames / frame_rate, channels=8, mel_bins=16)
    encoded_audio_latent = encoded_audio_latent[:, :, : audio_shape.frames]
    #print(encoded_audio_latent.shape) #torch.Size([1, 8, 126, 16])
    #print(waveform.shape) #torch.Size([2, 1409024])
    audio["waveform"]=waveform.unsqueeze(0)
    
    return encoded_audio_latent,audio


def inference_ltx2(model, positive,negative,latents,seed, steps,cfg,offload,block_group_size):

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

    video, audio = model(
        seed=seed,
        num_inference_steps=steps,
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params,
        images=latents,
        #tiling_config=tiling_config,
        context_p=context_p,
        context_n=context_n,
        offload=offload,
        block_group_size=block_group_size,
    )
    
    video_latents={"samples": video}
    audio_latents={"samples": audio,"steps":steps, "cfg":cfg,"seed":seed,"offload":offload}
    audio_latents.update(latents)
    return  video_latents, audio_latents


def inference_stage2(model,latents,audio_latents,block_group_size=2):
    video_features_size = 4096 if audio_latents["positives"][0][0].shape[2] == 6144 else 3840
    context_p=[audio_latents["positives"][0][0][..., :video_features_size],audio_latents["positives"][0][0][..., video_features_size:] ]

    #context_p= torch.split(audio_latents["positives"][0][0], 3840, dim=-1)

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(audio_latents["num_frames"], tiling_config)
    args=audio_latents["args"]
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
    context_n=[audio_latents["negatives"][0][0][..., :video_features_size],audio_latents["negatives"][0][0][..., video_features_size:] ]
    video,audio=model(
        seed=audio_latents["seed"],
        num_inference_steps=audio_latents["steps"],
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params,
        images=audio_latents,
        context_p=context_p,
        context_n=context_n,
        offload=audio_latents["offload"],
        stage_one=False,
        v_lat= latents["samples"],
        a_lat=audio_latents["samples"],   
        video_conditioning=[],
        conditioning_attention_strength = 1.0,
        conditioning_attention_mask= None,
        block_group_size=block_group_size,
        )

    return video,audio


def read_lat_emb(prefix, positive, negative,device):
    if prefix =="embeds":
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
    
    elif prefix =="latents":
        if not  os.path.exists(os.path.join(folder_paths.get_output_directory(),"raw_latents_sm.pt")) or not os.path.exists(os.path.join(folder_paths.get_output_directory(),"raw_audio_latents_sm.pt")):
            raise Exception("No backup latents found. Please run LTX2_SM_KSampler node first.")
        else:
            video_latents=torch.load(os.path.join(folder_paths.get_output_directory(),"raw_latents_sm.pt"),weights_only=False)
            audio_latents=torch.load(os.path.join(folder_paths.get_output_directory(),"raw_audio_latents_sm.pt"),weights_only=False)
            #print("Loaded backup latents",video_latents.shape,audio_latents.shape) #([1, 128, 11, 16, 24]) [1, 8, 84, 16] torch.Size([1, 84, 128])

        video_latents["samples"]=video_latents["samples"].to(device,torch.bfloat16)
        audio_lat=audio_latents["samples"].to(device,torch.bfloat16)  # [1, 8, 84, 16]
        print(f"audio shape: {audio_lat.shape}")  #audio shape: torch.Size([1, 84, 8, 16])
        print(f"video shape: {video_latents['samples'].shape}")
        # batch, frames, combined_dim = audio_lat.shape
        # reshaped = audio_lat.view(batch, frames, 8, 16)
        #audio_lat = audio_lat.permute(0, 2, 1, 3)
       
        audio_latents["samples"]=audio_lat
        return video_latents, audio_latents
    
def  save_lat_emb(save_prefix,data1,data2,mode=""):
    data1_prefix, data2_prefix = ("raw_embeds", "n_raw_embeds") if save_prefix == "embeds" else ("raw_latents", "raw_audio_latents")
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