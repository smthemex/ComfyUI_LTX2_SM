 # !/usr/bin/env python
# -*- coding: UTF-8 -*-
import time
import torch
import os
import gc
import folder_paths
import comfy.model_management as mm
from .model_loader_utils import nomarl_upscale,tensor2pillist_upscale
from .LTX2.ltx_pipelines.ti2vid_two_stages import load_pipeline_ltx,inference_ltx_two_stages,TI2VidTwoStagesPipeline
from .LTX2.ltx_pipelines.keyframe_interpolation import load_pipeline_Keyframe_ltx,inference_ltx_Keyframe,KeyframeInterpolationPipeline
from .LTX2.ltx_pipelines.ic_lora import ICLoraPipeline,load_pipeline_ICLora_ltx,inference_ltx_ICLora,create_conditionings_
from .LTX2.ltx_core.loader.primitives import LoraPathStrengthAndSDOps
from .LTX2.ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP
from .LTX2.ltx_pipelines.ti2vid_one_stage import load_pipeline_ltx_one_stage,inference_ltx_one_stage
from .LTX2.ltx_pipelines.distilled import load_pipeline_ltx_distilled,inference_ltx_distilled,DistilledPipeline
from .LTX2.ltx_pipelines.utils import ModelLedger,encode_prompts
from .LTX2.ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from .LTX2.ltx_core.model.video_vae import decode_video as vae_decode_video
from .LTX2.ltx_pipelines.utils.helpers import (
    image_conditionings_by_replacing_latent,generate_enhanced_prompt, 
    cleanup_memory,
)
from .LTX2.ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from .LTX2.ltx_core.model.upsampler import upsample_video
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
    if dit_path is not None:
        checkpoint_path=dit_path
        fp8transformer=True
    elif gguf_path is not None:
        checkpoint_path=gguf_path
        fp8transformer=False
        gguf_dit=True
    if lora_path is not None:
        lora_ops=[LoraPathStrengthAndSDOps(lora_path,1.0,None)]
    else:
        lora_ops=[]
    if distilled_lora_path is not None:
        lora_ops_distilled=[LoraPathStrengthAndSDOps(distilled_lora_path,0.6,LTXV_LORA_COMFY_RENAMING_MAP)]
    else:
        lora_ops_distilled=[]
    if sampling_mode=="one_stage":
        model = load_pipeline_ltx_one_stage(checkpoint_path, None, lora_ops,device="cpu",fp8transformer=fp8transformer,gguf_dit=gguf_dit)
    elif sampling_mode=="two_stage":
        model = load_pipeline_ltx(checkpoint_path, lora_ops_distilled, None,None, loras=lora_ops,device="cpu" ,fp8transformer=fp8transformer,gguf_dit=gguf_dit)
    elif sampling_mode=="distilled":     
        model = load_pipeline_ltx_distilled(checkpoint_path, None, None,lora_ops,device="cpu" ,fp8transformer=fp8transformer,gguf_dit=gguf_dit)
    elif sampling_mode=="keyframe":
        model = load_pipeline_Keyframe_ltx(checkpoint_path, lora_ops_distilled, None,None,lora_ops,device="cpu" ,fp8transformer=fp8transformer,gguf_dit=gguf_dit)
    elif sampling_mode=="ic_lora":
        model = load_pipeline_ICLora_ltx(checkpoint_path, lora_ops_distilled, None,None, lora_ops,device="cpu" ,fp8transformer=fp8transformer,gguf_dit=gguf_dit)
    else:
        raise Exception("sampling_mode is not valid")   
    return model
    

def load_clip(clip,connector,node_cr_path):
    clip_path=folder_paths.get_full_path("gguf", clip) if clip != "none" else None
    connector_path=folder_paths.get_full_path("checkpoints", connector) if connector != "none" else None
    repo_path=os.path.join(node_cr_path,"LTX2/gemma")
    assert clip_path is not None,"Please provide a vae_path"

    model_ledger = ModelLedger(
        dtype=torch.bfloat16,
        device="cpu",
        checkpoint_path=connector_path,
        gemma_root_path=repo_path,
        loras=[],
        fp8transformer=False,
        gguf_dit=True,
        load_mode="clip",
        clip_path=clip_path,
    )
    text_encoder=model_ledger.text_encoder()
    return text_encoder

def encoder_text(text_encoder,prompt,negative_prompt,image,enhanced_prompt,save_emb,device):
   
    if enhanced_prompt:
        print("enhanced_prompt")
        prompt = generate_enhanced_prompt(text_encoder, prompt, image if image is not None else None)
    prompts = [p for p in [prompt, negative_prompt] if p.strip()] 

    with torch.no_grad():
        context = encode_prompts(text_encoder, prompts=prompts)

    torch.cuda.empty_cache()
    gc.collect()
    prompt_embeds=torch.concat(context[0], dim=-1)
    if len(context)>1:
        negative_prompt_embeds=torch.concat(context[1], dim=-1)
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
        fp8transformer=False,
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
        fp8transformer=False,
        gguf_dit=True,
        load_mode="audio",
        clip_path=None,
    )
    return model_ledger

def en_decoder_video(vae,latent,image,width,height,device,spatial_upsampler,cond_image=None):
    video,out_lat,out_lat_distll=None,None,None
    if latent is not  None:
        spatial_upsampler_path=folder_paths.get_full_path("latent_upscale_models", spatial_upsampler) if spatial_upsampler != "none" else None
        lat=latent["samples"]
        if  spatial_upsampler_path is None:
           
            generator=latent.get("samples",None) if latent.get("samples",None) is not None else torch.manual_seed(0)
            video_decoder=vae.video_decoder()
            video_decoder.to(device)
            try:
                video = vae_decode_video(lat, video_decoder, generator=generator) #Decoded chunk [f, h, w, c], uint8 in [0, 255].
            except:
                video = vae_decode_video(lat, video_decoder,tiling_config=TilingConfig.default(), generator=generator,) #Decoded chunk [f, h, w, c], uint8 in [0, 255].
            video_result = torch.stack(list(video)) if hasattr(video, '__iter__') else video

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
                fp8transformer=False,
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

    elif image is not None:
        video_encoder = vae.video_encoder()
        video_encoder.to(device)
        if image.shape[0]>1 and cond_image is not None:
            num_frames =  min (image.shape[0], cond_image.shape[0])

            image=tensor2pillist_upscale(image, width, height)
            images=[]
            for i,img in enumerate(image):
                images.append((img,i,0.8))
            cond_image=cond_image.unsqueeze(0).permute(0, 4, 1, 2, 3) # FHWC -> (1, C, F, height, width)
            video_conditioning=[(cond_image,0.8)]

            out_lat_distll= create_conditionings_(
                images,#: list[tuple[str, int, float]],
                video_conditioning,#: list[tuple[str, float]],
                height// 2,#: int,
                width // 2,#: int,
                num_frames,#: int,
                video_encoder,#: VideoEncoder,
                torch.bfloat16,
                device,
            )
            out_lat = image_conditionings_by_replacing_latent(
                images=image,
                height=height,
                width=width,
                video_encoder=video_encoder,
                dtype=torch.bfloat16,
                device=torch.device("cuda"),
            )
        else:
            image=nomarl_upscale(image, width, height)
            image=[(image,0,0.8)]
           
            #stage_1_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
           
            out_lat = image_conditionings_by_replacing_latent(
                images=image,
                height=height,
                width=width,
                video_encoder=video_encoder,
                dtype=torch.bfloat16,
                device=torch.device("cuda"),
            )
            out_lat_distll = image_conditionings_by_replacing_latent(
                images=image,
                height=height// 2,
                width=width // 2,
                video_encoder=video_encoder,
                dtype=torch.bfloat16,
                device=torch.device("cuda"),
            )

        video_encoder.to("cpu")
        torch.cuda.empty_cache()
        cleanup_memory()
    else:
        raise Exception("Please provide a latent or image")
    
    output={"samples_d":out_lat_distll,"samples":out_lat}
    return video,output


def decoder_audio(audio_vae,audio_latents,device):

    audio_decoder=audio_vae.audio_decoder().to(device)
    vocoder=audio_vae.vocoder().to(device)
    samples = vae_decode_audio(audio_latents["samples"], audio_decoder, vocoder)
    audio_decoder.to(torch.device("cpu"))
    vocoder.to(torch.device("cpu"))
    torch.cuda.empty_cache()
    gc.collect()

    return {"waveform": samples.contiguous().reshape(1, -1).cpu().float().unsqueeze(0), "sample_rate": 16000}



def inference_ltx2(model, positive,negative,seed,height, width, num_frames,frame_rate, steps,cfg, cond_lat,offload,device):
    
    context_p= torch.split(positive[0][0], 3840, dim=-1)
    if negative is not None:
        context_n = torch.split(negative[0][0], 3840, dim=-1)
    else:
        context_n = None
    if isinstance(model, TI2VidTwoStagesPipeline):
        video, audio=inference_ltx_two_stages(model, context_p, context_n, seed,height, width, num_frames,frame_rate, steps,cfg, cond_lat,offload)
    elif isinstance(model, DistilledPipeline):     
        video, audio=inference_ltx_distilled(model, context_p, seed,height, width, num_frames,frame_rate,cond_lat,offload)
    elif isinstance(model, KeyframeInterpolationPipeline):     
        video, audio=inference_ltx_Keyframe(model, context_p, context_n, seed,height, width, num_frames,frame_rate, steps,cfg, cond_lat,offload)
    elif isinstance(model, ICLoraPipeline):     
        video, audio=inference_ltx_ICLora(model, context_p, context_n, seed,height, width, num_frames,frame_rate, steps,cfg, cond_lat,offload)
    else:
        video, audio=inference_ltx_one_stage(model, context_p, context_n, seed,height, width, num_frames,frame_rate, steps,cfg, cond_lat,offload)
    #print(f"video shape: {video.shape}",audio.shape) #video shape: torch.Size([1, 128, 11, 16, 24]) torch.Size([1, 8, 84, 16])
    
    latents={"samples": video}
    audio_latents={"samples": audio, "num_frames": num_frames, "frame_rate": frame_rate,
                   "height":height, "width": width,"images":cond_lat,"seed":seed,"offload":offload}
    return latents, audio_latents


def inference_stage2(model,latents,audio_latents,device):

    context_p= torch.split(audio_latents["positives"][0][0], 3840, dim=-1)

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(audio_latents["num_frames"], tiling_config)
    if  isinstance(model, TI2VidTwoStagesPipeline) or isinstance(model, KeyframeInterpolationPipeline):
        context_n = torch.split(audio_latents["negatives"][0][0], 3840, dim=-1)
        video,audio=model(
            None,
            seed=audio_latents["seed"],
            height=audio_latents["height"],
            width=audio_latents["width"],
            num_frames=audio_latents["num_frames"],
            frame_rate=audio_latents["frame_rate"],
            images=audio_latents["images"],
            tiling_config=tiling_config,
            enhance_prompt=False,
            context_p=context_p,
            context_n=context_n,
            offload=audio_latents["offload"],
            stage_one=False,
            v_lat= latents["samples"],
            a_lat=audio_latents["samples"],)
    elif isinstance(model, DistilledPipeline)  or isinstance(model, ICLoraPipeline):
        video,audio=model(
            None,
            seed=audio_latents["seed"],
            height=audio_latents["height"],
            width=audio_latents["width"],
            num_frames=audio_latents["num_frames"],
            frame_rate=audio_latents["frame_rate"],
            images=audio_latents["images"],
            tiling_config=tiling_config,
            enhance_prompt=False,
            context_p=context_p,
            offload=audio_latents["offload"],
            stage_one=False,
            v_lat= latents["samples"],
            a_lat=audio_latents["samples"],)
    return video,audio


def read_lat_emb(prefix, positive, negative,latents,audio_latents,num_frames,frame_rate,offload,height,width,cond_lat,seed,device):
    if prefix =="embeds":
        if not  os.path.exists(os.path.join(folder_paths.get_output_directory(),"raw_embeds_sm.pt")):
            raise Exception("No backup prompt embeddings found. Please run LTX2_SM_ENCODER node first.")
        else:
            prompt_embeds=torch.load(os.path.join(folder_paths.get_output_directory(),"raw_embeds_sm.pt")).to(device,torch.bfloat16)
            if os.path.exists(os.path.join(folder_paths.get_output_directory(),"n_raw_embeds_sm.pt")):
                negative_prompt_embeds=torch.load(os.path.join(folder_paths.get_output_directory(),"n_raw_embeds_sm.pt")).to(device,torch.bfloat16)
            else:
                negative_prompt_embeds=torch.zeros_like(prompt_embeds)
        positive=[[prompt_embeds,{"pooled_output": None}]]
        negative=[[negative_prompt_embeds,{"pooled_output": None}]]
        return positive,negative
    
    elif prefix =="latents":
        if not  os.path.exists(os.path.join(folder_paths.get_output_directory(),"raw_latents_sm.pt")) or not os.path.exists(os.path.join(folder_paths.get_output_directory(),"raw_audio_latents_sm.pt")):
            raise Exception("No backup latents found. Please run LTX2_SM_KSampler node first.")
        else:
            video_latents=torch.load(os.path.join(folder_paths.get_output_directory(),"raw_latents_sm.pt")).to(device,torch.bfloat16)
            audio_latents=torch.load(os.path.join(folder_paths.get_output_directory(),"raw_audio_latents_sm.pt")).to(device,torch.bfloat16)
            print("Loaded backup latents",video_latents.shape,audio_latents.shape) #([1, 128, 11, 16, 24]) [1, 8, 84, 16] torch.Size([1, 84, 128])
            
        latents={"samples": video_latents}
        batch, frames, combined_dim = audio_latents.shape
        reshaped = audio_latents.view(batch, frames, 8, 16)
        audio_latents = reshaped.permute(0, 2, 1, 3)
        audio_latents={"samples": audio_latents,"num_frames": num_frames, "frame_rate": frame_rate,"offload":offload,
                        "height": height, "width": width,"images":cond_lat,"seed":seed,"negatives":negative,"positives":positive}
        return latents, audio_latents
    
def  save_lat_emb(save_prefix,data1,data2):
    data1_prefix, data2_prefix = ("raw_embeds", "n_raw_embeds") if save_prefix == "embeds" else ("raw_latents", "raw_audio_latents")
    default_data1_path = os.path.join(folder_paths.get_output_directory(),f"{data1_prefix}_sm.pt")
    default_data2_path = os.path.join(folder_paths.get_output_directory(),f"{data2_prefix}_sm.pt")
    prefix = str(int(time.time()))
    if os.path.exists(default_data1_path): # use a different path if the file already exists
        default_save_path=os.path.join(folder_paths.get_output_directory(),f"{data1_prefix}_{prefix}.pt")
    torch.save(data1,default_save_path)
    if data2 is not None:
        if os.path.exists(default_data2_path):
            default_data2_path=os.path.join(folder_paths.get_output_directory(),f"{data2_prefix}_{prefix}.pt")
        torch.save(data2,default_data2_path)