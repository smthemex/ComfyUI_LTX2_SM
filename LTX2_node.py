 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import nodes
from .load_utils import (
    load_model,clear_comfyui_cache,load_vae,load_audio_vae,en_decoder_video,decoder_audio,infer_latent_shape,
    inference_ltx2,load_clip,encoder_text,inference_stage2,read_lat_emb,save_lat_emb,encoder_audio,get_latents
    )
MAX_SEED = np.iinfo(np.int32).max
node_cr_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

weigths_gguf_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_gguf_current_path):
    os.makedirs(weigths_gguf_current_path)

folder_paths.add_model_folder_path("gguf", weigths_gguf_current_path) #  gguf dir


class LTX2_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTX2_SM_Model",
            display_name="LTX2_SM_Model",
            category="LTX2_SM",
            inputs=[
                io.Combo.Input("dit",options= ["none"] + folder_paths.get_filename_list("diffusion_models") ),
                io.Combo.Input("gguf",options= ["none"] + folder_paths.get_filename_list("gguf")),
                io.Combo.Input("distilled_lora", options=["none"] + folder_paths.get_filename_list("loras")),
                io.Combo.Input("lora", options=["none"] + folder_paths.get_filename_list("loras")),
                io.Combo.Input("sampling_mode", options=["distilled","one_stage","two_stages","keyframe","ic_lora","audio2v","retake","twostages_hq"],),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls,dit,gguf,distilled_lora,lora,sampling_mode) -> io.NodeOutput:
        clear_comfyui_cache()
        model= load_model(dit, gguf, lora, distilled_lora,sampling_mode,)
        return io.NodeOutput(model)

class LTX2_SM_VAE(io.ComfyNode):
    @classmethod
    def define_schema(cls):       
        return io.Schema(
            node_id="LTX2_SM_VAE",
            display_name="LTX2_SM_VAE",
            category="LTX2_SM",
            inputs=[
                io.Combo.Input("vae",options= ["none"] + folder_paths.get_filename_list("vae") ),
            ],
            outputs=[io.Vae.Output(display_name="vae"),],
            )
    @classmethod
    def execute(cls,vae, ) -> io.NodeOutput:
        clear_comfyui_cache()
        vae=load_vae(vae)     
        return io.NodeOutput(vae)
    
class LTX2_SM_Clip(io.ComfyNode):
    @classmethod
    def define_schema(cls):       
        return io.Schema(
            node_id="LTX2_SM_Clip",
            display_name="LTX2_SM_Clip",
            category="LTX2_SM",
            inputs=[
                io.Combo.Input("clip",options= ["none"] + folder_paths.get_filename_list("gguf") ),
                io.Combo.Input("connector",options= ["none"] + folder_paths.get_filename_list("checkpoints") ),
            ],
            outputs=[io.Clip.Output(display_name="clip"),],
            )
    @classmethod
    def execute(cls,clip,connector ) -> io.NodeOutput:
        clear_comfyui_cache()
        clip=load_clip(clip,connector,node_cr_path)     
        return io.NodeOutput(clip)
    
class LTX2_SM_AUDIO_VAE(io.ComfyNode):
    @classmethod
    def define_schema(cls):      
        return io.Schema(
            node_id="LTX2_SM_AUDIO_VAE",
            display_name="LTX2_SM_AUDIO_VAE",
            category="LTX2_SM",
            inputs=[
                io.Combo.Input("audio_vae",options= ["none"] + folder_paths.get_filename_list("vae") ),
            ],
            outputs=[io.Vae.Output(display_name="audio_vae"),],
            )
    @classmethod
    def execute(cls,audio_vae, ) -> io.NodeOutput:
        clear_comfyui_cache()
        audio_vae=load_audio_vae(audio_vae)
        return io.NodeOutput(audio_vae)

class LTX2_EN_DECO_VIDEO(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTX2_EN_DECO_VIDEO",
            display_name="LTX2_EN_DECO_VIDEO",
            category="LTX2_SM",
            inputs=[
                io.Vae.Input("vae"),
                io.Latent.Input("latent"),
                io.Combo.Input("spatial_upsampler", options=["none"] + folder_paths.get_filename_list("latent_upscale_models")),  
                io.Boolean.Input("tile", default=True),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Image.Output(display_name="images"),
                ],
            )
    @classmethod
    def execute(cls,vae,latent,spatial_upsampler,tile) -> io.NodeOutput:
        clear_comfyui_cache()
        video,output=en_decoder_video(vae,latent,device,spatial_upsampler,tile)
        return io.NodeOutput(output,video)

class LTX2_DECO_AUDIO(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="LTX2_DECO_AUDIO",
            display_name="LTX2_DECO_AUDIO",
            category="LTX2_SM",
            inputs=[
                io.Vae.Input("audio_vae"),
                io.Latent.Input("audio_latents"),

            ],
            outputs=[
                io.Audio.Output(display_name="audio"),
                ],
            )
    @classmethod
    def execute(cls,audio_vae,audio_latents,) -> io.NodeOutput:
        clear_comfyui_cache()
        audio=decoder_audio(audio_vae,audio_latents,device)
        return io.NodeOutput(audio,None)
      
        
class LTX2_LATENTS(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTX2_LATENTS",
            display_name="LTX2_LATENTS",
            category="LTX2_SM",
            inputs=[
                io.Int.Input("width", default=768, min=256, max=nodes.MAX_RESOLUTION,step=32,display_mode=io.NumberDisplay.number),
                io.Int.Input("height", default=512, min=256, max=nodes.MAX_RESOLUTION,step=32,display_mode=io.NumberDisplay.number),
                io.Int.Input("num_frames", default=81, min=25, max=MAX_SEED,step=8,display_mode=io.NumberDisplay.number),
                io.Float.Input("frame_rate", default=24.0, min=8.0, max=120.0,step=1.0,display_mode=io.NumberDisplay.number),
                io.Float.Input("strength", default=1.0, min=0.1, max=1.0,step=0.01,display_mode=io.NumberDisplay.number),
                io.Float.Input("audio_start_time", default=0.0, min=0.0, max=10000.0,step=0.01,display_mode=io.NumberDisplay.number),
                io.Float.Input("audio_max_duration", default=0.0, min=0.0, max=10000.0,step=0.01,display_mode=io.NumberDisplay.number),
                io.Vae.Input("vae",optional=True),
                io.Vae.Input("audio_vae",optional=True),
                io.Image.Input("image",optional=True),
                io.Audio.Input("audio",optional=True),
                io.Image.Input("ic_lora_video",optional=True),
                io.Mask.Input("ic_lora_mask",optional=True),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                ],
            )
    @classmethod
    def execute(cls,width,height,num_frames,frame_rate,strength,audio_start_time,audio_max_duration,vae=None,audio_vae=None,image=None,audio=None,ic_lora_video=None,ic_lora_mask=None ) -> io.NodeOutput:
        clear_comfyui_cache() 
        width=(width //32)*32 if width % 32 != 0  else width 
        height=(height //32)*32 if height % 32 != 0  else height
        if ic_lora_video is not None:
            print("when use iclora  need to use 64x64 resolution")
            width=(width //64)*64 if width % 64 != 0  else width 
            height=(height //64)*64 if height % 64 != 0  else height
        stage_1_output_shape,stage_2_output_shape=infer_latent_shape(width,height,num_frames,frame_rate)
        output=get_latents(vae,image,audio_vae,audio,width,height,device,num_frames,frame_rate,stage_1_output_shape,stage_2_output_shape,strength,audio_start_time,audio_max_duration,ic_lora_video,ic_lora_mask)
        return io.NodeOutput(output)

    
class LTX2_SM_ENCODER(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="LTX2_SM_ENCODER",
            display_name="LTX2_SM_ENCODER",
            category="LTX2_SM",
            inputs=[
                io.Clip.Input("clip"),
                io.Boolean.Input("enhanced_prompt",default=False),
                io.Boolean.Input("save_emb",default=False),
                io.String.Input("prompt",multiline=True,default="A close-up of a cheerful girl puppet with curly auburn yarn hair and wide button eyes, holding a small red umbrella above her head. Rain falls gently around her. She looks upward and begins to sing with joy in English: It's raining, it's raining, I love it when its raining. Her fabric mouth opening and closing to a melodic tune. Her hands grip the umbrella handle as she sways slightly from side to side in rhythm. The camera holds steady as the rain sparkles against the soft lighting. Her eyes blink occasionally as she sings."),
                io.String.Input("negative_prompt",multiline=True,default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, unreadable text on shirt or hat, missing microphone, misplaced microphone, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, smiling, laughing, exaggerated sadness, wrong gaze direction, eyes looking at camera, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, missing sniff sounds, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, missing door or shelves, missing shallow depth of field, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."),
                io.Image.Input("image",optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                ],
            )
    @classmethod
    def execute(cls,clip,enhanced_prompt,save_emb,prompt,negative_prompt,image=None, ) -> io.NodeOutput:
        clear_comfyui_cache()
        positive,negative=encoder_text(clip,prompt,negative_prompt,image,enhanced_prompt,save_emb,device)
        return io.NodeOutput(positive,negative)

class LTX2_SM_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTX2_SM_KSampler",
            display_name="LTX2_SM_KSampler",
            category="LTX2_SM",
            inputs=[
                io.Model.Input("model"),     
                io.Latent.Input("latents",),    
                io.Int.Input("steps", default=8, min=1, max=nodes.MAX_RESOLUTION,step=1,display_mode=io.NumberDisplay.number),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED,display_mode=io.NumberDisplay.number),
                io.Float.Input("video_cfg_guidance_scale", default=1.0, min=0.0, max=10.0,step=0.01,display_mode=io.NumberDisplay.number),
                io.Float.Input("video_stg_guidance_scale", default=0.0, min=0.0, max=10.0,step=0.01,display_mode=io.NumberDisplay.number),
                io.Float.Input("video_rescale_scale", default=0.0, min=0.0, max=10.0,step=0.01,display_mode=io.NumberDisplay.number),
                io.Float.Input("a2v_guidance_scale", default=1.0, min=0.0, max=10.0,step=0.01,display_mode=io.NumberDisplay.number),
                io.Int.Input("video_skip_step", default=0, min=0, max=100,step=1,display_mode=io.NumberDisplay.number),
                io.Int.Input("video_stg_blocks", default=-1, min=-1, max=48,step=1,display_mode=io.NumberDisplay.number),
                io.Float.Input("audio_cfg_guidance_scale", default=1.0, min=0.0, max=10.0,step=0.01,display_mode=io.NumberDisplay.number),
                io.Float.Input("audio_stg_guidance_scale", default=0.0, min=0.0, max=10.0,step=0.01,display_mode=io.NumberDisplay.number),
                io.Float.Input("audio_rescale_scale", default=0.0, min=0.0, max=10.0,step=0.01,display_mode=io.NumberDisplay.number),
                io.Float.Input("v2a_guidance_scale", default=1.0, min=0.0, max=10.0,step=0.01,display_mode=io.NumberDisplay.number),
                io.Int.Input("audio_skip_step", default=0, min=0, max=100,step=1,display_mode=io.NumberDisplay.number),
                io.Int.Input("audio_stg_blocks", default=-1, min=-1, max=48,step=1,display_mode=io.NumberDisplay.number),
                io.Boolean.Input("offload", default=True),
                io.Boolean.Input("save_latents", default=True),
                io.Boolean.Input("pass_stage1", default=False),
                io.Conditioning.Input("positive",optional=True),
                io.Conditioning.Input("negative",optional=True),  
            ], 
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Latent.Output(display_name="audio_latents"),
            ],
        )
    @classmethod
    def execute(cls, model,latents,steps,seed,video_cfg_guidance_scale,video_stg_guidance_scale,
                video_rescale_scale,a2v_guidance_scale,video_skip_step,video_stg_blocks,audio_cfg_guidance_scale,audio_stg_guidance_scale,
                audio_rescale_scale,v2a_guidance_scale,audio_skip_step,audio_stg_blocks,
                offload,save_latents,pass_stage1,positive=None,negative=None,) -> io.NodeOutput:

        if positive is None:
            positive,negative=read_lat_emb("embeds", positive, negative,device)
        clear_comfyui_cache()
        if pass_stage1:
            video_latents,audio_latents=read_lat_emb("latents", positive, negative,device)
        else:

            cfg=dict(   
                video_cfg_guidance_scale=video_cfg_guidance_scale,
                video_stg_guidance_scale=video_stg_guidance_scale,
                video_rescale_scale=video_rescale_scale,
                a2v_guidance_scale=a2v_guidance_scale,
                video_skip_step=video_skip_step ,
                video_stg_blocks=video_stg_blocks if video_stg_blocks>=0 else [],
                audio_cfg_guidance_scale=audio_cfg_guidance_scale,
                audio_stg_guidance_scale=audio_stg_guidance_scale,
                audio_rescale_scale=audio_rescale_scale,
                v2a_guidance_scale=v2a_guidance_scale,
                audio_skip_step=audio_skip_step,
                audio_stg_blocks=audio_stg_blocks if audio_stg_blocks>=0 else [],
            )
            video_latents, audio_latents=inference_ltx2(model, positive,negative,latents,seed, steps,cfg,offload)
            audio_latents["positives"]=positive
            audio_latents["negatives"]=negative
            audio_latents["args"]=cfg
            if save_latents:
                save_lat_emb("latents",video_latents,audio_latents,model.infer_mode)

        return io.NodeOutput(video_latents, audio_latents)

class LTX2_SM_StageTwo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTX2_SM_StageTwo",
            display_name="LTX2_SM_StageTwo",
            category="LTX2_SM",
            inputs=[
                io.Model.Input("model"),      
                io.Latent.Input("latents"), 
                io.Latent.Input("audio_latents"),  
            ], 
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Latent.Output(display_name="audio_latents"),
            ],
        )
    @classmethod
    def execute(cls, model,latents,audio_latents,) -> io.NodeOutput:
        clear_comfyui_cache()
        video, audio=inference_stage2(model,latents,audio_latents)
        latents["samples"]=video
        audio_latents["samples"]= audio
        return io.NodeOutput(latents, audio_latents)

class  LTX2_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LTX2_SM_Model,
            LTX2_SM_VAE,
            LTX2_SM_Clip,
            LTX2_SM_AUDIO_VAE,
            LTX2_EN_DECO_VIDEO,
            LTX2_DECO_AUDIO,
            LTX2_LATENTS,
            LTX2_SM_ENCODER,
            LTX2_SM_KSampler,
            LTX2_SM_StageTwo,
        ]
async def comfy_entrypoint() -> LTX2_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return LTX2_SM_Extension()
