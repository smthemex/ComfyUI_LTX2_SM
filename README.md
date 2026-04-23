# LTX-2
 [LTX-2 ](https://github.com/Lightricks/LTX-2)is the first DiT-based audio-video foundation model that contains all core capabilities of modern video generation in one modelsynchronized audio and video, high fidelity, multiple performance modes, production-ready outputs, it's a test node.

# Update
*  简化节点，切换不同模式只需要菜单选择即刻，同步官方最新流式卸载代码，48G内存（峰值，如果开虚拟不需要），6G显存即可使用
*  only supports the ltx 2.3 model, ,need 6GVram + 48G Ram; 


# 1. Installation

In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_LTX2_SM.git
```

# 2. Requirements  
```
pip install -r requirements.txt
```

# 3. Model
* 3.1 gguf,te,connector checkpoints  from [smthem/LTX-2.3-test-gguf](https://huggingface.co/smthem/LTX-2.3-test-gguf/tree/main)  #  模型地址,包括text encoder transformer connector.safetensors，没梯子就去我[云盘](https://pan.quark.cn/s/d6af62159a11)拉取；
* 3.2 distill lora and upscaler from  [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3)  
* 3.3 vae and audio vae from [unsloth](https://huggingface.co/unsloth/LTX-2.3-GGUF/tree/main/vae) 
```
--  ComfyUI/models/diffusion_models/
    |--ltx-2.3-22b-distilled.safetensors # not transformer   optional   maybe kijai's dit
--  ComfyUI/models/gguf/
    |--ltx23-transformer-distill-Q8_0.gguf  # transformer only   optional 
    |--gemma-3-12b-it-qat-Q4_0.gguf # text encoder
    |--ltx23-transformer-distill-1.1-Q8_0.gguf  # transformer only   optional 
    |--ltx23-transformer-distill-1.1-Q6_K.gguf  # transformer only   optional 
--  ComfyUI/models/vae/
    |--ltx-2.3-22b-distilled_video_vae.safetensors  #  vae
    |--ltx-2.3-22b-distilled_audio_vae.safetensors # audio vae
--  ComfyUI/models/checkpoints/
    |--connector.safetensors   # text encoder
--  ComfyUI/models/lora/
    --any ltx2.3 lora

```

 
# 4. Example
* t2v
![](https://github.com/smthemex/ComfyUI_LTX2_SM/blob/main/example_workflows/example.png)

* i2v
![](https://github.com/smthemex/ComfyUI_LTX2_SM/blob/main/example_workflows/example_i.png)

* upsampler
![](https://github.com/smthemex/ComfyUI_LTX2_SM/blob/main/example_workflows/example_u.png)

* ic lora
![](https://github.com/smthemex/ComfyUI_LTX2_SM/blob/main/example_workflows/example_ic.png)



# 5. Citation
```
@article{hacohen2025ltx2,
  title={LTX-2: Efficient Joint Audio-Visual Foundation Model},
  author={HaCohen, Yoav and Brazowski, Benny and Chiprut, Nisan and Bitterman, Yaki and Kvochko, Andrew and Berkowitz, Avishai and Shalem, Daniel and Lifschitz, Daphna and Moshe, Dudu and Porat, Eitan and Richardson, Eitan and Guy Shiran and Itay Chachy and Jonathan Chetboun and Michael Finkelson and Michael Kupchick and Nir Zabari and Nitzan Guetta and Noa Kotler and Ofir Bibi and Ori Gordon and Poriya Panet and Roi Benita and Shahar Armon and Victor Kulikov and Yaron Inger and Yonatan Shiftan and Zeev Melumian and Zeev Farbman},
  journal={arXiv preprint arXiv:2601.03233},
  year={2025}
}

```


