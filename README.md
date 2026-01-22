# LTX-2
 [LTX-2 ](https://github.com/Lightricks/LTX-2)is the first DiT-based audio-video foundation model that contains all core capabilities of modern video generation in one modelsynchronized audio and video, high fidelity, multiple performance modes, production-ready outputs, it's a test node.

# Tips
* If you wan to  runing it in very low  Vram... ,use Q2 clip and Q4 dit(or slplit you  workflows), if upscaler 512x768 to 1024x1536 need 8G Vram (always OOM at vae decoder, use tiles is OK )
* Enable save clip emb and latents to save Ram and Vram too
* Will coming soon



 #
 
# Example
* 2x upscaler
![](https://github.com/smthemex/ComfyUI_LTX2_SM/blob/main/example_workflows/example.png)

* low Vram infer
![](https://github.com/smthemex/ComfyUI_LTX2_SM/blob/main/example_workflows/example_t.png)


# Citation
```
@article{hacohen2025ltx2,
  title={LTX-2: Efficient Joint Audio-Visual Foundation Model},
  author={HaCohen, Yoav and Brazowski, Benny and Chiprut, Nisan and Bitterman, Yaki and Kvochko, Andrew and Berkowitz, Avishai and Shalem, Daniel and Lifschitz, Daphna and Moshe, Dudu and Porat, Eitan and Richardson, Eitan and Guy Shiran and Itay Chachy and Jonathan Chetboun and Michael Finkelson and Michael Kupchick and Nir Zabari and Nitzan Guetta and Noa Kotler and Ofir Bibi and Ori Gordon and Poriya Panet and Roi Benita and Shahar Armon and Victor Kulikov and Yaron Inger and Yonatan Shiftan and Zeev Melumian and Zeev Farbman},
  journal={arXiv preprint arXiv:2601.03233},
  year={2025}
}

```


