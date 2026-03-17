import json

import safetensors
import torch

from .primitives import StateDict, StateDictLoader
from .sd_ops import SDOps

# 完整的原始配置
ORIGINAL_CONFIG_JSON = """
{"transformer": {"_class_name": "AVTransformer3DModel", "activation_fn": "gelu-approximate", "attention_bias": true, "attention_head_dim": 128, "attention_type": "default", "caption_channels": 3840, "cross_attention_dim": 4096, "double_self_attention": false, "dropout": 0.0, "in_channels": 128, "norm_elementwise_affine": false, "norm_eps": 1e-06, "norm_num_groups": 32, "num_attention_heads": 32, "num_embeds_ada_norm": 1000, "num_layers": 48, "num_vector_embeds": null, "only_cross_attention": false, "cross_attention_norm": true, "out_channels": 128, "upcast_attention": false, "use_linear_projection": false, "qk_norm": "rms_norm", "standardization_norm": "rms_norm", "positional_embedding_type": "rope", "positional_embedding_theta": 10000.0, "positional_embedding_max_pos": [20, 2048, 2048], "timestep_scale_multiplier": 1000, "av_ca_timestep_scale_multiplier": 1000.0, "causal_temporal_positioning": true, "audio_num_attention_heads": 32, "audio_attention_head_dim": 64, "use_audio_video_cross_attention": true, "share_ff": false, "audio_out_channels": 128, "audio_cross_attention_dim": 2048, "audio_positional_embedding_max_pos": [20], "av_cross_ada_norm": true, "use_embeddings_connector": true, "connector_attention_head_dim": 128, "connector_num_attention_heads": 32, "connector_num_layers": 8, "connector_positional_embedding_max_pos": [4096], "connector_num_learnable_registers": 128, "connector_norm_output": true, "use_middle_indices_grid": true, "apply_gated_attention": true, "connector_apply_gated_attention": true, "caption_projection_first_linear": false, "caption_projection_second_linear": false, "caption_proj_input_norm": false, "connector_learnable_registers_std": 1, "caption_proj_before_connector": true, "audio_connector_attention_head_dim": 64, "audio_connector_num_attention_heads": 32, "cross_attention_adaln": true, "text_encoder_norm_type": "per_token_rms", "rope_type": "split", "frequencies_precision": "float64"}, "vae": {"_class_name": "CausalVideoAutoencoder", "dims": 3, "in_channels": 3, "out_channels": 3, "latent_channels": 128, "encoder_blocks": [["res_x", {"num_layers": 4}], ["compress_space_res", {"multiplier": 2}], ["res_x", {"num_layers": 6}], ["compress_time_res", {"multiplier": 2}], ["res_x", {"num_layers": 4}], ["compress_all_res", {"multiplier": 2}], ["res_x", {"num_layers": 2}], ["compress_all_res", {"multiplier": 1}], ["res_x", {"num_layers": 2}]], "decoder_blocks": [["res_x", {"num_layers": 4}], ["compress_space", {"multiplier": 2}], ["res_x", {"num_layers": 6}], ["compress_time", {"multiplier": 2}], ["res_x", {"num_layers": 4}], ["compress_all", {"multiplier": 1}], ["res_x", {"num_layers": 2}], ["compress_all", {"multiplier": 2}], ["res_x", {"num_layers": 2}]], "scaling_factor": 1.0, "norm_layer": "pixel_norm", "patch_size": 4, "latent_log_var": "uniform", "use_quant_conv": false, "causal_decoder": false, "timestep_conditioning": false, "normalize_latent_channels": false, "encoder_base_channels": 128, "decoder_base_channels": 128, "spatial_padding_mode": "zeros"}, "scheduler": {"_class_name": "RectifiedFlowScheduler", "_diffusers_version": "0.25.1", "num_train_timesteps": 1000, "shifting": null, "base_resolution": null, "sampler": "LinearQuadratic"}, "audio_vae": {"model": {"params": {"ddconfig": {"double_z": true, "mel_bins": 64, "z_channels": 8, "resolution": 256, "downsample_time": false, "in_channels": 2, "out_ch": 2, "ch": 128, "ch_mult": [1, 2, 4], "num_res_blocks": 2, "attn_resolutions": [], "dropout": 0.0, "mid_block_add_attention": false, "norm_type": "pixel", "causality_axis": "height"}, "sampling_rate": 16000}}, "preprocessing": {"audio": {"sampling_rate": 16000, "max_wav_value": 32768.0, "duration": 5.12, "stereo": true, "causal_padding": 3}, "stft": {"filter_length": 1024, "hop_length": 160, "win_length": 1024, "causal": true}, "mel": {"n_mel_channels": 64, "mel_fmin": 0, "mel_fmax": 8000}}}, "vocoder": {"vocoder": {"upsample_initial_channel": 1536, "resblock": "AMP1", "upsample_rates": [5, 2, 2, 2, 2, 2], "resblock_kernel_sizes": [3, 7, 11], "upsample_kernel_sizes": [11, 4, 4, 4, 4, 4], "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]], "stereo": true, "use_tanh_at_final": false, "activation": "snakebeta", "use_bias_at_final": false}, "bwe": {"upsample_initial_channel": 512, "resblock": "AMP1", "upsample_rates": [6, 5, 2, 2, 2], "resblock_kernel_sizes": [3, 7, 11], "upsample_kernel_sizes": [12, 11, 4, 4, 4], "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]], "stereo": true, "use_tanh_at_final": false, "activation": "snakebeta", "use_bias_at_final": false, "apply_final_activation": false, "input_sampling_rate": 16000, "output_sampling_rate": 48000, "hop_length": 80, "n_fft": 512, "win_size": 512, "num_mels": 64}}}
"""

# keys with these prefixes will be filtered out from metadata
#_METADATA_FILTER_PREFIXES = ("audio_vae.decoder", "vae.decoder", "vocoder")


# def _filter_metadata_dict(d: dict) -> dict:
#     """Return a new dict excluding keys that start with any filtered prefix."""
#     out = {}
#     for k, v in d.items():
#         key = k.decode() if isinstance(k, (bytes, bytearray)) else k
#         if any(key.startswith(p) for p in _METADATA_FILTER_PREFIXES):
#             continue
#         out[key] = v
#     return out


class SafetensorsStateDictLoader(StateDictLoader):
    """
    Loads weights from safetensors files without metadata support.
    Use this for loading raw weight files. For model files that include
    configuration metadata, use SafetensorsModelStateDictLoader instead.
    """

    def metadata(self, path: str) -> dict:
        raise NotImplementedError("Not implemented")

    def load(self, path: str | list[str], sd_ops: SDOps, device: torch.device | None = None) -> StateDict:
        """
        Load state dict from path or paths (for sharded model storage) and apply sd_ops
        """
        sd = {}
        size = 0
        dtype = set()
        device = device or torch.device("cpu")
        model_paths = path if isinstance(path, list) else [path]
        for shard_path in model_paths:
            with safetensors.safe_open(shard_path, framework="pt", device=str(device)) as f:
                safetensor_keys = f.keys()
                for name in safetensor_keys:
                    expected_name = name if sd_ops is None else sd_ops.apply_to_key(name)
                    if expected_name is None:
                        continue
                    value = f.get_tensor(name).to(device=device, non_blocking=True, copy=False)
                    key_value_pairs = ((expected_name, value),)
                    if sd_ops is not None:
                        key_value_pairs = sd_ops.apply_to_key_value(expected_name, value)
                    for key, value in key_value_pairs:
                        size += value.nbytes
                        dtype.add(value.dtype)
                        sd[key] = value

        return StateDict(sd=sd, device=device, size=size, dtype=dtype)


class SafetensorsModelStateDictLoader(StateDictLoader):
    """
    Loads weights and configuration metadata from safetensors model files.
    Unlike SafetensorsStateDictLoader, this loader can read model configuration
    from the safetensors file metadata via the metadata() method.
    """

    def __init__(self, weight_loader: SafetensorsStateDictLoader | None = None):
        self.weight_loader = weight_loader if weight_loader is not None else SafetensorsStateDictLoader()

    def metadata(self, path: str,laod_model) -> dict:
        if laod_model in ["spatial"]:
            with safetensors.safe_open(path, framework="pt") as f:
                md = json.loads(f.metadata()["config"])       
            return md
        else:
            md = json.loads(ORIGINAL_CONFIG_JSON)
        return md



    def load(self, path: str | list[str], sd_ops: SDOps | None = None, device: torch.device | None = None) -> StateDict:
        return self.weight_loader.load(path, sd_ops, device)


