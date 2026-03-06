import logging
from dataclasses import dataclass, field, replace
from typing import Generic
import gc
import torch

from .fuse_loras import apply_loras,apply_loras_gguf
from .module_ops import ModuleOps
from .primitives import (
    LoRAAdaptableProtocol,
    LoraPathStrengthAndSDOps,
    LoraStateDictWithStrength,
    ModelBuilderProtocol,
    StateDict,
    StateDictLoader,
)
from .registry import DummyRegistry, Registry
from .sd_ops import SDOps, ContentReplacement, ContentMatching
from .sft_loader import SafetensorsModelStateDictLoader
from ..model.model_protocol import ModelConfigurator, ModelType
from diffusers.quantizers.gguf.utils import GGUFParameter,GGUFLinear
import torch.nn as nn  
from contextlib import nullcontext
from accelerate import init_empty_weights
from diffusers.utils import is_accelerate_available
from ..model.audio_vae import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
)
from ..model.transformer import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
)
from ...ltx_core.model.video_vae import (
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
)

logger: logging.Logger = logging.getLogger(__name__)



@dataclass(frozen=True)
class SingleGPUModelBuilder(Generic[ModelType], ModelBuilderProtocol[ModelType], LoRAAdaptableProtocol):
    """
    Builder for PyTorch models residing on a single GPU.
    """

    model_class_configurator: type[ModelConfigurator[ModelType]]
    model_path: str | tuple[str, ...]
    model_sd_ops: SDOps | None = None
    module_ops: tuple[ModuleOps, ...] = field(default_factory=tuple)
    loras: tuple[LoraPathStrengthAndSDOps, ...] = field(default_factory=tuple)
    model_loader: StateDictLoader = field(default_factory=SafetensorsModelStateDictLoader)
    registry: Registry = field(default_factory=DummyRegistry)
    load_model: str = "dit"
    

    def lora(self, lora_path: str, strength: float = 1.0, sd_ops: SDOps | None = None) -> "SingleGPUModelBuilder":
        return replace(self, loras=(*self.loras, LoraPathStrengthAndSDOps(lora_path, strength, sd_ops)))

    def model_config(self) -> dict:
        first_shard_path = self.model_path[0] if isinstance(self.model_path, tuple) else self.model_path
        return self.model_loader.metadata(first_shard_path,self.load_model)

    def meta_model(self, config: dict, module_ops: tuple[ModuleOps, ...]) -> ModelType:
        with torch.device("meta"):
            model = self.model_class_configurator.from_config(config)
        for module_op in module_ops:
            if module_op.matcher(model):
                model = module_op.mutator(model)
        return model

    def meta_model_(self, config: dict, module_ops: tuple[ModuleOps, ...]) -> ModelType:
        ctx = init_empty_weights if is_accelerate_available() else nullcontext
        with ctx():
            model = self.model_class_configurator.from_config(config)
        for module_op in module_ops:
            if module_op.matcher(model):
                model = module_op.mutator(model)
        return model
    
    def load_sd(
        self, paths: list[str], registry: Registry, device: torch.device | None, sd_ops: SDOps | None = None,gguf_dit: bool = False
    ) -> StateDict:
        if gguf_dit:
            if  isinstance(paths, str) and paths.endswith(".safetensors"):
                if self.load_model == "vae" or self.load_model == "audio_vae":
                    from safetensors.torch import load_file
                    state_dict = load_file(paths)
                else:
                    state_dict = self.model_loader.load(paths, sd_ops=sd_ops, device=device)
            else:
                state_dict = load_gguf_checkpoint(paths[0], sd_ops=sd_ops)
        else:
            state_dict = registry.get(paths, sd_ops)
            if state_dict is None:
                state_dict = self.model_loader.load(paths, sd_ops=sd_ops, device=device)
            registry.add(paths, sd_ops=sd_ops, state_dict=state_dict)
        return state_dict
    

    def _return_model(self, meta_model: ModelType, device: torch.device) -> ModelType:
        uninitialized_params = [name for name, param in meta_model.named_parameters() if str(param.device) == "meta"]
        uninitialized_buffers = [name for name, buffer in meta_model.named_buffers() if str(buffer.device) == "meta"]
        if uninitialized_params or uninitialized_buffers:
            logger.warning(f"Uninitialized parameters or buffers: {uninitialized_params + uninitialized_buffers}")
            return meta_model
        retval = meta_model.to(device)
        return retval

    def build(self, device: torch.device | None = None, dtype: torch.dtype | None = None, gguf_dit: bool = False,encoded: bool = True) -> ModelType:
        device = torch.device("cuda") if device is None else device
        config = self.model_config()
        
        if self.load_model in ["vae","audio","clip","spatial"]:
            first_shard_path = self.model_path[0] if isinstance(self.model_path, tuple) else self.model_path
            meta_model = self.meta_model_(config, self.module_ops)
            model_state_dict = self.load_sd(first_shard_path, sd_ops=self.model_sd_ops, registry=self.registry, device=device, gguf_dit=gguf_dit)
            if self.load_model == "vae" :
                if encoded and  self.load_model == "vae":
                    sd = {key.replace("encoder.",""): value for key, value in model_state_dict.items() if  not key.startswith("decoder.") }
                elif not encoded and  self.load_model == "vae":
                    sd = {key.replace("decoder.",""): value for key, value in model_state_dict.items() if  not key.startswith("encoder.") }
                if dtype is not None  and not gguf_dit :
                    sd = {key: value.to(dtype=dtype) for key, value in model_state_dict.items()}
                del model_state_dict
                gc.collect()
            else:
                sd = model_state_dict.sd
                del model_state_dict
                gc.collect()
                if dtype is not None  and not gguf_dit :
                    sd = {key: value.to(dtype=dtype) for key, value in sd.items()}
            # 打印 meta_model 和 state_dict 的键以进行比较
            
           
            meta_model_keys = set(meta_model.state_dict().keys())   
            state_dict_keys = set(sd.keys())
 
            # 打印匹配的键的数量
            matching_keys = meta_model_keys.intersection(state_dict_keys)
            print(f"Matching keys count: {len(matching_keys)}")
            
            # 打印不在 meta_model 中但在 state_dict 中的键（多余键）
            extra_keys = state_dict_keys - meta_model_keys
            if extra_keys:
                print(f"Extra keys in state_dict (not in meta_model): {len(extra_keys)}")
                for key in list(extra_keys)[:10]:  # 只显示前10个
                    print(f"  - {key}")
            
            # 打印不在 state_dict 中但在 meta_model 中的键（缺失键）
            missing_keys = meta_model_keys - state_dict_keys
            if missing_keys:
                print(f"Missing keys in state_dict (not in state_dict): {len(missing_keys)}")
                for key in list(missing_keys)[:10]:  # 只显示前10个
                    print(f"  - {key}")
            
            # 如果需要，也可以打印部分匹配的键
            print(f"Sample matching keys: {list(matching_keys)[:5]}")

            meta_model.load_state_dict(sd, strict=False, assign=True)
            del sd
            gc.collect()
            return self._return_model(meta_model, device)
                
        elif gguf_dit :
            meta_model = self.meta_model_(config, self.module_ops)
        else:
            meta_model = self.meta_model(config, self.module_ops)
        
        model_paths = self.model_path if isinstance(self.model_path, tuple) else [self.model_path]

        model_state_dict = self.load_sd(model_paths, sd_ops=self.model_sd_ops, registry=self.registry, device=device, gguf_dit=gguf_dit)

        lora_strengths = [lora.strength for lora in self.loras]
        if not lora_strengths or (min(lora_strengths) == 0 and max(lora_strengths) == 0):
            if gguf_dit:
               meta_model= self.set_gguf2meta_model(meta_model,model_state_dict,dtype,device,None)
               del model_state_dict
               gc.collect()
            else:
                sd = model_state_dict.sd
                del model_state_dict
                gc.collect()
                if dtype is not None  and not gguf_dit :
                    sd = {key: value.to(dtype=dtype) for key, value in sd.items()}
                meta_model.load_state_dict(sd, strict=False, assign=True)
                del sd
                gc.collect()
            return self._return_model(meta_model, device)

        lora_state_dicts = [
            self.load_sd([lora.path], sd_ops=lora.sd_ops, registry=self.registry, device=device) for lora in self.loras
        ]
        lora_sd_and_strengths = [
            LoraStateDictWithStrength(sd, strength)
            for sd, strength in zip(lora_state_dicts, lora_strengths, strict=True)
        ]
        del lora_state_dicts
        gc.collect()
        if self.load_model == "dit":
            meta_model= self.set_gguf2meta_model(meta_model,model_state_dict,dtype,device,lora_sd_and_strengths)
            del model_state_dict
            gc.collect()
        else:
            final_sd = apply_loras(
                model_sd=model_state_dict,
                lora_sd_and_strengths=lora_sd_and_strengths,
                dtype=dtype,
                destination_sd=model_state_dict if isinstance(self.registry, DummyRegistry) else None,
            )
            del model_state_dict
            gc.collect()
            meta_model.load_state_dict(final_sd.sd, strict=False, assign=True)
            del final_sd
            gc.collect()
        return self._return_model(meta_model, device)
        
    def set_gguf2meta_model(self,meta_model,model_state_dict,dtype,device,lora_sd_and_strengths=None):
        from diffusers import GGUFQuantizationConfig
        from diffusers.quantizers.gguf import GGUFQuantizer

        g_config = GGUFQuantizationConfig(compute_dtype=dtype or torch.bfloat16)
        hf_quantizer = GGUFQuantizer(quantization_config=g_config)
        hf_quantizer.pre_quantized = True
        
        if lora_sd_and_strengths is not None:
            print("Applying LoRAs to GGUF model")
            model_state_dict=apply_loras_gguf(model_state_dict, lora_sd_and_strengths, dtype)

        hf_quantizer._process_model_before_weight_loading(
            meta_model,
            device_map={"": device} if device else None,
            state_dict=model_state_dict
        )
        from diffusers.models.model_loading_utils import load_model_dict_into_meta
        load_model_dict_into_meta(
            meta_model, 
            model_state_dict, 
            hf_quantizer=hf_quantizer,
            device_map={"": device} if device else None,
            dtype=dtype
        )

        hf_quantizer._process_model_after_weight_loading(meta_model)

        # 修复：确保 video_args_preprocessor.simple_preprocessor.patchify_proj 引用正确的模块
        if hasattr(meta_model, 'video_args_preprocessor'):
            vp = meta_model.video_args_preprocessor
            if hasattr(vp, 'simple_preprocessor'):
                sp = vp.simple_preprocessor
                # 找到 LTXModel 中的 patchify_proj 模块
                if hasattr(meta_model, 'patchify_proj'):
                    new_patchify_proj = meta_model.patchify_proj
                    sp.patchify_proj = new_patchify_proj
            
        if hasattr(meta_model, 'audio_args_preprocessor'):
            ap = meta_model.audio_args_preprocessor
            if hasattr(ap, 'simple_preprocessor'):
                sp = ap.simple_preprocessor
                # 找到 LTXModel 中的 audio_patchify_proj 模块
                if hasattr(meta_model, 'audio_patchify_proj'):
                    new_audio_patchify_proj = meta_model.audio_patchify_proj
                    sp.patchify_proj = new_audio_patchify_proj
            
        del model_state_dict
        gc.collect()
        return meta_model.to(dtype=dtype)



def load_gguf_checkpoint(gguf_checkpoint_path, sd_ops=None, return_tensors=False):
    """
    Load a GGUF file and return a dictionary of parsed parameters containing tensors, the parsed tokenizer and config
    attributes.

    Args:
        gguf_checkpoint_path (`str`):
            The path the to GGUF file to load
        sd_ops (`SDOps`, optional):
            SDOps object to apply key transformations during loading. If provided, keys will be transformed
            during loading to avoid extra memory overhead.
        return_tensors (`bool`, defaults to `True`):
            Whether to read the tensors from the file and return them. Not doing so is faster and only loads the
            metadata in memory.
    """
    from  diffusers.utils  import is_gguf_available, is_torch_available
    if is_gguf_available() and is_torch_available():
        import gguf
        from gguf import GGUFReader
        from diffusers.quantizers.gguf.utils import SUPPORTED_GGUF_QUANT_TYPES, GGUFParameter,dequantize_gguf_tensor
    else:
        logger.error(
            "Loading a GGUF checkpoint in PyTorch, requires both PyTorch and GGUF>=0.10.0 to be installed. Please see "
            "https://pytorch.org/ and https://github.com/ggerganov/llama.cpp/tree/master/gguf-py for installation instructions."
        )
        raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

    reader = GGUFReader(gguf_checkpoint_path)

    parsed_parameters = {}
    
    # 检查是否为简单替换（如LTXV_MODEL_COMFY_RENAMING_MAP）
    is_simple_replacement = False
    replacement_from = None
    replacement_to = None
    
    if sd_ops is not None and hasattr(sd_ops, 'mapping'):
        # 检查是否只有一个ContentReplacement操作
        replacements = [op for op in sd_ops.mapping if isinstance(op, ContentReplacement)]
        matchings = [op for op in sd_ops.mapping if isinstance(op, ContentMatching)]
        
        # 如果是简单的替换（只有一个替换操作，没有匹配限制或匹配为空）
        if len(replacements) == 1 and (not matchings or (matchings[0].prefix == "" and matchings[0].suffix == "")):
            is_simple_replacement = True
            replacement_from = replacements[0].content
            replacement_to = replacements[0].replacement

    
    for i, tensor in enumerate(reader.tensors):
        name = tensor.name
        quant_type = tensor.tensor_type

        # if the tensor is a torch supported dtype do not use GGUFParameter
        is_gguf_quant = quant_type not in [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16]
        if is_gguf_quant and quant_type not in SUPPORTED_GGUF_QUANT_TYPES:
            _supported_quants_str = "\n".join([str(type) for type in SUPPORTED_GGUF_QUANT_TYPES])
            raise ValueError(
                (
                    f"{name} has a quantization type: {str(quant_type)} which is unsupported."
                    "\n\nCurrently the following quantization types are supported: \n\n"
                    f"{_supported_quants_str}"
                    "\n\nTo request support for this quantization type please open an issue here: https://github.com/huggingface/diffusers"
                )
            )

        weights = torch.from_numpy(tensor.data.copy())
        if sd_ops is not None:
            if is_simple_replacement:
                if replacement_from in name:
                    new_key = name.replace(replacement_from, replacement_to)
                    parsed_parameters[new_key] = GGUFParameter(weights, quant_type=quant_type) if is_gguf_quant else weights
            else:
                expected_key = sd_ops.apply_to_key(name)
                if expected_key is None:
                    # 如果键不匹配，跳过这个张量
                    continue
                    
                # 应用键值对转换
                key_value_pairs = sd_ops.apply_to_key_value(expected_key, weights)
                
                for new_key, new_value in key_value_pairs:
                    # 存储转换后的键值对
                    parsed_parameters[new_key] = GGUFParameter(new_value, quant_type=quant_type) if is_gguf_quant else new_value
        else:
            # 如果没有sd_ops，使用原始键名
            parsed_parameters[name] = GGUFParameter(weights, quant_type=quant_type) if is_gguf_quant else weights
        del tensor,weights
        if i > 0 and i % 1000 == 0:  # 每1000个tensor执行一次gc
            logger.info(f"Processed {i}tensors...")
            gc.collect()
    del reader
    gc.collect()
    return parsed_parameters
