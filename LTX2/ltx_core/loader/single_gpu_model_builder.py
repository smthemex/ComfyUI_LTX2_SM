import logging
from dataclasses import dataclass, field, replace
from typing import Generic
from contextlib import nullcontext
from accelerate import init_empty_weights
from diffusers.utils import is_accelerate_available
import torch
import gc
from .fuse_loras import apply_loras
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
from .sd_ops import SDOps
from .sft_loader import SafetensorsModelStateDictLoader
from ..model.model_protocol import ModelConfigurator, ModelType

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SingleGPUModelBuilder(Generic[ModelType], ModelBuilderProtocol[ModelType], LoRAAdaptableProtocol):
    """
    Builder for PyTorch models residing on a single GPU.
    Attributes:
        model_class_configurator: Class responsible for constructing the model from a config dict.
        model_path: Path (or tuple of shard paths) to the model's `.safetensors` checkpoint(s).
        model_sd_ops: Optional state-dict operations applied when loading the model weights.
        module_ops: Sequence of module-level mutations applied to the meta model before weight loading.
        loras: Sequence of LoRA adapters (path, strength, optional sd_ops) to fuse into the model.
        model_loader: Strategy for loading state dicts from disk. Defaults to
            :class:`SafetensorsModelStateDictLoader`.
        registry: Cache for already-loaded state dicts. Defaults to :class:`DummyRegistry` (no caching).
        lora_load_device: Device used when loading LoRA weight tensors from disk. Defaults to
            ``torch.device("cpu")``, which keeps LoRA weights in CPU memory and transfers them to
            the target GPU sequentially during fusion, reducing peak GPU memory usage compared to
            loading all LoRA weights directly onto the GPU at once.
    """

    model_class_configurator: type[ModelConfigurator[ModelType]]
    model_path: str | tuple[str, ...]
    model_sd_ops: SDOps | None = None
    module_ops: tuple[ModuleOps, ...] = field(default_factory=tuple)
    loras: tuple[LoraPathStrengthAndSDOps, ...] = field(default_factory=tuple)
    model_loader: StateDictLoader = field(default_factory=SafetensorsModelStateDictLoader)
    registry: Registry = field(default_factory=DummyRegistry)
    lora_load_device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    load_model: str = "dit"
    def lora(self, lora_path: str, strength: float = 1.0, sd_ops: SDOps | None = None) -> "SingleGPUModelBuilder":
        return replace(self, loras=(*self.loras, LoraPathStrengthAndSDOps(lora_path, strength, sd_ops)))

    def with_sd_ops(self, sd_ops: SDOps | None) -> "SingleGPUModelBuilder":
        return replace(self, model_sd_ops=sd_ops)

    def with_module_ops(self, module_ops: tuple[ModuleOps, ...]) -> "SingleGPUModelBuilder":
        return replace(self, module_ops=module_ops)

    def with_loras(self, loras: tuple[LoraPathStrengthAndSDOps, ...]) -> "SingleGPUModelBuilder":
        return replace(self, loras=loras)

    def with_registry(self, registry: Registry) -> "SingleGPUModelBuilder":
        return replace(self, registry=registry)

    def with_lora_load_device(self, device: torch.device) -> "SingleGPUModelBuilder":
        return replace(self, lora_load_device=device)

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
    
    # def load_sd(
    #     self, paths: list[str], registry: Registry, device: torch.device | None, sd_ops: SDOps | None = None
    # ) -> StateDict:
    #     state_dict = registry.get(paths, sd_ops)
    #     if state_dict is None:
    #         state_dict = self.model_loader.load(paths, sd_ops=sd_ops, device=device)
    #         registry.add(paths, sd_ops=sd_ops, state_dict=state_dict)
    #     return state_dict

    def load_sd(
        self, paths: list[str], registry: Registry, device: torch.device | None, sd_ops: SDOps | None = None,use_gguf: bool = False
    ) -> StateDict:
        if use_gguf:
            if  isinstance(paths, str) and paths.endswith(".safetensors"):
                if self.load_model in["imageconditioner", "videodecoder" ]:
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

    def build(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        use_gguf: bool = False,
        **kwargs: object,  # noqa: ARG002
    ) -> ModelType:
       
        
        device = torch.device("cuda") if device is None else device
        config = self.model_config()
        if self.load_model in ["audiodecoder","audioconditioner","prompt_encoder","upsampler","imageconditioner","videodecoder"]:
            first_shard_path = self.model_path[0] if isinstance(self.model_path, tuple) else self.model_path
            meta_model = self.meta_model_(config, self.module_ops)
            if self.load_model == "prompt_encoder"  and first_shard_path.endswith(".gguf"):
                from diffusers import GGUFQuantizationConfig
                from diffusers.quantizers.gguf import GGUFQuantizer
                g_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
                hf_quantizer = GGUFQuantizer(quantization_config=g_config)
                hf_quantizer.pre_quantized = True
                print("loading prompt_encoder",first_shard_path)
                model_state_dict=load_gguf_checkpoint_gemma(first_shard_path) 
            else:
                model_state_dict = self.load_sd(first_shard_path, sd_ops=self.model_sd_ops, registry=self.registry, device=device, use_gguf=use_gguf)
            if self.load_model in["imageconditioner","videodecoder" ]:
                if  self.load_model == "imageconditioner" :
                    sd = {key.replace("encoder.",""): value for key, value in model_state_dict.items() if  not key.startswith("decoder.") }
                else:
                    sd = {key.replace("decoder.",""): value for key, value in model_state_dict.items() if  not key.startswith("encoder.") }
                
                if dtype is not None  and not use_gguf :
                    sd = {key: value.to(dtype=dtype) for key, value in model_state_dict.items()}
                del model_state_dict
                gc.collect()
                #match_state_dict(meta_model, sd,show_num=10)
            elif self.load_model == "prompt_encoder"and first_shard_path.endswith(".gguf") :
                def adjust_key_name(key):
                    from packaging import version
                    import transformers
                    transformers_version = version.parse(transformers.__version__)
                    required_version = version.parse('4.57.0')
                    if "vision_model.embeddings.position_ids" in key or "language_model.embed_tokens.embed_scale" in key:
                        print(key)
                    if transformers_version >= required_version:
                        if "language_model.model." in key:
                            return 'model.model.language_model.' + key[len('language_model.model.'):]
                        else:
                            return 'model.model.' + key
                    else:
                        if key.startswith('language_model.model.') :
                            return 'model.language_model.' + key[len('language_model.model.'):]
                        elif key.startswith('vision_tower.'):
                            return 'model.' + key
                        elif key.startswith('multi_modal_projector.'):
                            return 'model.' + key
                        else:
                            return key
                #match_state_dict(meta_model, model_state_dict,50)    
                adjusted_state_dict = {}
                for key, value in model_state_dict.items():
                    new_key = adjust_key_name(key)
                    adjusted_state_dict[new_key] = value
                del model_state_dict
                #match_state_dict(meta_model, adjusted_state_dict,50)
                has_embed_tokens_key = False

                for key in adjusted_state_dict.keys():
                    if 'lm_head.weight' in key:
                        has_embed_tokens_key = True
                        break
                for key in adjusted_state_dict.keys():
                    if 'embed_tokens.weight' in key:
                        embed_tokens_key = key
                        break
                print("has_embed_tokens_key",has_embed_tokens_key)     
                if not  has_embed_tokens_key:
                    adjusted_state_dict['model.lm_head.weight'] = adjusted_state_dict[embed_tokens_key]
                #match_state_dict(meta_model, adjusted_state_dict,50)    

                hf_quantizer._process_model_before_weight_loading(
                    meta_model,
                    device_map=None,
                    state_dict=adjusted_state_dict
                )

                from diffusers.models.model_loading_utils import load_model_dict_into_meta

                load_model_dict_into_meta(
                    meta_model, 
                    adjusted_state_dict, 
                    hf_quantizer=hf_quantizer,
                    device_map=None,
                    dtype=torch.bfloat16
                )
                    
                hf_quantizer._process_model_after_weight_loading(meta_model)
                meta_model.eval()
                gc.collect()
                return self._return_model(meta_model, device)
            else:
                sd = model_state_dict.sd
                del model_state_dict
                gc.collect()
                if dtype is not None  and not use_gguf :
                    sd = {key: value.to(dtype=dtype) for key, value in sd.items()}
            # 打印 meta_model 和 state_dict 的键以进行比较
            
           
            #match_state_dict(meta_model, sd)
            meta_model.load_state_dict(sd, strict=False, assign=True)
            del sd
            gc.collect()
            return self._return_model(meta_model, device)
                
        elif self.load_model=="dit" :
            meta_model = self.meta_model_(config, self.module_ops)
        else:
            meta_model = self.meta_model(config, self.module_ops)
        
        model_paths = self.model_path if isinstance(self.model_path, tuple) else [self.model_path]

        model_state_dict = self.load_sd(model_paths, sd_ops=self.model_sd_ops, registry=self.registry, device=device, use_gguf=use_gguf)

        lora_strengths = [lora.strength for lora in self.loras]
        if not lora_strengths or (min(lora_strengths) == 0 and max(lora_strengths) == 0):
            if use_gguf:
               print("use_gguf no lora",use_gguf)
               #match_state_dict(meta_model, model_state_dict,show_num=10)
               meta_model= self.set_gguf2meta_model(meta_model,model_state_dict,dtype,device,None)
               del model_state_dict
               gc.collect()
            else:
                sd = model_state_dict.sd
                del model_state_dict
                gc.collect()
                if dtype is not None  and not use_gguf :
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
            print("use_gguf add lora",use_gguf)
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

        # meta_model = self.meta_model(config, self.module_ops)
        # model_paths = list(self.model_path) if isinstance(self.model_path, tuple) else [self.model_path]
        # model_state_dict = self.load_sd(model_paths, sd_ops=self.model_sd_ops, registry=self.registry, device=device)

        # lora_strengths = [lora.strength for lora in self.loras]
        # if not lora_strengths or (min(lora_strengths) == 0 and max(lora_strengths) == 0):
        #     sd = model_state_dict.sd
        #     if dtype is not None:
        #         sd = {key: value.to(dtype=dtype) for key, value in model_state_dict.sd.items()}
        #     meta_model.load_state_dict(sd, strict=False, assign=True)
        #     return self._return_model(meta_model, device)

        # lora_state_dicts = [
        #     self.load_sd([lora.path], sd_ops=lora.sd_ops, registry=self.registry, device=self.lora_load_device)
        #     for lora in self.loras
        # ]
        # lora_sd_and_strengths = [
        #     LoraStateDictWithStrength(sd, strength)
        #     for sd, strength in zip(lora_state_dicts, lora_strengths, strict=True)
        # ]
        # final_sd = apply_loras(
        #     model_sd=model_state_dict,
        #     lora_sd_and_strengths=lora_sd_and_strengths,
        #     dtype=dtype,
        #     destination_sd=model_state_dict if isinstance(self.registry, DummyRegistry) else None,
        # )
        # meta_model.load_state_dict(final_sd.sd, strict=False, assign=True)
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

def apply_loras_gguf(
    model_sd,
    lora_sd_and_strengths: list[LoraStateDictWithStrength],
    dtype: torch.dtype,
):
    sd = {}
    device = torch.device("meta")
    for key, weight in model_sd.items():
        if weight is None:
            continue
        device = weight.device
        #target_dtype = dtype if dtype is not None else weight.dtype
        #deltas_dtype = target_dtype if target_dtype not in [torch.float8_e4m3fn, torch.float8_e5m2] else torch.bfloat16
        deltas_dtype =  torch.bfloat16
        deltas = _prepare_deltas(lora_sd_and_strengths, key, deltas_dtype, device)
        if deltas is None:
            deltas = weight
        elif weight.dtype == torch.bfloat16:
            deltas.add_(weight)
        else:
            raise ValueError(f"Unsupported dtype: {weight.dtype}")
        sd[key] = deltas
        del weight,deltas
    del model_sd
    gc.collect()
    return sd

def _prepare_deltas(
    lora_sd_and_strengths: list[LoraStateDictWithStrength], key: str, dtype: torch.dtype, device: torch.device
) -> torch.Tensor | None:
    deltas = []
    prefix = key[: -len(".weight")]
    key_a = f"{prefix}.lora_A.weight"
    key_b = f"{prefix}.lora_B.weight"
    for lsd, coef in lora_sd_and_strengths:
        if key_a not in lsd.sd or key_b not in lsd.sd:
            continue
        a = lsd.sd[key_a].to(device=device)
        b = lsd.sd[key_b].to(device=device)
        product = torch.matmul(b * coef, a)
        del a, b
        deltas.append(product.to(dtype=dtype))
    if len(deltas) == 0:
        return None
    elif len(deltas) == 1:
        return deltas[0]
    return torch.sum(torch.stack(deltas, dim=0), dim=0)

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
    from .sd_ops import SDOps, ContentReplacement, ContentMatching
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

        weights = torch.from_numpy(tensor.data) #tensor.data.copy()
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
            #logger.info(f"Processed {i}tensors...")
            gc.collect()
    del reader
    gc.collect()
    return parsed_parameters

def match_state_dict(meta_model, sd,show_num=10):

    meta_model_keys = set(meta_model.state_dict().keys())   
    state_dict_keys = set(sd.keys())

    # 打印匹配的键的数量
    matching_keys = meta_model_keys.intersection(state_dict_keys)
    print(f"Matching keys count: {len(matching_keys)}")
    
    # 打印不在 meta_model 中但在 state_dict 中的键（多余键）
    extra_keys = state_dict_keys - meta_model_keys
    if extra_keys:
        print(f"Extra keys in state_dict (not in meta_model): {len(extra_keys)}")
        for key in list(extra_keys)[:show_num]:  # 只显示前10个
            print(f"  - {key}")
    
    # 打印不在 state_dict 中但在 meta_model 中的键（缺失键）
    missing_keys = meta_model_keys - state_dict_keys
    if missing_keys:
        print(f"Missing keys in state_dict (not in state_dict): {len(missing_keys)}")
        for key in list(missing_keys)[:show_num]:  # 只显示前10个
            print(f"  - {key}")
    
    # 如果需要，也可以打印部分匹配的键
    print(f"Sample matching keys: {list(matching_keys)[:5]}")

def load_gguf_checkpoint_gemma(gguf_checkpoint_path):

    from  diffusers.utils  import is_gguf_available, is_torch_available
    if is_gguf_available() and is_torch_available():
        import gguf
        from gguf import GGUFReader
        from diffusers.quantizers.gguf.utils import SUPPORTED_GGUF_QUANT_TYPES, GGUFParameter,dequantize_gguf_tensor
    else:
        raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

    reader = GGUFReader(gguf_checkpoint_path)
    parsed_parameters = {}
 
    for tensor in reader.tensors:
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
        parsed_parameters[name] = GGUFParameter(weights, quant_type=quant_type) if is_gguf_quant else weights
    
    del reader
    gc.collect()
    return parsed_parameters