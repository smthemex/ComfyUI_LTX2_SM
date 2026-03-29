from enum import Enum
import gc
from typing import Any, Callable, List, Optional, Tuple
import torch
import torch.nn as nn
from ...guidance.perturbations import BatchedPerturbationConfig
from .adaln import AdaLayerNormSingle, adaln_embedding_coefficient
from .attention import AttentionCallable, AttentionFunction
from .modality import Modality
from .rope import LTXRopeType
from .transformer import BasicAVTransformerBlock, TransformerConfig
from .transformer_args import (
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
    TransformerArgsPreprocessor,
)
from ...utils import to_denoised
import copy

class BlockGPUManager_:
    def __init__(self, device="cuda",block_group_size=1 ):
        self.device = torch.device(device)
        self.managed_modules = [] 
        self.submodule = []    
        self.block_group_size = block_group_size
        self._original_model_ref = None
        self._original_blocks_ref = None

    def setup_for_inference(self, transformer_model, ):
        self._collect_managed_modules(transformer_model)
        self._initialize_submodule()
        return self

    def _collect_managed_modules(self, transformer_model):
        self.submodule = []
        self._original_model_ref = transformer_model
        self._original_blocks_ref = transformer_model.transformer_blocks
    
        for attr in ['patchify_proj', 'audio_patchify_proj', 'adaln_single', 
                     'audio_adaln_single', 'caption_projection','audio_caption_projection',
                     "audio_scale_shift_table", "scale_shift_table","av_ca_video_scale_shift_adaln_single",
                     "av_ca_audio_scale_shift_adaln_single", "av_ca_a2v_gate_adaln_single",
                     "av_ca_v2a_gate_adaln_single", "prompt_adaln_single", "audio_prompt_adaln_single",
                     "cross_attn_rope", "cross_attn_audio_rope",  
                     "norm_out", "proj_out", "audio_norm_out","audio_proj_out",
                     ]:
            if hasattr(transformer_model, attr):
                self.submodule.append(getattr(transformer_model, attr))

        self.managed_modules = [None] * len(self._original_blocks_ref)

    def _get_block(self, block_index):
        """按需获取层，避免一次性加载所有层"""
        if self.managed_modules[block_index] is None:
            block = self._original_blocks_ref[block_index]
            block.to(self.device)
            self.managed_modules[block_index] = block
        return self.managed_modules[block_index]
    
    def _pass_block(self, block_index):
        if block_index > 0 and (block_index - 1) < len(self.managed_modules):
            prev_block = self.managed_modules[block_index - 1]
            if prev_block is not None and hasattr(prev_block, 'to'):
                prev_block.to('cpu')
                self.managed_modules[block_index - 1] = None
            del prev_block
            torch.cuda.empty_cache()
            gc.collect()


    def _initialize_submodule(self):
        for module in self.submodule:
            if hasattr(module, 'to'):
                module.to(self.device)
        return self

    def unload_blocks_to_cpu(self):
        for  module in self.managed_modules:
            if hasattr(module, 'to'):
                module.to('cpu')
        
        for module in self.submodule:
            if hasattr(module, 'to'):
                module.to('cpu', non_blocking=True)
        torch.cuda.empty_cache()
        return self


class BlockGPUManager:
    def __init__(self, device="cuda", block_group_size=2):
        self.device = torch.device(device)
        self.managed_modules = []
        self.submodule = []
        self.block_group_size = block_group_size  # 每次加载的连续层数
        self._original_model_ref = None
        self._original_layers_ref = None
        self._num_groups = 0  # 总批次数
        self._current_group = -1  # 当前正在计算的批次索引
        self._prefetched_group = -1  # 预取中的批次索引
        self._group_loaded: list[bool] = []
        self._group_ready_events: list[Optional[torch.cuda.Event]] = []
        self._prefetch_stream: Optional[torch.cuda.Stream] = None
        self._parameter_pinned = False  # 只需 pin memory 一次

    def setup_for_inference(self, transformer_model):
        self._collect_managed_modules(transformer_model)
        self._initialize_submodule()
        self._create_prefetch_stream()
        return self

    def _create_prefetch_stream(self):
        if self.device.type == "cuda":
            self._prefetch_stream = torch.cuda.Stream(device=self.device)
        else:
            self._prefetch_stream = None

    def _collect_managed_modules(self, transformer_model):
        self.submodule = []
        self._original_model_ref = transformer_model
        self._original_layers_ref = transformer_model.transformer_blocks
        self._num_groups = (len(self._original_layers_ref) + self.block_group_size - 1) // self.block_group_size
        print(f"Total number of layers: {len(self._original_layers_ref)}")
        print(f"Total number of groups: {self._num_groups}")

        # pin memory only for CPU path; if weights are already on GPU, move them back first.
        if self.device.type == "cuda" and not self._parameter_pinned:
            for layer in self._original_layers_ref:
                if any(p.is_cuda for p in layer.parameters()):
                    layer.to("cpu")
                for p in layer.parameters():
                    if not p.is_pinned():
                        p.pin_memory()
                for b in layer.buffers():
                    if not b.is_pinned():
                        b.pin_memory()
            self._parameter_pinned = True

        for attr in ['patchify_proj', 'audio_patchify_proj', 'adaln_single', 
                     'audio_adaln_single', 'caption_projection','audio_caption_projection',
                     "audio_scale_shift_table", "scale_shift_table","av_ca_video_scale_shift_adaln_single",
                     "av_ca_audio_scale_shift_adaln_single", "av_ca_a2v_gate_adaln_single",
                     "av_ca_v2a_gate_adaln_single", "prompt_adaln_single", "audio_prompt_adaln_single",
                     "cross_attn_rope", "cross_attn_audio_rope",  
                     "norm_out", "proj_out", "audio_norm_out","audio_proj_out",
                     ]:
            if hasattr(transformer_model, attr):
                self.submodule.append(getattr(transformer_model, attr))

        self.managed_modules = [None] * self._num_groups
        self._group_loaded = [False] * self._num_groups
        self._group_ready_events = [None] * self._num_groups

    def _get_layer(self, layer_index):
        """按需获取批次中的层，实现双缓冲预取"""
        group_index = layer_index // self.block_group_size
        local_idx = layer_index % self.block_group_size

        next_group = group_index + 1
        keep = {group_index}
        if next_group < self._num_groups:
            keep.add(next_group)

        
        self._unload_unused_groups(keep=keep)

        if not self._group_loaded[group_index]:
            self._prefetch_group(group_index)
        self._wait_group_ready(group_index)

        if next_group < self._num_groups and not self._group_loaded[next_group]:
            self._prefetch_group(next_group)

        self._current_group = group_index
        return self.managed_modules[group_index][local_idx]

    def _prefetch_group(self, group_index):
        if self._group_loaded[group_index]:
            return

        start_idx = group_index * self.block_group_size
        end_idx = min(start_idx + self.block_group_size, len(self._original_layers_ref))

        group = nn.ModuleList()
        if self._prefetch_stream is not None:
            with torch.cuda.stream(self._prefetch_stream):
                for layer in self._original_layers_ref[start_idx:end_idx]:
                    #cpu_layer = copy.deepcopy(layer)
                    layer.to(self.device, non_blocking=True)
                    group.append(layer)
                    # cpu_layer = copy.deepcopy(layer)
                    # cpu_layer.to(self.device, non_blocking=True)
                    # group.append(cpu_layer)
                event = torch.cuda.Event()
                event.record(self._prefetch_stream)
            self._group_ready_events[group_index] = event
        else:
            for layer in self._original_layers_ref[start_idx:end_idx]:
                layer.to(self.device)
                group.append(layer)
                # cpu_layer = copy.deepcopy(layer)
                # cpu_layer.to(self.device)
                # group.append(cpu_layer)
            self._group_ready_events[group_index] = None

        self.managed_modules[group_index] = group
        self._group_loaded[group_index] = True
        self._prefetched_group = group_index

    def _wait_group_ready(self, group_index):
        event = self._group_ready_events[group_index]
        if event is not None:
            torch.cuda.current_stream(self.device).wait_event(event)
            self._group_ready_events[group_index] = None

    def _unload_unused_groups(self, keep: set[int]):
        for index in range(self._num_groups):
            if self._group_loaded[index] and index not in keep:
                self._unload_group(index)

    def _unload_group(self, group_index):
        if not self._group_loaded[group_index]:
            return

        group = self.managed_modules[group_index]
        if group is not None:
            for layer in group:
                if hasattr(layer, 'to'):
                    layer.to('cpu')
                    layer = None

        self.managed_modules[group_index] = None
        self._group_loaded[group_index] = False
        self._group_ready_events[group_index] = None
        if self._current_group == group_index:
            self._current_group = -1
        if self._prefetched_group == group_index:
            self._prefetched_group = -1
        del group
        torch.cuda.empty_cache()
        gc.collect()

    def _initialize_submodule(self):
        for module in self.submodule:
            if hasattr(module, 'to'):
                module.to(self.device)
        return self

    def unload_blocks_to_cpu(self):
        # 卸载所有批次
        for group_index in range(self._num_groups):
            self._unload_group(group_index)
        self._current_group = -1
        self._prefetched_group = -1
        
        # 将embedder和output模块移到CPU
        for module in self.submodule:
            if hasattr(module, 'to'):
                module.to('cpu')
        torch.cuda.empty_cache()
        return self



class LTXModelType(Enum):
    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class LTXModel(torch.nn.Module):
    """
    LTX model transformer implementation.
    This class implements the transformer blocks for the LTX model.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_type: LTXModelType = LTXModelType.AudioVideo,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-06,
        attention_type: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        audio_positional_embedding_max_pos: list[int] | None = None,
        av_ca_timestep_scale_multiplier: int = 1,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        double_precision_rope: bool = False,
        apply_gated_attention: bool = False,
        caption_projection: torch.nn.Module | None = None,
        audio_caption_projection: torch.nn.Module | None = None,
        cross_attention_adaln: bool = False,
    ):
        super().__init__()
        self._enable_gradient_checkpointing = False
        self.cross_attention_adaln = cross_attention_adaln
        self.use_middle_indices_grid = use_middle_indices_grid
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.model_type = model_type
        cross_pe_max_pos = None
        if model_type.is_video_enabled():
            if positional_embedding_max_pos is None:
                positional_embedding_max_pos = [20, 2048, 2048]
            self.positional_embedding_max_pos = positional_embedding_max_pos
            self.num_attention_heads = num_attention_heads
            self.inner_dim = num_attention_heads * attention_head_dim
            self._init_video(
                in_channels=in_channels,
                out_channels=out_channels,
                norm_eps=norm_eps,
                caption_projection=caption_projection,
            )

        if model_type.is_audio_enabled():
            if audio_positional_embedding_max_pos is None:
                audio_positional_embedding_max_pos = [20]
            self.audio_positional_embedding_max_pos = audio_positional_embedding_max_pos
            self.audio_num_attention_heads = audio_num_attention_heads
            self.audio_inner_dim = self.audio_num_attention_heads * audio_attention_head_dim
            self._init_audio(
                in_channels=audio_in_channels,
                out_channels=audio_out_channels,
                norm_eps=norm_eps,
                caption_projection=audio_caption_projection,
            )

        if model_type.is_video_enabled() and model_type.is_audio_enabled():
            cross_pe_max_pos = max(self.positional_embedding_max_pos[0], self.audio_positional_embedding_max_pos[0])
            self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier
            self.audio_cross_attention_dim = audio_cross_attention_dim
            self._init_audio_video(num_scale_shift_values=4)

        self._init_preprocessors(cross_pe_max_pos)
        # Initialize transformer blocks
        self._init_transformer_blocks(
            num_layers=num_layers,
            attention_head_dim=attention_head_dim if model_type.is_video_enabled() else 0,
            cross_attention_dim=cross_attention_dim,
            audio_attention_head_dim=audio_attention_head_dim if model_type.is_audio_enabled() else 0,
            audio_cross_attention_dim=audio_cross_attention_dim,
            norm_eps=norm_eps,
            attention_type=attention_type,
            apply_gated_attention=apply_gated_attention,
        )
        
    @property
    def _adaln_embedding_coefficient(self) -> int:
        return adaln_embedding_coefficient(self.cross_attention_adaln)
        
    def _init_video(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize video-specific components."""
        # Video input components
        self.patchify_proj = torch.nn.Linear(in_channels, self.inner_dim, bias=True)
        if caption_projection is not None:
            self.caption_projection = caption_projection
        self.adaln_single = AdaLayerNormSingle(self.inner_dim, embedding_coefficient=self._adaln_embedding_coefficient)
        

        self.prompt_adaln_single = (
            AdaLayerNormSingle(self.inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None
        )

        # Video output components
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = torch.nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = torch.nn.Linear(self.inner_dim, out_channels)

    def _init_audio(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize audio-specific components."""

        # Audio input components
        self.audio_patchify_proj = torch.nn.Linear(in_channels, self.audio_inner_dim, bias=True)
        if caption_projection is not None:
            self.audio_caption_projection = caption_projection
        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,embedding_coefficient=self._adaln_embedding_coefficient,
        )

        self.audio_prompt_adaln_single = (
            AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None
        )

        # Audio output components
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = torch.nn.LayerNorm(self.audio_inner_dim, elementwise_affine=False, eps=norm_eps)
        self.audio_proj_out = torch.nn.Linear(self.audio_inner_dim, out_channels)

    def _init_audio_video(
        self,
        num_scale_shift_values: int,
    ) -> None:
        """Initialize audio-video cross-attention components."""
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=1,
        )

        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=1,
        )

    def _init_preprocessors(
        self,
        cross_pe_max_pos: int | None = None,
    ) -> None:
        """Initialize preprocessors for LTX."""

        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            self.video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
            )
            self.audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )
        elif self.model_type.is_video_enabled():
            self.video_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
            )
        elif self.model_type.is_audio_enabled():
            self.audio_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        norm_eps: float,
        attention_type: AttentionFunction | AttentionCallable,
        apply_gated_attention: bool,
    ) -> None:
        """Initialize transformer blocks for LTX."""
        video_config = (
            TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            
            if self.model_type.is_video_enabled()
            else None
        )
        audio_config = (
            TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.audio_num_attention_heads,
                d_head=audio_attention_head_dim,
                context_dim=audio_cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_audio_enabled()
            else None
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    idx=idx,
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                    attention_function=attention_type,
                )
                for idx in range(num_layers)
            ]
        )

    def set_gradient_checkpointing(self, enable: bool) -> None:
        """Enable or disable gradient checkpointing for transformer blocks.
        Gradient checkpointing trades compute for memory by recomputing activations
        during the backward pass instead of storing them. This can significantly
        reduce memory usage at the cost of ~20-30% slower training.
        Args:
            enable: Whether to enable gradient checkpointing
        """
        self._enable_gradient_checkpointing = enable

    def _process_transformer_blocks(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig,
        gpu_manager: BlockGPUManager | None,
    ) -> tuple[TransformerArgs, TransformerArgs]:
        """Process transformer blocks for LTXAV."""

        # # Process transformer blocks
        # for block_index,block in enumerate(self.transformer_blocks):
        #     if gpu_manager is not None:
        #         # 加载当前block到GPU
        #         if block_index < len(gpu_manager.managed_modules):
        #             module = gpu_manager.managed_modules[block_index]
        #             if hasattr(module, 'to'):
        #                 module.to(gpu_manager.device)
        #                 #print(f"[GPU Manager] 加载dual block {block_index}到GPU")
                
        #         # 卸载上一个block（如果是第一个block，则不需要卸载）
        #         if block_index > 0 and (block_index - 1) < len(gpu_manager.managed_modules):
        #             prev_module = gpu_manager.managed_modules[block_index - 1]
        #             if hasattr(prev_module, 'to'):
        #                 prev_module.to('cpu')
        #                 #print(f"[GPU Manager] 卸载dual block {block_index-1}到CPU")
        for block_index in range(len(self.transformer_blocks)):
            if gpu_manager is not None:
                block = gpu_manager._get_layer(block_index)
                #print(f"[GPU Manager] 使用dual block {block_index}进行前向计算")
                #gpu_manager._pass_block(block_index) 
            
            else:
                block = self.transformer_blocks[block_index]

            if self._enable_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory during training.
                # With use_reentrant=False, we can pass dataclasses directly -
                # PyTorch will track all tensor leaves in the computation graph.
                video, audio = torch.utils.checkpoint.checkpoint(
                    block,
                    video,
                    audio,
                    perturbations,
                    use_reentrant=False,
                )
            else:
                video, audio = block(
                    video=video,
                    audio=audio,
                    perturbations=perturbations,
                )

        return video, audio

    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: torch.nn.LayerNorm,
        proj_out: torch.nn.Linear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Process output for LTXV."""
        # Apply scale-shift modulation
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)
        return x

    def forward(
        self, video: Modality | None, audio: Modality | None, perturbations: BatchedPerturbationConfig,gpu_manager=None, 
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for LTX models.
        Returns:
            Processed output tensors
        """
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")

        video_args = self.video_args_preprocessor.prepare(video, audio) if video is not None else None
        audio_args = self.audio_args_preprocessor.prepare(audio, video) if audio is not None else None
        # Process transformer blocks
        video_out, audio_out = self._process_transformer_blocks(
            video=video_args,
            audio=audio_args,
            perturbations=perturbations,
            gpu_manager=gpu_manager,
        )

        # Process output
        vx = (
            self._process_output(
                self.scale_shift_table, self.norm_out, self.proj_out, video_out.x, video_out.embedded_timestep
            )
            if video_out is not None
            else None
        )
        ax = (
            self._process_output(
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
                audio_out.x,
                audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )
        return vx, ax


class LegacyX0Model(torch.nn.Module):
    """
    Legacy X0 model implementation.
    Returns fully denoised output based on the velocities produced by the base model.
    """

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
        sigma: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations)
        denoised_video = to_denoised(video.latent, vx, sigma) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, sigma) if ax is not None else None
        return denoised_video, denoised_audio


class X0Model(torch.nn.Module):
    """
    X0 model implementation.
    Returns fully denoised outputs based on the velocities produced by the base model.
    Applies scaled denoising to the video and audio according to the timesteps = sigma * denoising_mask.
    """

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
        gpu_manager: BlockGPUManager,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations,gpu_manager)
        denoised_video = to_denoised(video.latent, vx, video.timesteps) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, audio.timesteps) if ax is not None else None
        return denoised_video, denoised_audio
