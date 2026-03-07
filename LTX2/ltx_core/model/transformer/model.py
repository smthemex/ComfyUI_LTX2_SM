from enum import Enum

import torch

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

class BlockGPUManager:
    def __init__(self, device="cuda",block_group_size=1 ):
        self.device = device
        self.managed_modules = []
        self.embedder_modules = []
        self.output_modules = []     
        self.block_group_size = block_group_size
        # 跟踪哪些blocks当前在GPU上
        self.blocks_on_gpu = set()
    
    def get_gpu_memory_usage(self):
        """获取GPU内存使用情况"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
            total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024  # MB
            free = total - allocated
            
            return {
                'allocated_mb': allocated,
                'reserved_mb': reserved,
                'total_mb': total,
                'free_mb': free
            }
        return None
    
    def setup_for_inference(self, transformer_model, ):
        self._collect_managed_modules(transformer_model)
        self._initialize_embedder_output_modules()
        return self

    
    def _collect_managed_modules(self, transformer_model):
        """收集所有要管理的模块（简化版本，使用已知的block大小）"""
        self.managed_modules = []
        self.embedder_modules = []
        self.output_modules = []

        for i, block in enumerate(transformer_model.transformer_blocks):
            self.managed_modules.append(block)
    
        # 收集子模块
        if hasattr(transformer_model, 'patchify_proj'):
            self.embedder_modules.append(transformer_model.patchify_proj)
        if hasattr(transformer_model, 'audio_patchify_proj'):
            self.embedder_modules.append(transformer_model.audio_patchify_proj)

        if hasattr(transformer_model, 'adaln_single'):
            self.embedder_modules.append(transformer_model.adaln_single)
        if hasattr(transformer_model, 'audio_adaln_single'):
            self.embedder_modules.append(transformer_model.audio_adaln_single)

        if hasattr(transformer_model, 'caption_projection'):
            self.embedder_modules.append(transformer_model.caption_projection)
        if hasattr(transformer_model, 'audio_caption_projection'):
            self.embedder_modules.append(transformer_model.audio_caption_projection)

        if hasattr(transformer_model, 'audio_scale_shift_table'):
            self.embedder_modules.append(transformer_model.audio_scale_shift_table)
        if hasattr(transformer_model, 'scale_shift_table'):
            self.embedder_modules.append(transformer_model.scale_shift_table)

        if hasattr(transformer_model, 'av_ca_video_scale_shift_adaln_single'):
            self.embedder_modules.append(transformer_model.av_ca_video_scale_shift_adaln_single)
        if hasattr(transformer_model, 'av_ca_audio_scale_shift_adaln_single'):
            self.embedder_modules.append(transformer_model.av_ca_audio_scale_shift_adaln_single)

        if hasattr(transformer_model, 'av_ca_a2v_gate_adaln_single'):
            self.embedder_modules.append(transformer_model.av_ca_a2v_gate_adaln_single)
        if hasattr(transformer_model, 'av_ca_v2a_gate_adaln_single'):
            self.embedder_modules.append(transformer_model.av_ca_v2a_gate_adaln_single)

        if hasattr(transformer_model, 'cross_attn_rope'):
            self.embedder_modules.append(transformer_model.cross_attn_rope)
        if hasattr(transformer_model, 'cross_attn_audio_rope'):
            self.embedder_modules.append(transformer_model.cross_attn_audio_rope)      

        if hasattr(transformer_model, 'scale_shift_table'):
            self.output_modules.append(transformer_model.scale_shift_table)
        if hasattr(transformer_model, 'audio_scale_shift_table'):
            self.output_modules.append(transformer_model.audio_scale_shift_table) 

        if hasattr(transformer_model, 'norm_out'):
            self.output_modules.append(transformer_model.norm_out)  
        if hasattr(transformer_model, 'proj_out'):
            self.output_modules.append(transformer_model.proj_out)
        
        if hasattr(transformer_model, 'audio_norm_out'):
            self.output_modules.append(transformer_model.audio_norm_out)  
        if hasattr(transformer_model, 'audio_proj_out'):
            self.output_modules.append(transformer_model.audio_proj_out)
    
    def _initialize_embedder_output_modules(self):
        """初始化embedder和output模块，将它们移到GPU"""
        #print(f"[GPU Manager] 初始化: 将 {len(self.embedder_modules)} 个embedder模块和 {len(self.output_modules)} 个output模块移到GPU")
        
        for module in self.embedder_modules:
            if hasattr(module, 'to'):
                module.to(self.device)
        
        for module in self.output_modules:
            if hasattr(module, 'to'):
                module.to(self.device)
        
        return self

    def unload_all_blocks_to_cpu(self):
        """卸载所有block到CPU"""
        print(f"[GPU Manager] 卸载所有block到CPU")
        
        # 将所有模块移到CPU
        for i, module in enumerate(self.managed_modules):
            if hasattr(module, 'to'):
                module.to('cpu')
        
        for module in self.embedder_modules:
            if hasattr(module, 'to'):
                module.to('cpu')
        
        for module in self.output_modules:
            if hasattr(module, 'to'):
                module.to('cpu')
        
        # 清空GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return self

    def unload_blocks_to_cpu(self):
        """卸载所有block到CPU"""
        print(f"[GPU Manager] 卸载所有block到CPU")
        
        # 将最后一个模块移到CPU
        for i, module in enumerate(self.managed_modules):
            if i==len(self.managed_modules)-self.block_group_size:
                if hasattr(module, 'to'):
                    module.to('cpu')

        # 清空GPU缓存
        if torch.cuda.is_available():
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
                caption_channels=caption_channels,
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

        # Process transformer blocks
        for block_index,block in enumerate(self.transformer_blocks):
            if gpu_manager is not None:
                # 加载当前block到GPU
                if block_index < len(gpu_manager.managed_modules):
                    module = gpu_manager.managed_modules[block_index]
                    if hasattr(module, 'to'):
                        module.to(gpu_manager.device)
                        #print(f"[GPU Manager] 加载dual block {block_index}到GPU")
                
                # 卸载上一个block（如果是第一个block，则不需要卸载）
                if block_index > 0 and (block_index - 1) < len(gpu_manager.managed_modules):
                    prev_module = gpu_manager.managed_modules[block_index - 1]
                    if hasattr(prev_module, 'to'):
                        prev_module.to('cpu')
                        #print(f"[GPU Manager] 卸载dual block {block_index-1}到CPU")
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
