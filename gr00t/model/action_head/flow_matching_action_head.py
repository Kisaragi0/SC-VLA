# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from .cross_attention_dit import DiT, SelfAttentionTransformer

# =============================== Normalizer =============================== #
import json
from gr00t.model.action_head import rotation_utils

POS_SCALE = 50.0   
ROT_SCALE = 10.0    
GRIP_SCALE = 10.0    

dataset = 'maniskill'
# dataset = 'realbot'
delta_state_dim = 7
progress_dim = 1

# Path to the dataset statistics file.
# IMPORTANT: This must correspond to the SAME dataset root specified in `dataset_path`.
# For example, if dataset_path contains:
#   data/scvla_dataset/panda_wristcam
# then STATS_PATH should point to:
#   <dataset_root>/meta/stats.json
STATS_PATH = "<dataset_root>/meta/stats.json"

#  Create Normalizer
def load_state_normalizer(
        json_path=None,
        mode="min_max"
    ):

    with open(json_path, "r") as f:
        stats_full = json.load(f)

    # stats_full["observation.state"]  dict → { "mean":[...], "std":[...], ... }
    state_stats = stats_full["observation.state"]

    return rotation_utils.Normalizer(mode=mode, statistics=state_stats)

# Delta-state computation with de-/re-scaling
class DeltaStateProcessor:
    """
    A standalone processor for delta-state computation.
    """

    def __init__(self, dataset: str):
        self.dataset = dataset
        self.state_norm = load_state_normalizer(
            STATS_PATH,
            mode="min_max"
        )
    
    def scale_delta_state(self, delta_raw):
        """Apply per-component scaling to raw delta states."""
        delta_raw_scaled = torch.cat([
            delta_raw[..., 0:3] * POS_SCALE,
            delta_raw[..., 3:6] * ROT_SCALE,
            delta_raw[..., 6:7] * GRIP_SCALE
        ], dim=-1)
        return delta_raw_scaled

    def unscale_delta_state(self, delta_raw):
        """Inverse scaling for delta states."""
        delta_raw_scaled = torch.cat([
            delta_raw[..., 0:3] / POS_SCALE,
            delta_raw[..., 3:6] / ROT_SCALE,
            delta_raw[..., 6:7] / GRIP_SCALE
        ], dim=-1)
        return delta_raw_scaled

    @torch.no_grad()
    def compute(self, curr_state_norm, mid_state_norm):
        """
        Compute normalized delta states between two normalized states.
        """
        if self.dataset == "maniskill":
            curr_state = self.state_norm.inverse(curr_state_norm[..., :8])
            mid_state  = self.state_norm.inverse(mid_state_norm[..., :8])
            delta_raw = rotation_utils.compute_delta_state_quat2euler(curr_state, mid_state)
        elif self.dataset == "realbot":
            curr_state = self.state_norm.real_inverse(curr_state_norm[..., :7])
            mid_state  = self.state_norm.real_inverse(mid_state_norm[..., :7])
            delta_raw = rotation_utils.compute_delta_state_euler(curr_state, mid_state)

        delta_raw_scaled = self.scale_delta_state(delta_raw)
        
        return delta_raw_scaled

class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x

@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=4, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )

        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )

        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
 
        self.delta_state_encoder = MultiEmbodimentActionEncoder(
            action_dim=delta_state_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )

        self.delta_state_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=delta_state_dim,
        )       

        self.progress_encoder = MultiEmbodimentActionEncoder(
            action_dim=progress_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )

        self.progress_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=progress_dim,
        )

        self.align_norm = nn.LayerNorm(self.input_embedding_dim)
        self.future_state_proj = nn.Linear(self.input_embedding_dim, self.hidden_size)
        self.delta_processor = DeltaStateProcessor(dataset)

        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg)
            if config.use_vlln
            else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)

            self.delta_state_encoder.requires_grad_(False)
            self.delta_state_decoder.requires_grad_(False)
            self.progress_encoder.requires_grad_(False)
            self.progress_decoder.requires_grad_(False)

            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()

                self.delta_state_encoder.eval()
                self.delta_state_decoder.eval()
                self.progress_encoder.eval()
                self.progress_decoder.eval()

                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        # vl_embs = self.vl_qformer(backbone_output.backbone_features)
        device = vl_embs.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # # Embed state.
        states = action_input.state                                                                 
        curr_state, mid_state = states[:, 0:1], states[:, 1:2]
        curr_state_features = self.state_encoder(curr_state, embodiment_id)   

        delta_state = self.delta_processor.compute(curr_state, mid_state)

        delta_state_noise = torch.randn(delta_state.shape, device=delta_state.device, dtype=delta_state.dtype)  
        noisy_delta_state = (1 - t) * delta_state_noise + t * delta_state                          
        delta_state_velocity = delta_state - delta_state_noise                                          
        delta_state_features = self.delta_state_encoder(noisy_delta_state, t_discretized, embodiment_id)      

        progress = action_input.progress                                                                     
        progress_noise = torch.randn(progress.shape, device=progress.device, dtype=progress.dtype)           
        noisy_progress = (1 - t) * progress_noise + t * progress                                         
        progress_velocity = progress - progress_noise                                                   
        progress_features = self.progress_encoder(noisy_progress, t_discretized, embodiment_id)              

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((curr_state_features, future_tokens, progress_features, delta_state_features, action_features), dim=1)

        vl_attn_mask = backbone_output.backbone_attention_mask

        model_output, all_hidden = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=True,  # NOTE (YL): not using flare now
        )
        align_layer = all_hidden[12]
        align_layer = self.align_norm(align_layer)

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # 12th layer
        state_layer = self.future_state_proj(align_layer)
        pred_delta_state = self.delta_state_decoder(state_layer, embodiment_id)
        pred_progress = self.progress_decoder(state_layer, embodiment_id)

        # slice
        delta_state_idx_start = curr_state_features.shape[1] + future_tokens.shape[1] + progress_features.shape[1]
        delta_state_idx_end = delta_state_idx_start + delta_state_features.shape[1]
        pred_delta_state = pred_delta_state[:, delta_state_idx_start:delta_state_idx_end]

        progress_idx_start = curr_state_features.shape[1] + future_tokens.shape[1]
        progress_idx_end = progress_idx_start + progress_features.shape[1]
        pred_progress = pred_progress[:, progress_idx_start:progress_idx_end]       

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        delta_state_mask = action_input.state_mask[:, 1:2, :delta_state_dim]
        progress_mask = action_input.progress_mask
        
        # compute loss
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        delta_state_loss =  F.mse_loss(pred_delta_state, delta_state_velocity, reduction="none") * delta_state_mask
        progress_loss =  F.mse_loss(pred_progress, progress_velocity, reduction="none") * progress_mask

        action_loss = action_loss.sum() / action_mask.sum()
        delta_state_loss = delta_state_loss.sum() / delta_state_mask.sum()
        progress_loss = progress_loss.sum() / progress_mask.sum()

        loss = action_loss + 0.1 * delta_state_loss + 0.1 * progress_loss

        output_dict = {
            "loss": loss,
        }
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        # vl_embs = self.vl_qformer(backbone_output.backbone_features)
        embodiment_id = action_input.embodiment_id

        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device

        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        delta_state = torch.randn(
            size=(batch_size, 1, delta_state_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        progress = torch.randn(
            size=(batch_size, 1, progress_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )

            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            delta_state_features = self.delta_state_encoder(delta_state, timesteps_tensor, embodiment_id)
            progress_features = self.progress_encoder(progress, timesteps_tensor, embodiment_id)

            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            
            sa_embs = torch.cat((state_features, future_tokens, progress_features, delta_state_features, action_features), dim=1)

            # Run model forward.
            model_output, all_hidden = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
                return_all_hidden_states=True
            )
            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon :]

            # mid layer
            align_layer = all_hidden[12]
            align_layer = self.align_norm(align_layer)
            state_layer = self.future_state_proj(align_layer)

            pred_delta_state = self.delta_state_decoder(state_layer, embodiment_id)
            pred_progress = self.progress_decoder(state_layer, embodiment_id)

            # slice
            delta_state_idx_start = state_features.shape[1] + future_tokens.shape[1] + progress_features.shape[1]
            delta_state_idx_end = delta_state_idx_start + delta_state_features.shape[1]
            pred_delta_state_velocity = pred_delta_state[:, delta_state_idx_start:delta_state_idx_end]

            progress_idx_start = state_features.shape[1] + future_tokens.shape[1]
            progress_idx_end = progress_idx_start + progress_features.shape[1]
            pred_progress_velocity = pred_progress[:, progress_idx_start:progress_idx_end]                 

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
            delta_state = delta_state + dt * pred_delta_state_velocity
            progress = progress + dt * pred_progress_velocity

        # inverse
        delta_state_unscaled = self.delta_processor.unscale_delta_state(delta_state)

        # 2. Return as BatchFeature
        data = {
            "action_pred": actions,                  # actions
            "progress_pred": progress,               # progress
            "delta_state_pred": delta_state_unscaled # delta_state
        }
            
        return BatchFeature(data)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
