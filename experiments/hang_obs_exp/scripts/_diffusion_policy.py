"""
Shared diffusion-policy modules for behavior-cloning across obs modalities.

Architecture (ported from the Push-T diffusion-policy demo notebook
`diffusion_policy_state_pusht_demo.py`):

  - Action chunking: predict `pred_horizon` future actions per inference,
    execute the middle `action_horizon` open-loop, then re-predict.
  - Conditioning: flatten an `obs_horizon`-long obs window through the
    modality-specific encoder to a single (B, obs_horizon*feat_dim) global
    conditioning vector. This vector is FiLM-modulated into a 1-D U-Net
    that predicts the noise added to the action chunk.
  - Diffusion: DDPM with `num_diffusion_iters` train steps and a squared-
    cos beta schedule. Same scheduler used at train and inference time.

Three obs encoders share a common forward signature:
  forward(obs: tensor) -> features: tensor   (per-frame: B*To -> B*To, feat_dim)

  StateObsEncoder       — identity / linear, for privileged low-dim state.
  RGBObsEncoder         — ResNet-18 with GroupNorm (BatchNorm is unsafe at
                          eval time when rollouts pass batch=1 frames).
  PointCloudObsEncoder  — SSG PointNet++ backbone shared with the SB3
                          pointnet2_extractor (auto-selects CUDA / pure-
                          Python). Works on (B*To, N_pts, 3) world coords.

Usage from training script:
  encoder = build_encoder(obs_mode, obs_spec)
  policy  = DiffusionPolicy(action_dim=6, obs_feat_dim=encoder.feat_dim,
                            obs_horizon=2, pred_horizon=16, action_horizon=8,
                            num_diffusion_iters=100)
  loss = policy.compute_loss(obs_seq, action_seq, encoder)
  action_chunk = policy.predict_action(obs_seq, encoder)
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Helpers
# =============================================================================

def _replace_bn_with_gn(module: nn.Module, num_groups: int = 16) -> nn.Module:
    """Replace every nn.BatchNorm{1,2,3}d in `module` with a GroupNorm.

    Why: at eval time we infer with batch_size=1 (single rollout); BN
    running-stats freeze in eval() mode but their running estimates can
    still mismatch the imagery our scripted demos render under (different
    camera config / lighting than ImageNet pretraining), and BN's per-
    layer assumptions are brittle. GroupNorm is invariant to batch size
    and standard in the diffusion-policy paper.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            num_features = child.num_features
            ng = min(num_groups, num_features)
            while num_features % ng != 0 and ng > 1:
                ng -= 1
            setattr(module, name, nn.GroupNorm(ng, num_features))
        else:
            _replace_bn_with_gn(child, num_groups=num_groups)
    return module


# =============================================================================
# Obs encoders (one per modality)
# =============================================================================

class StateObsEncoder(nn.Module):
    """Identity encoder for low-dim state obs. Feature dim = state dim."""

    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.feat_dim = state_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B*To, state_dim) — already in normalized [-1, 1] from dataset.
        return obs


class _GripProjection(nn.Module):
    """Small MLP that projects 12-dim gripper proprioception. Matches the
    convention in experiments/hang_obs_exp/scripts/pointnet2_extractor.py
    so PPO and diffusion-BC see the same grip-feature shape."""

    def __init__(self, grip_dim: int = 12, out_dim: int = 12):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(grip_dim, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, grip: torch.Tensor) -> torch.Tensor:
        return self.proj(grip)


class _GoalProjection(nn.Module):
    """Small MLP that projects the 3-dim hanger goal vector. RGB and PCD
    encoders concatenate this with their primary visual feature so the
    diffusion U-Net's global conditioning vector includes the same goal
    info the privileged state vector embeds — equal-footing comparison."""

    def __init__(self, goal_dim: int = 3, out_dim: int = 8):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(goal_dim, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        return self.proj(goal)


class RGBObsEncoder(nn.Module):
    """ResNet-18 (GroupNorm) on RGB + grip projection + goal projection.

    Input is a dict:
        {'image': (B*To, H, W, 3) uint8 OR (B*To, 3, H, W) float32,
         'grip':  (B*To, 12) float32 in roughly [-1, 1] (/WBOX-normalized
                  at collection time),
         'goal':  (B*To,  3) float32 in roughly [-1, 1] (/WBOX-normalized
                  hanger pose; matches the last 3 dims of the privileged
                  state vector — given to RGB/PCD so all three modalities
                  see identical goal info)}

    Output: (B*To, 512 + grip_out_dim + goal_out_dim).
    """

    def __init__(self, grip_dim: int = 12, grip_out_dim: int = 12,
                 goal_dim: int = 3, goal_out_dim: int = 8,
                 pretrained: bool = False):
        super().__init__()
        try:
            from torchvision.models import resnet18
        except ImportError as e:
            raise ImportError(
                'torchvision is required for RGBObsEncoder. '
                'Install with: pip install torchvision==0.19.1') from e

        # pretrained=True loads ImageNet weights so the conv stack starts
        # with general-purpose visual features. The BN->GN swap below then
        # re-initializes the normalization layers (BN's gamma/beta/running
        # stats don't transfer to GN), but the conv weights — which carry
        # the bulk of the transferable signal — are preserved. This matches
        # the standard recipe in Chi et al.'s diffusion-policy paper.
        if pretrained:
            backbone = resnet18(weights='IMAGENET1K_V1')
        else:
            backbone = resnet18(weights=None)
        backbone = _replace_bn_with_gn(backbone, num_groups=16)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.grip_proj = _GripProjection(grip_dim, grip_out_dim)
        self.goal_proj = _GoalProjection(goal_dim, goal_out_dim)
        self.grip_dim = grip_dim
        self.goal_dim = goal_dim
        self.feat_dim = 512 + grip_out_dim + goal_out_dim

    def forward(self, obs) -> torch.Tensor:
        # `obs` may be a dict {'image', 'grip', 'goal'} or, for backward
        # compat with single-tensor callers, just an image (grip and goal
        # are then treated as zeros — used by older deploy paths).
        if isinstance(obs, dict):
            image = obs['image']
            grip = obs['grip']
            goal = obs.get('goal')
            if goal is None:
                goal = torch.zeros(image.shape[0], self.goal_dim,
                                   device=image.device, dtype=torch.float32)
        else:
            image = obs
            grip = torch.zeros(image.shape[0], self.grip_dim,
                               device=image.device, dtype=torch.float32)
            goal = torch.zeros(image.shape[0], self.goal_dim,
                               device=image.device, dtype=torch.float32)
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        if image.ndim == 4 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2).contiguous()
        img_feat = self.backbone(image)
        grip_feat = self.grip_proj(grip.float())
        goal_feat = self.goal_proj(goal.float())
        return torch.cat([img_feat, grip_feat, goal_feat], dim=-1)


class PointCloudObsEncoder(nn.Module):
    """SSG PointNet++ (CUDA or pure-Python backend) + grip + goal projection.

    Input is a dict:
        {'pcd':  (B*To, n_points, 3) float32 (normalized to ~unit cube
                 by the dataset normalizer; cloth-only post seg-mask
                 filtering at collection time),
         'grip': (B*To, 12) float32 in roughly [-1, 1] (/WBOX-normalized),
         'goal': (B*To,  3) float32 in roughly [-1, 1] (/WBOX-normalized
                 hanger pose — matches the privileged state vector's
                 last 3 dims so all modalities see equivalent goal info)}

    Output: (B*To, pcd_feat_dim + grip_out_dim + goal_out_dim).
    """

    def __init__(self, n_points: int = 512, pcd_feat_dim: int = 256,
                 grip_dim: int = 12, grip_out_dim: int = 12,
                 goal_dim: int = 3, goal_out_dim: int = 8):
        super().__init__()
        import os
        force_pure = os.environ.get(
            'POINTNET2_FORCE_PURE_PYTHON', '0') == '1'
        has_cuda = False
        if not force_pure:
            try:
                from pointnet2_ops.pointnet2_modules import PointnetSAModule  # noqa: F401
                has_cuda = True
            except Exception:
                has_cuda = False
        self._has_cuda = has_cuda

        # Ball-query radii tuned for our HangProcCloth scene scale.
        #
        # Collection-time seg-mask filtering keeps ONLY cloth points in
        # the PCD; the peg/pole/flag/base are dropped. ObsNormalizer
        # then mean-centers and divides by max(half-range over xyz),
        # putting normalized cloth points roughly in [-1, 1]^3.
        #
        # With cloth-only points, the global normalization scale is
        # dominated by the *envelope of cloth motion across the demo
        # pool* (cloth moves through ~3 m vertically + ~2-3 m laterally
        # as the gripper threads it). A single cloth at any one frame
        # occupies roughly 0.3-0.5 of each normalized axis, with the
        # hole loop spanning ~0.10-0.15.
        #
        # Canonical PointNet++ SSG radii (0.2 / 0.4) are designed for
        # unit-cube-filling objects (e.g. ShapeNet). On cloth-only data
        # the cloth fills ~half the normalized space, so 0.1 / 0.3
        # gives layer 1 local-cloth resolution (hole-edge curvature,
        # wrinkles), layer 2 cloth-wide regional structure (cloth pose),
        # and layer 3 group_all captures the global shape.
        _R1, _R2 = 0.1, 0.3

        if has_cuda:
            from pointnet2_ops.pointnet2_modules import PointnetSAModule
            self.sa1 = PointnetSAModule(
                npoint=128, radius=_R1, nsample=32,
                mlp=[0, 64, 64, 128], use_xyz=True, bn=True)
            self.sa2 = PointnetSAModule(
                npoint=32, radius=_R2, nsample=32,
                mlp=[128, 128, 128, 256], use_xyz=True, bn=True)
            self.sa3 = PointnetSAModule(
                npoint=None, radius=None, nsample=None,
                mlp=[256, 256, 512, pcd_feat_dim], use_xyz=True, bn=True)
        else:
            from third_party.pointnet2_utils import PointNetSetAbstraction
            self.sa1 = PointNetSetAbstraction(
                npoint=128, radius=_R1, nsample=32,
                in_channel=3, mlp=[64, 64, 128], group_all=False)
            self.sa2 = PointNetSetAbstraction(
                npoint=32, radius=_R2, nsample=32,
                in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
            self.sa3 = PointNetSetAbstraction(
                npoint=None, radius=None, nsample=None,
                in_channel=256 + 3, mlp=[256, 512, pcd_feat_dim], group_all=True)

        # Swap BatchNorm2d → GroupNorm in the SA modules so eval-time
        # batch=1 inference doesn't depend on running-mean stats that
        # were estimated under training-time batch and may be stale
        # under procgen-cloth distribution shift. Idempotent: if the
        # backend doesn't expose nn.BatchNorm submodules (e.g. CUDA
        # ops bake BN into compiled C++), the recursion is a no-op.
        _replace_bn_with_gn(self.sa1, num_groups=8)
        _replace_bn_with_gn(self.sa2, num_groups=8)
        _replace_bn_with_gn(self.sa3, num_groups=8)

        self.grip_proj = _GripProjection(grip_dim, grip_out_dim)
        self.goal_proj = _GoalProjection(goal_dim, goal_out_dim)
        self.n_points = n_points
        self.grip_dim = grip_dim
        self.goal_dim = goal_dim
        self.feat_dim = pcd_feat_dim + grip_out_dim + goal_out_dim

    def forward(self, obs) -> torch.Tensor:
        if isinstance(obs, dict):
            pcd = obs['pcd']
            grip = obs['grip'].float()
            goal = obs.get('goal')
            if goal is None:
                goal = torch.zeros(pcd.shape[0], self.goal_dim,
                                   device=pcd.device, dtype=torch.float32)
        else:
            # Backward compat: single-tensor input is pcd alone; grip
            # and goal zeroed.
            pcd = obs
            grip = torch.zeros(pcd.shape[0], self.grip_dim,
                               device=pcd.device, dtype=torch.float32)
            goal = torch.zeros(pcd.shape[0], self.goal_dim,
                               device=pcd.device, dtype=torch.float32)
        pcd = pcd.float()

        if self._has_cuda:
            xyz = pcd.contiguous()
            l1_xyz, l1_feat = self.sa1(xyz, None)
            l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)
            _, l3_feat = self.sa3(l2_xyz, l2_feat)
        else:
            xyz = pcd.permute(0, 2, 1).contiguous()
            l1_xyz, l1_feat = self.sa1(xyz, None)
            l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)
            _, l3_feat = self.sa3(l2_xyz, l2_feat)
        pcd_feat = l3_feat.squeeze(-1)
        grip_feat = self.grip_proj(grip)
        goal_feat = self.goal_proj(goal.float())
        return torch.cat([pcd_feat, grip_feat, goal_feat], dim=-1)


def build_encoder(obs_mode: str, obs_kwargs: dict) -> nn.Module:
    """Factory: dispatch on obs_mode string used by the training script."""
    if obs_mode == 'state':
        return StateObsEncoder(state_dim=obs_kwargs['state_dim'])
    if obs_mode == 'rgb':
        return RGBObsEncoder(
            pretrained=obs_kwargs.get('pretrained', False))
    if obs_mode == 'pcd':
        return PointCloudObsEncoder(
            n_points=obs_kwargs.get('n_points', 512),
            pcd_feat_dim=obs_kwargs.get('feat_dim', 256))
    raise ValueError(f'unknown obs_mode {obs_mode!r}')


# =============================================================================
# ConditionalUnet1D (verbatim from pusht diffusion-policy demo)
# =============================================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(self, in_ch, out_ch, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size,
                      padding=kernel_size // 2),
            nn.GroupNorm(min(n_groups, out_ch), out_ch),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Two Conv1dBlocks with FiLM conditioning between them and a residual."""

    def __init__(self, in_ch, out_ch, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_ch, out_ch, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_ch, out_ch, kernel_size, n_groups=n_groups),
        ])
        self.out_ch = out_ch
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_ch * 2),
            nn.Unflatten(-1, (-1, 1)),
        )
        self.residual_conv = (nn.Conv1d(in_ch, out_ch, 1)
                              if in_ch != out_ch else nn.Identity())

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).reshape(
            cond.shape[0], 2, self.out_ch, 1)
        scale, bias = embed[:, 0], embed[:, 1]
        out = scale * out + bias
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """1-D U-Net that predicts action noise given a noisy action chunk,
    diffusion step k, and a global conditioning vector."""

    def __init__(self, input_dim, global_cond_dim,
                 diffusion_step_embed_dim=256,
                 down_dims=(256, 512, 1024),
                 kernel_size=5, n_groups=8):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim,
                                       kernel_size, n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim,
                                       kernel_size, n_groups),
        ])
        self.down_modules = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(in_out):
            is_last = i >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(d_in, d_out, cond_dim,
                                           kernel_size, n_groups),
                ConditionalResidualBlock1D(d_out, d_out, cond_dim,
                                           kernel_size, n_groups),
                Downsample1d(d_out) if not is_last else nn.Identity(),
            ]))
        self.up_modules = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(reversed(in_out[1:])):
            is_last = i >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(d_out * 2, d_in, cond_dim,
                                           kernel_size, n_groups),
                ConditionalResidualBlock1D(d_in, d_in, cond_dim,
                                           kernel_size, n_groups),
                Upsample1d(d_in) if not is_last else nn.Identity(),
            ]))
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(self, sample, timestep, global_cond):
        # sample: (B, T, action_dim) -> (B, action_dim, T)
        x = sample.moveaxis(-1, -2)
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long,
                                     device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        gfeat = self.diffusion_step_encoder(timesteps)
        gfeat = torch.cat([gfeat, global_cond], dim=-1)
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, gfeat)
            x = resnet2(x, gfeat)
            h.append(x)
            x = downsample(x)
        for mid in self.mid_modules:
            x = mid(x, gfeat)
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, gfeat)
            x = resnet2(x, gfeat)
            x = upsample(x)
        x = self.final_conv(x)
        return x.moveaxis(-1, -2)


# =============================================================================
# DiffusionPolicy — wraps encoder + noise predictor + DDPM scheduler.
# =============================================================================

class DiffusionPolicy(nn.Module):
    """Train: compute_loss. Inference: predict_action.

    Action conventions:
      - Actions are normalized to [-1, 1] at the dataset level (the
        scripted-demo collector already writes them this way after
        dividing by MAX_ACT_VEL).
      - Predicted actions are in the same [-1, 1] space; pass directly to
        env.step().

    Obs conventions:
      - The encoder takes a (B*To, *obs_shape) tensor and returns
        (B*To, feat_dim). The policy reshapes to (B, To*feat_dim) for the
        global cond.
    """

    def __init__(self, action_dim: int, obs_feat_dim: int,
                 obs_horizon: int = 2, pred_horizon: int = 16,
                 action_horizon: int = 8,
                 num_diffusion_iters: int = 100,
                 down_dims=(256, 512, 1024)):
        super().__init__()
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        self.action_dim = action_dim
        self.obs_feat_dim = obs_feat_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.num_diffusion_iters = num_diffusion_iters

        global_cond_dim = obs_horizon * obs_feat_dim
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            down_dims=down_dims)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon')

    # ------------------------------------------------------------------
    # Helpers — handle both tensor and dict obs.
    # ------------------------------------------------------------------
    @staticmethod
    def _obs_batch_horizon(obs):
        """Pull (B, To) shape from either a tensor or a dict-of-tensors."""
        if isinstance(obs, dict):
            t = next(iter(obs.values()))
        else:
            t = obs
        return t.shape[0], t.shape[1]

    @staticmethod
    def _flatten_obs(obs, B, To):
        """Collapse (B, To, *) → (B*To, *) for either tensor or dict obs."""
        if isinstance(obs, dict):
            return {k: v.reshape(B * To, *v.shape[2:]) for k, v in obs.items()}
        return obs.reshape(B * To, *obs.shape[2:])

    @staticmethod
    def _obs_device(obs):
        if isinstance(obs, dict):
            return next(iter(obs.values())).device
        return obs.device

    # ------------------------------------------------------------------
    # Training: predict noise epsilon, MSE against true noise.
    # ------------------------------------------------------------------
    def compute_loss(self, obs_seq,
                     action_seq: torch.Tensor,
                     encoder: nn.Module) -> torch.Tensor:
        """obs_seq:    tensor (B, To, *obs_shape) OR dict of such tensors.
           action_seq: (B, pred_horizon, action_dim) normalized in [-1, 1].
        """
        B, To = self._obs_batch_horizon(obs_seq)
        obs_flat = self._flatten_obs(obs_seq, B, To)
        obs_feat = encoder(obs_flat).reshape(B, To * self.obs_feat_dim)
        noise = torch.randn_like(action_seq)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=action_seq.device).long()
        noisy_actions = self.noise_scheduler.add_noise(
            action_seq, noise, timesteps)
        noise_pred = self.noise_pred_net(
            noisy_actions, timesteps, global_cond=obs_feat)
        return F.mse_loss(noise_pred, noise)

    # ------------------------------------------------------------------
    # Inference: iteratively denoise a Gaussian sample.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_action(self, obs_seq,
                       encoder: nn.Module,
                       num_inference_steps: Optional[int] = None
                       ) -> torch.Tensor:
        """obs_seq: tensor (B, To, *obs_shape) OR dict of such tensors.
        Returns (B, pred_horizon, action_dim) normalized in [-1, 1]."""
        B, To = self._obs_batch_horizon(obs_seq)
        obs_flat = self._flatten_obs(obs_seq, B, To)
        obs_feat = encoder(obs_flat).reshape(B, To * self.obs_feat_dim)
        n_steps = num_inference_steps or self.num_diffusion_iters
        self.noise_scheduler.set_timesteps(n_steps)
        naction = torch.randn(
            (B, self.pred_horizon, self.action_dim),
            device=self._obs_device(obs_seq))
        for k in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                naction, k, global_cond=obs_feat)
            naction = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k,
                sample=naction).prev_sample
        return naction


# =============================================================================
# Per-mode obs normalizer — fitted on the demo dataset.
# =============================================================================

class ObsNormalizer:
    """Per-modality obs normalization, fitted once on the demo dataset.

    State (tensor in (N, D)):
        Per-dim min/max → [-1, 1].
    RGB (dict with 'image' uint8 (N,H,W,3) and 'grip' float32 (N,12)):
        Image: pass-through (encoder divides by 255).
        Grip:  pass-through (already /WBOX-normalized at collection time).
    PCD (dict with 'pcd' float32 (N, n_pts, 3) and 'grip' float32 (N, 12)):
        PCD:  subtract per-axis mean, uniform scale by max half-range so
              points roughly land in [-1, 1]^3 (preserves PN2 radii).
        Grip: pass-through.
    """

    def __init__(self, obs_mode: str):
        self.obs_mode = obs_mode
        self.stats: Dict[str, np.ndarray] = {}

    def fit(self, primary: np.ndarray) -> None:
        """Fit on the PRIMARY obs (state for state-mode, image for rgb,
        pcd for pcd). Grip stats are unused — grip is pass-through."""
        if self.obs_mode == 'state':
            flat = primary.reshape(-1, primary.shape[-1])
            self.stats['min'] = flat.min(axis=0).astype(np.float32)
            self.stats['max'] = flat.max(axis=0).astype(np.float32)
        elif self.obs_mode == 'rgb':
            pass  # encoder divides image by 255
        elif self.obs_mode == 'pcd':
            flat = primary.reshape(-1, 3)
            self.stats['mean'] = flat.mean(axis=0).astype(np.float32)
            half = (flat.max(axis=0) - flat.min(axis=0)) / 2.0
            self.stats['scale'] = np.array(float(half.max()),
                                           dtype=np.float32)
        else:
            raise ValueError(f'unknown obs_mode {self.obs_mode!r}')

    def apply(self, obs):
        """Apply to a SAMPLE — either an ndarray (state) or a dict (rgb/pcd).
        Returns the same structure with normalization applied."""
        if self.obs_mode == 'state':
            mn, mx = self.stats['min'], self.stats['max']
            span = np.maximum(mx - mn, 1e-6)
            return (((obs - mn) / span) * 2.0 - 1.0).astype(np.float32)
        if self.obs_mode == 'rgb':
            # `obs` is a dict {'image': ..., 'grip': ...}; both pass-through.
            return obs
        if self.obs_mode == 'pcd':
            mean = self.stats['mean']
            scale = float(self.stats['scale']) or 1.0
            pcd_norm = ((obs['pcd'] - mean) / scale).astype(np.float32)
            # Preserve all auxiliary inputs (grip, goal, …) — the
            # normalizer only acts on the primary PCD tensor.
            out = {'pcd': pcd_norm}
            for k, v in obs.items():
                if k != 'pcd':
                    out[k] = v
            return out
        raise AssertionError

    def state_dict(self) -> dict:
        return {'obs_mode': self.obs_mode,
                'stats': {k: np.asarray(v).tolist()
                          for k, v in self.stats.items()}}

    def load_state_dict(self, sd: dict) -> None:
        self.obs_mode = sd['obs_mode']
        self.stats = {k: np.asarray(v, dtype=np.float32)
                      for k, v in sd['stats'].items()}


# =============================================================================
# Action normalizer — actions are already in [-1, 1] from the demo
# collector (after dividing by MAX_ACT_VEL); we still expose a class for
# parity / future expansion (e.g. if someone collects unnormalized demos).
# =============================================================================

class ActionNormalizer:
    """No-op normalizer: actions are stored normalized to [-1, 1]."""

    def fit(self, acts: np.ndarray) -> None:
        # Sanity check; log if any demo went outside [-1, 1].
        if np.abs(acts).max() > 1.01:
            print(f'[ActionNormalizer] WARN: max |action| = '
                  f'{np.abs(acts).max():.3f} > 1; demos may not be '
                  f'pre-normalized. Diffusion policy assumes [-1, 1].')

    def apply(self, acts):
        return acts.astype(np.float32) if isinstance(acts, np.ndarray) else acts

    def unapply(self, acts):
        return acts
