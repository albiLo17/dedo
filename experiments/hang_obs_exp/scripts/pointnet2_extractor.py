"""
SB3 PointNet++ features extractor with two backends:

  * 'cuda'        — erikwijmans/Pointnet2_PyTorch (CUDA C++ ops). 5–20×
                    faster FPS / ball query than pure Python. Used
                    automatically when `pointnet2_ops` is importable.
  * 'pure_python' — yanx27/Pointnet_Pointnet2_pytorch (vendored under
                    third_party/pointnet2_utils.py). Slower but no
                    CUDA / MSVC build needed. Fallback when the CUDA
                    package is unavailable.

Backend choice is automatic. To force pure Python (e.g. for debugging
or running on machines without the CUDA ops), set env var
POINTNET2_FORCE_PURE_PYTHON=1.

================================================================
Installing the CUDA ops (one-time, on Windows / Linux):
================================================================
Prerequisites:
  * torch with CUDA support installed (verify:
    `python -c "import torch; print(torch.cuda.is_available())"` → True)
  * CUDA toolkit matching torch's CUDA version
    (matches torch.version.cuda; if torch is cu118, install CUDA 11.8)
  * C++ compiler:
      - Linux: gcc/g++
      - Windows: Visual Studio Build Tools w/ "Desktop development
        with C++" workload, run from "x64 Native Tools Command Prompt
        for VS 2022" so cl.exe is on PATH

One-liner install (from any cmd prompt with CUDA + compiler available):

    pip install ninja
    pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#subdirectory=pointnet2_ops_lib"

Verify after install:

    python -c "from pointnet2_ops.pointnet2_modules import PointnetSAModule; print('OK')"

If install fails on Windows, the most common causes are:
  * Not in "x64 Native Tools Command Prompt for VS 2022" → cl.exe missing.
  * CUDA_HOME env var not set → install matching CUDA toolkit and add
    `set CUDA_HOME=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8`
  * torch is CPU-only → reinstall torch with +cuXXX build first.
================================================================
"""
import os
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ---------------------------------------------------------------------------
# Backend selection.
# ---------------------------------------------------------------------------
_FORCE_PURE = os.environ.get('POINTNET2_FORCE_PURE_PYTHON', '0') == '1'

_HAS_CUDA_OPS = False
if not _FORCE_PURE:
    try:
        from pointnet2_ops.pointnet2_modules import PointnetSAModule
        _HAS_CUDA_OPS = True
    except Exception as _e:
        _HAS_CUDA_OPS = False
        _import_err = _e

if _HAS_CUDA_OPS:
    print('[pointnet2] using CUDA ops backend (erikwijmans/Pointnet2_PyTorch)')
else:
    if _FORCE_PURE:
        print('[pointnet2] POINTNET2_FORCE_PURE_PYTHON=1 — using pure-Python '
              'backend (yanx27/Pointnet_Pointnet2_pytorch)')
    else:
        print('[pointnet2] CUDA ops not installed — falling back to '
              'pure-Python backend (yanx27). To enable the fast path:\n'
              '            pip install ninja && pip install '
              '"git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#'
              'subdirectory=pointnet2_ops_lib"\n'
              f'            (import error: {_import_err!r})')
    from third_party.pointnet2_utils import PointNetSetAbstraction


# ---------------------------------------------------------------------------
# Feature extractor — same external API across backends.
# ---------------------------------------------------------------------------
class PointNet2FeaturesExtractor(BaseFeaturesExtractor):
    """Splits flat obs into [grip | flat_pcd], runs SSG PointNet++ on the
    PCD reshaped to (B, n_points, 3), projects gripper, concatenates."""

    def __init__(self,
                 observation_space,
                 n_points: int = 512,
                 grip_dim: int = 12,
                 features_dim: int = 512,
                 sa1_npoint: int = 128, sa1_radius: float = 0.2,
                 sa1_nsample: int = 32,  sa1_mlp=(64, 64, 128),
                 sa2_npoint: int = 32,   sa2_radius: float = 0.4,
                 sa2_nsample: int = 32,  sa2_mlp=(128, 128, 256),
                 sa3_mlp=(256, 512)):
        super().__init__(observation_space, features_dim)

        expected = grip_dim + n_points * 3
        actual = int(observation_space.shape[0])
        assert actual == expected, (
            f'PointNet2FeaturesExtractor: obs dim {actual} != '
            f'grip_dim({grip_dim}) + 3*n_points({n_points}) = {expected}')

        self.n_points = n_points
        self.grip_dim = grip_dim
        pcd_feat_dim = features_dim - grip_dim
        sa3_mlp = list(sa3_mlp) + [pcd_feat_dim]
        sa1_mlp = list(sa1_mlp)
        sa2_mlp = list(sa2_mlp)

        if _HAS_CUDA_OPS:
            # erikwijmans's PointnetSAModule:
            #   mlp[0] = input feature channels (NOT including xyz; xyz is
            #   appended internally when use_xyz=True). For SA1 with no
            #   input features, that's 0.
            self.sa1 = PointnetSAModule(
                npoint=sa1_npoint, radius=sa1_radius, nsample=sa1_nsample,
                mlp=[0] + sa1_mlp, use_xyz=True, bn=True)
            self.sa2 = PointnetSAModule(
                npoint=sa2_npoint, radius=sa2_radius, nsample=sa2_nsample,
                mlp=[sa1_mlp[-1]] + sa2_mlp, use_xyz=True, bn=True)
            # Global pool: npoint/radius/nsample = None routes to GroupAll.
            self.sa3 = PointnetSAModule(
                npoint=None, radius=None, nsample=None,
                mlp=[sa2_mlp[-1]] + sa3_mlp, use_xyz=True, bn=True)
        else:
            # yanx27's PointNetSetAbstraction:
            #   in_channel = (extra feature channels) + 3 for relative xyz.
            self.sa1 = PointNetSetAbstraction(
                npoint=sa1_npoint, radius=sa1_radius, nsample=sa1_nsample,
                in_channel=3, mlp=sa1_mlp, group_all=False)
            self.sa2 = PointNetSetAbstraction(
                npoint=sa2_npoint, radius=sa2_radius, nsample=sa2_nsample,
                in_channel=sa1_mlp[-1] + 3, mlp=sa2_mlp, group_all=False)
            self.sa3 = PointNetSetAbstraction(
                npoint=None, radius=None, nsample=None,
                in_channel=sa2_mlp[-1] + 3, mlp=sa3_mlp, group_all=True)

        self.gripper_proj = nn.Sequential(
            nn.Linear(grip_dim, grip_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations):
        B = observations.shape[0]
        grip = observations[:, :self.grip_dim]
        pcd = observations[:, self.grip_dim:].view(B, self.n_points, 3)

        if _HAS_CUDA_OPS:
            # erikwijmans expects xyz: (B, N, 3), features: (B, C, N) or None.
            xyz = pcd.contiguous()
            l1_xyz, l1_feat = self.sa1(xyz, None)
            l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)
            _, l3_feat = self.sa3(l2_xyz, l2_feat)
            pcd_global = l3_feat.squeeze(-1)
        else:
            # yanx27 expects channels-first: xyz (B, 3, N), points (B, D, N).
            xyz = pcd.permute(0, 2, 1).contiguous()
            l1_xyz, l1_feat = self.sa1(xyz, None)
            l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)
            _, l3_feat = self.sa3(l2_xyz, l2_feat)
            pcd_global = l3_feat.squeeze(-1)

        grip_feat = self.gripper_proj(grip)
        return torch.cat([pcd_global, grip_feat], dim=-1)
