"""BC anchor callback: replay BC (obs, action) pairs after each PPO
rollout to keep the actor near the BC-pretrained policy.

Vanilla PPO+BC tends to walk the actor off the BC manifold once the
critic starts emitting non-trivial advantages: BC gives a starting
point but the policy gradient has no built-in incentive to stay near
it. This callback adds that incentive by interleaving small MSE
updates against the saved demo dataset between PPO updates. The demos
are normalized once at init using the current VecNormalize stats; the
running stats drift slowly enough that re-normalizing every step is
unnecessary.

Lives alongside the PPO optimizer (separate Adam state on the same
policy parameters), so anchor steps and PPO steps cumulatively shape
the policy without either one being aware of the other.
"""

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback


class BCAnchorCallback(BaseCallback):
    def __init__(self, demo_obs, demo_acts, vec_normalize,
                 n_batches=4, batch_size=256, lr=1e-4, verbose=0):
        super().__init__(verbose)
        self._demo_obs_raw = np.asarray(demo_obs, dtype=np.float32)
        self._demo_acts = np.asarray(demo_acts, dtype=np.float32)
        self._vec_normalize = vec_normalize
        self.n_batches = int(n_batches)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self._optimizer = None
        self._obs_t = None
        self._act_t = None

    def _init_callback(self) -> None:
        if self.n_batches <= 0 or len(self._demo_obs_raw) == 0:
            return
        device = self.model.device
        norm_obs = self._vec_normalize.normalize_obs(
            self._demo_obs_raw).astype(np.float32)
        self._obs_t = torch.as_tensor(norm_obs, device=device)
        self._act_t = torch.as_tensor(self._demo_acts, device=device)
        self._optimizer = torch.optim.Adam(
            self.model.policy.parameters(), lr=self.lr)
        if self.verbose:
            print(f'[bc-anchor] {len(self._obs_t)} demo pairs, '
                  f'{self.n_batches} batches/rollout, '
                  f'batch={self.batch_size}, lr={self.lr}')

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self._optimizer is None:
            return
        policy = self.model.policy
        n = len(self._obs_t)
        last_loss = 0.0
        for _ in range(self.n_batches):
            idx = torch.randperm(n, device=self.model.device)[:self.batch_size]
            o = self._obs_t[idx]
            a = self._act_t[idx]
            features = policy.extract_features(o)
            latent_pi, _ = policy.mlp_extractor(features)
            mean_a = policy.action_net(latent_pi)
            loss = F.mse_loss(mean_a, a)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            last_loss = float(loss.item())
        if self.logger is not None:
            self.logger.record('train/bc_anchor_mse', last_loss)
