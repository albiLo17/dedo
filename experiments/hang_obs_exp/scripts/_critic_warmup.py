"""Critic warmup callbacks: freeze the actor for the first K updates so
the randomly-initialized value head can converge on early rollouts before
the actor moves. Prevents BC erasure — the standard failure mode of
"BC pretrain + on-policy / off-policy RL with random V or Q init."

Why it's needed: BC trains only the actor's mu head. The value/Q networks
start at Xavier init, so V(s) is random and advantages (or actor-loss
Q-values for SAC) are pure noise for the first 1-2 update calls. Lower
LR slows actor drift, but the first updates still push mu in noise
directions. Freezing the actor while the critic catches up is the
direct fix.

Two variants:
- `PPOCriticWarmupCallback`: counts rollouts (PPO is on-policy; one
  rollout = one PPO.train() call).
- `SACCriticWarmupCallback`: counts env steps (SAC is off-policy and
  trains every `train_freq` step after `learning_starts`).

Both attempt to freeze ALL actor params and the actor branch of the MLP
extractor. For pixel PPO with shared features_extractor (default for
ActorCriticPolicy) the CNN is *not* frozen — critic gradients still flow
through it and the CNN drifts mildly. That's an unavoidable trade-off
in shared-feature setups; for pure pixel BC preservation, run with
`policy_kwargs=dict(share_features_extractor=False)` (not exposed as a
flag here).
"""
import torch
from stable_baselines3.common.callbacks import BaseCallback


def _collect_actor_params(policy):
    """Return list of parameters that belong to the 'actor side' of an
    SB3 ActorCriticPolicy (PPO MlpPolicy / CnnPolicy / MultiInputPolicy).
    Specifically:
      - action_net (mu output head)
      - log_std (nn.Parameter for DiagGaussianDistribution)
      - mlp_extractor.policy_net (actor MLP trunk; separate from value_net)

    Does NOT include features_extractor (shared with critic by default
    for PPO; freezing it would also stop critic from learning visual
    features). For pure low-dim obs this is FlattenExtractor with no
    params anyway, so no concern.
    """
    params = []
    if hasattr(policy, 'action_net'):
        params.extend(policy.action_net.parameters())
    if hasattr(policy, 'log_std'):
        ls = policy.log_std
        if isinstance(ls, torch.nn.Parameter):
            params.append(ls)
    if hasattr(policy, 'mlp_extractor'):
        mlp = policy.mlp_extractor
        if hasattr(mlp, 'policy_net'):
            params.extend(mlp.policy_net.parameters())
    return params


class PPOCriticWarmupCallback(BaseCallback):
    """Freeze actor for the first `n_warmup_rollouts` PPO rollouts so
    the value head can converge on noisy initial returns BEFORE the
    actor moves. After unfreezing, normal PPO training resumes.

    With PPO defaults (`n_steps=4096`, `num_envs=4`), one rollout is
    16384 env steps. `n_warmup_rollouts=2` freezes the actor for the
    first ~32k env steps (= first 2 PPO.train() calls).

    Set `n_warmup_rollouts=0` to disable.
    """

    def __init__(self, n_warmup_rollouts: int = 2, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.n_warmup = int(n_warmup_rollouts)
        self._rollouts_seen = 0
        self._frozen = False
        self._frozen_params = []

    def _on_training_start(self) -> None:
        if self.n_warmup <= 0:
            return
        self._frozen_params = _collect_actor_params(self.model.policy)
        if not self._frozen_params:
            print('[critic_warmup] WARN: no actor params identified; '
                  'callback is a no-op (check policy structure)')
            return
        for p in self._frozen_params:
            p.requires_grad_(False)
        self._frozen = True
        if self.verbose > 0:
            print(f'[critic_warmup] PPO: froze '
                  f'{len(self._frozen_params)} actor tensors for first '
                  f'{self.n_warmup} rollouts (critic-only updates while '
                  f'V(s) stabilizes; protects BC pretrain)')

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # Fires after each rollout collection, BEFORE PPO.train() runs.
        # We want updates 1..n_warmup frozen, then unfreeze for n+1.
        # _rollouts_seen is incremented HERE, so when it equals n_warmup
        # we've completed the warmup quota; the next train() call should
        # be unfrozen.
        self._rollouts_seen += 1
        if self._frozen and self._rollouts_seen >= self.n_warmup:
            for p in self._frozen_params:
                p.requires_grad_(True)
            self._frozen = False
            if self.verbose > 0:
                print(f'[critic_warmup] PPO: unfroze actor after '
                      f'{self.n_warmup} rollouts '
                      f'(~{self.model.num_timesteps:,} env steps); '
                      f'PPO now updates actor + critic normally.')


class SACCriticWarmupCallback(BaseCallback):
    """For SAC: freeze actor (and its features extractor) for
    `n_warmup_env_steps` env steps after `learning_starts`. SAC's
    actor loss is `alpha*log_pi - min(Q1, Q2)(s, pi(s))`; until Q is
    informative the actor would be pushed in noise directions.

    SAC uses *separate* features extractors for actor and critic by
    default (`SACPolicy.share_features_extractor=False`), so freezing
    `policy.actor.parameters()` cleanly isolates the actor including
    its CNN (if pixels). The critic continues to learn from buffer
    samples normally.

    Default `n_warmup_env_steps=10000`: with `learning_starts=1000`,
    actor unfreezes at env step 11_000 — by then the critic has done
    ~10k gradient updates with `gradient_steps=1`, which is enough
    for Q to be roughly accurate on the BC-policy distribution.

    Set `n_warmup_env_steps=0` to disable.
    """

    def __init__(self, n_warmup_env_steps: int = 10000, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.n_warmup = int(n_warmup_env_steps)
        self._unfreeze_at = None
        self._frozen = False
        self._frozen_params = []

    def _on_training_start(self) -> None:
        if self.n_warmup <= 0:
            return
        actor = getattr(self.model.policy, 'actor', None)
        if actor is None:
            print('[critic_warmup] WARN: policy has no .actor attribute; '
                  'callback is a no-op (check SAC policy structure)')
            return
        self._frozen_params = list(actor.parameters())
        for p in self._frozen_params:
            p.requires_grad_(False)
        self._frozen = True
        learning_starts = int(getattr(self.model, 'learning_starts', 0))
        self._unfreeze_at = learning_starts + self.n_warmup
        if self.verbose > 0:
            print(f'[critic_warmup] SAC: froze '
                  f'{len(self._frozen_params)} actor tensors '
                  f'(critic-only updates); will unfreeze at env step '
                  f'{self._unfreeze_at:,} (learning_starts'
                  f'={learning_starts} + warmup={self.n_warmup})')

    def _on_step(self) -> bool:
        if self._frozen and self.model.num_timesteps >= self._unfreeze_at:
            for p in self._frozen_params:
                p.requires_grad_(True)
            self._frozen = False
            if self.verbose > 0:
                print(f'[critic_warmup] SAC: unfroze actor at env step '
                      f'{self.model.num_timesteps:,}; SAC now updates '
                      f'actor + critic normally.')
        return True
