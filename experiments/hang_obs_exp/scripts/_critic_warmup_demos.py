"""Demo-based critic warmup for PPO+BC.

Vanilla PPO+BC fails because the critic is randomly initialized and the
PG signal `A = Q - V` is just noise until V catches up — by which point
the actor has been pushed off the BC manifold. The fix is to seed the
critic with a value function fit to the BC demos themselves: for each
demo state, the target is the Monte Carlo discounted return from that
state to the end of the episode under the demo policy.

After this pretrain step:
- V(s) outputs sensible values on BC-visited states
- VecNormalize.ret_rms is bootstrapped with demo running returns so
  reward normalization at PPO start is in the right ballpark
- PPO's first updates have non-noisy advantages and don't push the
  actor in arbitrary directions

This is the proper structural fix for "PPO erases BC" — a separate
optimizer state from PPO, training only the value head and critic-side
trunk (MlpExtractor.value_net) so the actor isn't touched.
"""

import numpy as np
import torch
import torch.nn.functional as F


def _compute_running_and_mc_returns(rewards, gamma):
    """Per-episode forward running returns and backward MC returns."""
    rewards = np.asarray(rewards, dtype=np.float32)
    n = len(rewards)
    running = np.zeros(n, dtype=np.float32)
    G = 0.0
    for t in range(n):
        G = G * gamma + rewards[t]
        running[t] = G
    mc = np.zeros(n, dtype=np.float32)
    G = 0.0
    for t in reversed(range(n)):
        G = rewards[t] + gamma * G
        mc[t] = G
    return running, mc


def critic_warmup_on_demos(agent, demo_obs, demo_rewards_per_ep,
                            vec_normalize, gamma=0.99,
                            epochs=50, batch_size=256, lr=3e-4,
                            use_wandb=False):
    """Pretrain V(s) on Monte Carlo returns from BC demos.

    Args:
        agent: SB3 PPO agent. Its policy has `mlp_extractor.value_net`
            (critic trunk) and `value_net` (critic head); this is what
            we update.
        demo_obs: (N, obs_dim) array of all demo states concatenated.
        demo_rewards_per_ep: list of (ep_len,) arrays of per-step raw
            rewards, in episode order. sum(ep_lens) must equal N.
        vec_normalize: the VecNormalize wrapper. obs_rms must already
            be populated (BC pretrain does this); we will additionally
            bootstrap ret_rms from demo running returns.
        gamma: discount factor; should match PPO's gamma.
        epochs / batch_size / lr: training hyperparams.

    Returns: dict with summary stats for logging.
    """
    if not demo_rewards_per_ep:
        raise ValueError(
            'critic_warmup_on_demos: demo_rewards_per_ep is empty. '
            'Per-step rewards are required — collect fresh demos via '
            '--bc_episodes or use a demo dir whose pkls contain the '
            "'rewards' field.")
    total_steps = sum(len(r) for r in demo_rewards_per_ep)
    if total_steps != len(demo_obs):
        raise ValueError(
            f'critic_warmup_on_demos: per-episode reward lengths sum '
            f'to {total_steps} but demo_obs has {len(demo_obs)} rows. '
            f'These must align (one reward per demo state).')

    # 1. Compute forward running returns + backward MC returns per episode.
    all_running = []
    all_mc = []
    for ep_rewards in demo_rewards_per_ep:
        running, mc = _compute_running_and_mc_returns(ep_rewards, gamma)
        all_running.append(running)
        all_mc.append(mc)
    all_running = np.concatenate(all_running, axis=0).astype(np.float32)
    all_mc = np.concatenate(all_mc, axis=0).astype(np.float32)

    # 2. Bootstrap VecNormalize.ret_rms with demo running returns so
    #    PPO's normalize_reward at startup uses demo-scale stats.
    vec_normalize.ret_rms.update(all_running)
    print(f'[critic-warmup] bootstrapped ret_rms from {len(all_running)} '
          f'demo steps: mean={float(vec_normalize.ret_rms.mean):.3f} '
          f'var={float(vec_normalize.ret_rms.var):.3f}')

    # 3. Normalize MC targets the same way VecNormalize.normalize_reward
    #    will normalize incoming rewards during PPO. Per-step normalize
    #    + clip first, then sum with gamma — matches what V would target
    #    if PPO trained on normalized rewards.
    norm_mc = []
    for ep_rewards in demo_rewards_per_ep:
        ep_norm = vec_normalize.normalize_reward(
            np.asarray(ep_rewards, dtype=np.float32))
        mc = np.zeros_like(ep_norm)
        G = 0.0
        for t in reversed(range(len(ep_norm))):
            G = ep_norm[t] + gamma * G
            mc[t] = G
        norm_mc.append(mc)
    norm_mc = np.concatenate(norm_mc, axis=0).astype(np.float32)

    # 4. Normalize obs (obs_rms already populated by BC pretrain).
    norm_obs = vec_normalize.normalize_obs(demo_obs).astype(np.float32)

    # 5. Train V(s) with MSE. Only update critic-side params:
    #    mlp_extractor.value_net (critic trunk) + value_net (head).
    #    feature_extractor for MLP is FlattenExtractor (no params), so
    #    the actor side is not touched.
    device = agent.device
    obs_t = torch.as_tensor(norm_obs, device=device)
    tgt_t = torch.as_tensor(norm_mc, device=device)

    policy = agent.policy
    value_params = (
        list(policy.mlp_extractor.value_net.parameters())
        + list(policy.value_net.parameters()))
    if not value_params:
        raise RuntimeError(
            'critic_warmup_on_demos: could not locate critic-side '
            'parameters (mlp_extractor.value_net + value_net). The '
            'policy structure may differ from the SB3 ActorCriticPolicy '
            'with split MlpExtractor branches that this routine assumes.')
    optimizer = torch.optim.Adam(value_params, lr=lr)
    n = len(obs_t)
    print(f'[critic-warmup] training V(s) on {n} demo states for {epochs} '
          f'epochs (lr={lr}, batch={batch_size}); target stats '
          f'mean={float(norm_mc.mean()):.3f} std={float(norm_mc.std()):.3f}')

    last_avg = float('nan')
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        count = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            o = obs_t[idx]
            tgt = tgt_t[idx]
            features = policy.extract_features(o)
            latent_vf = policy.mlp_extractor.forward_critic(features)
            v = policy.value_net(latent_vf).squeeze(-1)
            loss = F.mse_loss(v, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            count += len(idx)
        avg = total_loss / max(count, 1)
        last_avg = avg
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f'[critic-warmup] epoch {epoch+1}/{epochs}  mse={avg:.5f}')
        if use_wandb:
            import wandb
            wandb.log({'critic_warmup/mse': avg,
                       'critic_warmup/epoch': epoch + 1})

    return {
        'final_mse': last_avg,
        'target_mean': float(norm_mc.mean()),
        'target_std': float(norm_mc.std()),
        'ret_rms_mean': float(vec_normalize.ret_rms.mean),
        'ret_rms_var': float(vec_normalize.ret_rms.var),
        'n_states': int(n),
    }
