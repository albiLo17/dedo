"""Topology-aware mesh encoder for the diffusion-BC policy (`--obs_mode mesh`).

WHY THIS EXISTS, and what it deliberately reuses
------------------------------------------------
The other obs modes answer "can a policy act from what a camera sees?". This one
answers the question the project is actually about: *does the cloth's mesh state
— with its connectivity — buy anything over an unstructured point cloud?* A
PointNet++ on 6 channels would ignore edges and quietly concede that question, so
this encoder does message passing over the real topology.

It reuses the representation and batching convention of UniClothDiff's
`GPSStateEstModel` (`clothparticles/src/models/gps/state_est.py`) exactly:

    nodes       (B, max_V, 6)  padded  [position || rest_position]
    edge_index  (B, max_E, 2)  padded  bidirectional
    node_mask   (B, max_V)     bool

That is not incidental. It is what makes the estimated-mesh swap trivial later:
the state estimator emits meshes in precisely this format, so `--obs_mode mesh`
can be fed a GT mesh today and a reconstructed one tomorrow with no change here.
`rest_position` also doubles as the per-vertex identity token — a canonical
coordinate that says *which* material point this is, which a bare point cloud
cannot express.

It does NOT instantiate `GPSLayer` itself, though the local-message-passing and
masked-attention structure is modelled on it. `GPSLayer` is a *denoiser* layer:
it requires `adaln_params` from a diffusion timestep and cross-attends to a
point-cloud embedding. A policy encoder has neither, and neutralising them is not
free — `adaln_params = 0` sets `gate = 0`, which zeroes the residual updates and
turns the layer into a no-op. So the diffusion-specific machinery is dropped and
the graph machinery kept.

Live in the dedo env (python 3.8): `GPSLayer` is pure torch, but its module pulls
`src.registry` and `PointcloudEmbed` -> `torch_cluster`, which is not installed
here (and clothparticles is python 3.12). Hence a small local implementation
rather than a cross-env import.
"""
import numpy as np
import torch
import torch.nn as nn


def faces_to_bidir_edges(faces: np.ndarray) -> np.ndarray:
    """Triangle faces (F, 3) -> unique bidirectional edge list (E, 2) int64.

    Bidirectional because the message passing below sums over incoming edges
    only; without both directions a node never hears from half its neighbours.
    """
    e = set()
    for f in np.asarray(faces, dtype=np.int64):
        for k in range(3):
            a, b = int(f[k]), int(f[(k + 1) % 3])
            if a != b:
                e.add((a, b))
                e.add((b, a))
    if not e:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(sorted(e), dtype=np.int64)


class MeshLayer(nn.Module):
    """One round of masked message passing + masked global self-attention.

    Mirrors GPSLayer's graph half (gather src/dst, MLP on
    [src || dst || edge_feat], mask padded edges, scatter-add to dst, normalise
    by in-degree) and its padded self-attention, minus AdaLN and cross-attention.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim))
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_feat, edge_mask, node_mask):
        B, V, D = h.shape
        hn = self.norm1(h)

        # Padded edges point at index 0 after clamping; edge_mask zeroes their
        # messages, so the clamp cannot corrupt a real node.
        src = edge_index[..., 0].clamp(0, V - 1)
        dst = edge_index[..., 1].clamp(0, V - 1)
        h_src = torch.gather(hn, 1, src.unsqueeze(-1).expand(-1, -1, D))
        h_dst = torch.gather(hn, 1, dst.unsqueeze(-1).expand(-1, -1, D))
        msg = self.msg_mlp(torch.cat([h_src, h_dst, edge_feat], dim=-1))
        msg = msg * edge_mask.unsqueeze(-1).to(msg.dtype)

        agg = torch.zeros_like(h)
        agg.scatter_add_(1, dst.unsqueeze(-1).expand(-1, -1, D), msg)
        deg = torch.zeros(B, V, device=h.device, dtype=h.dtype)
        deg.scatter_add_(1, dst, edge_mask.to(h.dtype))
        h = h + agg / deg.clamp(min=1.0).unsqueeze(-1)

        hn = self.norm2(h)
        attn, _ = self.self_attn(hn, hn, hn, key_padding_mask=~node_mask)
        h = h + self.drop(attn)
        return h + self.ffn(self.norm3(h))


class MeshObsEncoder(nn.Module):
    """Mesh -> one feature vector, with the same aux inputs as the other modes.

    Input dict:
        'mesh'       (B*To, max_V, 6)  [pos || rest], normalized
        'edge_index' (B*To, max_E, 2)  int64, padded
        'node_mask'  (B*To, max_V)     bool
        'edge_mask'  (B*To, max_E)     bool
        'grip'       (B*To, 12)
        'goal'       (B*To, 3)

    Output: (B*To, mesh_feat_dim + grip_out_dim + goal_out_dim).

    `grip` and `goal` get the SAME projection treatment as in
    PointCloudObsEncoder/RGBObsEncoder. That is a fairness invariant, not a
    detail: every modality must see equivalent proprioception and goal
    information, or the comparison measures plumbing instead of representation.
    """

    def __init__(self, mesh_feat_dim: int = 256, hidden_dim: int = 128,
                 num_layers: int = 4, num_heads: int = 4, dropout: float = 0.0,
                 grip_dim: int = 12, grip_out_dim: int = 12,
                 goal_dim: int = 3, goal_out_dim: int = 8):
        super().__init__()
        self.node_embed = nn.Linear(6, hidden_dim)
        # Edge features are the relative offset between endpoints plus its
        # length — the same geometric quantities GPSStateEstModel builds in
        # _compute_edge_features, so a stretched edge is distinguishable from a
        # slack one regardless of where the cloth sits in the workspace.
        self.edge_embed = nn.Linear(4, hidden_dim)
        self.layers = nn.ModuleList([
            MeshLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim * 2, mesh_feat_dim)
        self.grip_proj = nn.Sequential(
            nn.Linear(grip_dim, grip_out_dim), nn.ReLU())
        self.goal_proj = nn.Sequential(
            nn.Linear(goal_dim, goal_out_dim), nn.ReLU())
        self.feat_dim = mesh_feat_dim + grip_out_dim + goal_out_dim

    def forward(self, obs):
        mesh = obs['mesh']
        edge_index = obs['edge_index'].long()
        node_mask = obs['node_mask'].bool()
        edge_mask = obs['edge_mask'].bool()

        pos = mesh[..., :3]
        V = mesh.shape[1]
        src = edge_index[..., 0].clamp(0, V - 1)
        dst = edge_index[..., 1].clamp(0, V - 1)
        p_src = torch.gather(pos, 1, src.unsqueeze(-1).expand(-1, -1, 3))
        p_dst = torch.gather(pos, 1, dst.unsqueeze(-1).expand(-1, -1, 3))
        rel = p_dst - p_src
        edge_feat = self.edge_embed(
            torch.cat([rel, rel.norm(dim=-1, keepdim=True)], dim=-1))

        h = self.node_embed(mesh)
        for layer in self.layers:
            h = layer(h, edge_index, edge_feat, edge_mask, node_mask)
        h = self.out_norm(h)

        # Masked mean+max pool. Padded slots must not dilute the mean (a cloth
        # with 219 real verts in 250 slots would otherwise be scaled by 0.88),
        # and max needs -inf rather than 0 or padding wins on negative features.
        m = node_mask.unsqueeze(-1).to(h.dtype)
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        mx = h.masked_fill(~node_mask.unsqueeze(-1), float('-inf')).max(1).values
        mx = torch.nan_to_num(mx, neginf=0.0)  # a fully-padded row cannot happen, but be safe
        feat = self.out_proj(torch.cat([mean, mx], dim=-1))

        return torch.cat([feat, self.grip_proj(obs['grip']),
                          self.goal_proj(obs['goal'])], dim=-1)
