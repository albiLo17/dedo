# Coordinate Frames and Sim-to-Real Alignment

Reference document for the three coordinate frames used across the
HangProcCloth pipeline and the transforms between them.  Companion to
[diffusion_runs.md](diffusion_runs.md) and
[scripts/project_traj_to_camframe.py](scripts/project_traj_to_camframe.py).

---

## Frames at a glance

| Frame | Abbreviation | Units | Origin | Convention |
|-------|-------------|-------|--------|------------|
| Sim world | **S** | sim-metres (≈ 22× real) | PyBullet world origin | Z-up |
| Camera | **C** | sim-metres | Camera optical centre | +Z forward, +X right, +Y down (OpenCV) |
| Real world | **W** | real metres | Physical lab origin | determined by calibration |

Three key facts to keep in mind:

1. **The camera frame is in sim units**, not real metres.  A depth of
   14 in C corresponds to `14 × 0.045 = 0.63 m` from the real camera.
2. **The camera convention** (+Z forward, +X right, +Y down) is fixed
   regardless of which `cam_viewmat` was used.  The specific position
   and orientation of the C origin in S depends on the viewmat.
3. **The sim-to-real scale (0.045) applies to every conversion that
   crosses the S→W or C→W boundary**, including velocities.

---

## The three transforms

### S → C  (sim world to camera frame)

Derived from `cam_viewmat = [dist, pitch, yaw, tx, ty, tz]` via
`pybullet.computeViewMatrixFromYawPitchRoll` followed by an OpenGL→OpenCV
axis flip (`R_flip = diag([1, −1, −1])`):

```
p_C = R_SC @ p_S + t_SC          # positions
v_C = R_SC @ v_S                  # velocity / direction vectors (no t)
```

| Symbol | Meaning |
|--------|---------|
| `R_SC` | 3×3 rotation, sim world → camera frame |
| `t_SC` | 3-vector translation (sim units) — encodes distance from world origin to camera |

Inverse (C → S):

```
p_S = R_SC.T @ p_C − R_SC.T @ t_SC
    = R_CS   @ p_C + cam_orig_S
```

where `cam_orig_S = −R_SC.T @ t_SC` is the camera's position in sim world
and `R_CS = R_SC.T`.

`project_traj_to_camframe.py` computes `R_SC` and `t_SC` fresh from the
`cam_viewmat` stored in each pkl — it never hard-codes specific numbers,
so it works for any viewmat.

---

### S → W  (sim world to real world)

Calibrated rigid transform plus uniform scale:

```
p_W = R_z(θ) @ (scale × p_S) + offset     # positions
v_W = scale  × R_z(θ) @ v_S               # velocities — scale yes, offset no
```

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Uniform scale | `scale` | **0.045** |
| Rotation angle | `θ` | **90°** about Z |
| Rotation matrix | `R_z90` | `[[0,−1,0],[1,0,0],[0,0,1]]` |
| Offset (real metres) | `offset` | **[0.5, 0.0, −0.039]** |

In full matrix form:

```
p_W = R_z(90°) @ (p_sim * 0.045) + [0.5, 0.0, -0.039]
```

The offset is applied **after** rotation and scaling, and is in real metres.

Inverse (W → S):

```
p_S = (1/scale) × R_z90.T @ (p_W − offset)
```

---

### C → W  (camera frame to real world — the combined transform)

Substituting S→C into S→W:

```
p_W = R_z90 @ (scale × (R_CS @ p_C + cam_orig_S)) + offset
    = scale × (R_z90 @ R_CS) @ p_C  +  [R_z90 @ (scale × cam_orig_S) + offset]
    = scale × R_CW @ p_C  +  cam_orig_W
```

```
p_W = scale × R_CW @ p_C + cam_orig_W     # positions
v_W = scale × R_CW @ v_C                  # velocities
```

| Symbol | Meaning |
|--------|---------|
| `R_CW = R_z90 @ R_CS` | rotation, camera frame → real world |
| `cam_orig_W` | camera's physical position in real world (metres) |

Inverse (W → C):

```
p_C = (1/scale) × R_CW.T @ (p_W − cam_orig_W)
```

> **Key point:** `scale` still multiplies `p_C` when converting to real
> world because the camera frame lives in sim units.  This applies to
> PCD coordinates, grip/goal positions, and action velocities alike.

---

## What changes (and what doesn't) across conversions

| Conversion | Scale (0.045) | Rotation | Translation |
|------------|:---:|:---:|:---:|
| S → C positions | — | ✓ R_SC | ✓ t_SC |
| S → C velocities | — | ✓ R_SC | — |
| S → W positions | ✓ | ✓ R_z90 | ✓ offset |
| S → W velocities | ✓ | ✓ R_z90 | — |
| C → W positions | ✓ | ✓ R_CW | ✓ cam_orig_W |
| C → W velocities | ✓ | ✓ R_CW | — |
| C → S positions | — | ✓ R_CS | ✓ cam_orig_S |
| C → S velocities | — | ✓ R_CS | — |

### WBOX normalisation

`grip` and `goal` in demo pkls are stored as `sim_metres / WBOX`
(WBOX = 20).  In the `_camframe.pkl` they stay WBOX-normalised, just
rotated into camera frame.  To reach real-world metres from a
WBOX-normalised camera-frame value:

```
p_W = scale × WBOX × R_CW @ p_C_wbox + cam_orig_W
    =    0.9          × R_CW @ p_C_wbox + cam_orig_W
```

The combined factor `scale × WBOX = 0.045 × 20 = 0.9` is the single
multiplier from WBOX-normalised sim coords to real metres.

### PCD coordinates

`obs['pcd']` is stored in raw sim metres (not WBOX-normalised), in
whatever frame the pkl uses.  In a `_camframe.pkl`:

```
p_W = 0.045 × R_CW @ p_C_pcd + cam_orig_W
```

The scene target at `cam_viewmat` target distance of 14 sim units maps
to `14 × 0.045 = 0.63 m` from the real camera — a useful sanity check.

### Action velocities

Actions are stored as `v_sim / MAX_ACT_VEL` (dimensionless, in [−1, 1]).
In the `_camframe.pkl` they are rotated into camera frame but the
magnitude (normalisation) is unchanged.  To recover a real-world
velocity from a camera-frame normalised action:

```
v_W = MAX_ACT_VEL_real × R_CW @ act_C
where MAX_ACT_VEL_real = MAX_ACT_VEL_sim × scale = 4.0 × 0.045 = 0.18 m/s
```

---

## Specific values for cam_viewmat = [14, −5, 45, 0, 0, 5.5]

This is the camera used for all v3/v4 demo collection and evaluation.
The script reads `cam_viewmat` from each pkl, so these numbers are
for reference and verification only.

### Camera intrinsics

| Parameter | Value | Notes |
|-----------|-------|-------|
| FOV (vertical) | 60° | matches `proj_matrix(fov=60)` in `_bc_obs_helpers.py` |
| Resolution | 128 × 128 px | square; `cam_resolution` in pkl |
| Aspect | 1.0 | |
| Near / Far | 0.1 / 30.0 sim units | depth buffer clipping planes |
| fx = fy | **110.85 px** | `(res/2) / tan(fov/2)` |
| cx = cy | **64.0 px** | image centre |

K matrix:
```
K = [[110.85,     0,  64],
     [    0,  110.85,  64],
     [    0,      0,   1]]
```

### Camera extrinsics — sim world frame

| Property | Value |
|----------|-------|
| Camera position `cam_orig_S` | `[9.862, −9.862, 6.720]` sim units |
| R_SC (sim→cam) | see below |
| t_SC (sim→cam) | `[0, 5.479, 14.479]` sim units |

```
R_SC =  [[ 0.707107,  0.707107,  0.      ],
         [ 0.061628, -0.061628, -0.996195],
         [-0.704416,  0.704416, -0.087156]]

R_CS =  [[ 0.707107,  0.061628, -0.704416],
         [ 0.707107, -0.061628,  0.704416],
         [ 0.      , -0.996195, -0.087156]]
```

Camera axes in sim world (columns of R_CS):

| Cam axis | Direction in sim world |
|----------|----------------------|
| +X (right) | `[ 0.707,  0.707,  0.000]` — diagonal in sim XY |
| +Y (down) | `[ 0.062, −0.062, −0.996]` — mostly −Z_sim (downward) |
| +Z (forward) | `[−0.704,  0.704, −0.087]` — toward sim origin, slight downward |

### Camera extrinsics — real world frame

| Property | Value |
|----------|-------|
| Camera position `cam_orig_W` | `[0.944, 0.444, 0.263]` real metres |
| Quaternion wxyz | `[−0.2585, 0.2821, 0.6812, −0.6242]` |

```
R_CW = [[-0.707107,  0.061628, -0.704416],
        [ 0.707107,  0.061628, -0.704416],
        [ 0.      , -0.996195, -0.087156]]
```

Verification: `cam_viewmat` target `(0, 0, 5.5)` in sim maps to
`(0.5, 0, 0.209)` in real world, and to `(0, 0, 14)` in camera frame
(exactly 14 sim units along +Z, centred at (0,0) — correct).

---

## Sim-to-real alignment procedure

The goal is to mount a real RGBD camera so its depth observations match
the sim camera's PCD observations.

### 1. Rough placement

Mount the real camera at approximately:
- **Position:** `[0.944, 0.444, 0.263]` metres in the lab frame
- **Orientation:** pointing toward the workspace such that the peg / hanger
  is roughly centred and approximately 0.63 m from the lens

The 0.63 m figure comes from `dist × scale = 14 × 0.045`.  Pitch is −5°
(slightly downward) and yaw is 45° (diagonal) in the lab frame after
applying the 90° sim-to-real rotation.

### 2. FOV matching

Configure the real camera for **60° vertical FOV** (or apply a centre-crop
and resize to achieve equivalent FOV at 128×128).  A mismatched FOV
causes the correct scale/depth to be geometrically inconsistent —
objects will look right in position but wrong in apparent size.

### 3. Visual overlay in viser

Load a `_camframe.pkl` and display `obs['pcd']` with the viser camera at:

```python
position = [0, 0, 0]     # camera frame origin
look_at  = [0, 0, 1]     # +Z forward
up       = [0, -1, 0]    # +Y is down in camera frame
```

Side-by-side with the real camera's live depth stream projected through
K, adjust the camera mounting until the cloth/peg geometry overlaps.

To overlay the sim PCD on top of the world-frame scene in viser (when
displaying the original, non-camframe pkl alongside real data):

```python
# Place a viser frame at the sim camera's real-world pose
server.scene.add_frame(
    "/sim_cam",
    wxyz=np.array([-0.2585, 0.2821, 0.6812, -0.6242]),   # quat wxyz
    position=np.array([0.9438, 0.4438, 0.2634]),
)
# Attach camframe PCD as a child — it renders in the right world location
server.scene.add_point_cloud("/sim_cam/pcd", points=pcd_camframe[t], ...)
```

### 4. Scale verification

Pick a known landmark (e.g. the peg tip) visible in both the sim PCD and
the real depth image.

- Read its camera-frame position `p_C` from the PCD
- Compute real-world position: `p_W = 0.045 × R_CW @ p_C + cam_orig_W`
- Measure the real distance with a ruler or from the RGBD sensor

If the measured real distance disagrees with `p_W`, update `scale` in the
sim-to-real transform.  The offset and rotation are less sensitive —
scale errors are the most common source of systematic overlay drift.

### 5. Re-running the conversion script

After any update to the sim-to-real calibration:

```bash
conda run -n dedo python experiments/hang_obs_exp/scripts/project_traj_to_camframe.py \
    --pkl logs/hang_obs_exp/eval_trajs/.../traj_ep001_....pkl \
    --sim_to_real_scale    0.045 \
    --sim_to_real_offset   0.5 0.0 -0.039 \
    --sim_to_real_rotation_z_deg 90
```

Defaults in the script already match the calibrated values above, so the
flags can be omitted unless you are testing a revised calibration.

The output `_camframe.pkl` stores all transforms under:

| Key | Contents |
|-----|----------|
| `cam_extrinsics_sim` | `position_sim`, `R_cam_to_sim`, `quat_wxyz_sim` |
| `cam_extrinsics_real` | `position_real`, `R_cam_to_real`, `quat_wxyz_real` |
| `cam_intrinsics` | `fov_deg`, `fx_px`, `fy_px`, `cx_px`, `cy_px`, `K` |
| `sim_to_real` | `scale`, `offset_xyz`, `rotation_z_deg` |

---

## Quick-reference cheat sheet

```
Sim units → real metres:  multiply by 0.045
WBOX-normalised → real:   multiply by 0.9  (= 0.045 × 20)
Real metres → sim units:  divide by 0.045  (multiply by 22.2)

Camera depth (sim)  →  real distance from camera:  × 0.045
Action velocity     →  real m/s:                   × 0.045 × MAX_ACT_VEL
                                                  = × 0.045 × 4.0 = × 0.18

Frames at origin:
  S origin → C frame:  R_SC @ [0,0,0] + t_SC = [0, 5.48, 14.48]
  S origin → W frame:  R_z90 @ ([0,0,0]*0.045) + offset = [0.5, 0, -0.039]
  C origin → S frame:  cam_orig_S = [9.862, -9.862, 6.720]
  C origin → W frame:  cam_orig_W = [0.944, 0.444, 0.263]
```
