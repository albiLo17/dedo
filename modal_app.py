"""
Modal deployment for the HangProcCloth diffusion-BC pipeline.

Mirrors the server-side tmux workflow: stage 1 collects demos on CPU, stage 2
trains the three obs modes (state, rgb, pcd) in parallel on three GPUs.

------------------------------------------------------------------------------
One-time setup
------------------------------------------------------------------------------

  pip install modal
  modal token new
  modal secret create wandb WANDB_API_KEY=<your-key>

------------------------------------------------------------------------------
Usage (run from repo root)
------------------------------------------------------------------------------

  # First, validate the image builds and pybullet imports (~2 min, near-free).
  modal run modal_app.py::verify_env

  # Full pipeline: collect + 3 parallel trainings.
  modal run modal_app.py::main

  # Or run stages individually.
  modal run modal_app.py::main --stage collect
  modal run modal_app.py::main --stage train

  # Run a single training (useful for iterating).
  modal run modal_app.py::train_state

  # If you have demos already on the server / locally, upload them to the
  # volume instead of re-collecting:
  modal volume put hang-bc-data \\
      /local/path/bc_demos_15hz_pcd2048_n1000_randgoal_v1 \\
      bc_demos_15hz_pcd2048_n1000_randgoal_v1

  # Download trained checkpoints / videos / logs:
  modal volume get hang-bc-data diffusion_bc ./modal_results

------------------------------------------------------------------------------
Notes
------------------------------------------------------------------------------

- gym==0.21.0 (pinned in dedo/setup.py) forces Python 3.8 + legacy pip
  resolver. The image handles this.
- pybullet uses DIRECT mode (no GUI) and CPU rasterization for camera
  renders — no EGL/X11 setup needed.
- All persistent state (demos, training runs) lives in the `hang-bc-data`
  Modal Volume. It survives across runs.
- Three trainings spawn on three separate A10G GPUs concurrently. Modal
  bills per-second per-GPU.
- First image build takes 5-10 min. Subsequent runs reuse the cached image.
"""
import modal

APP_NAME = "hang-diffusion-bc"
VOLUME_NAME = "hang-bc-data"
DEMOS_SUBDIR = "bc_demos_randgoal_0.3_full"
LOGS_SUBDIR = "diffusion_bc"

app = modal.App(APP_NAME)

# -----------------------------------------------------------------------------
# Image
# -----------------------------------------------------------------------------
# Python 3.11. dedo's setup.py pins gym==0.21.0 which won't install cleanly
# on 3.10+ with modern build tooling — we pre-install pinned pip/setuptools/
# wheel that still understand gym 0.21's old-style setup.py
# (`pkg_resources.declare_namespace` was removed in setuptools 66+, and
# pip 24's resolver fails on its loose deps).
# apt deps cover pybullet's runtime needs (libGL for camera renderer,
# ffmpeg for moviepy/imageio video encoding).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg", "git")
    # Pin the build toolchain so the gym 0.21 install below works. These
    # are the last versions known to handle gym's old-style setup.py.
    .run_commands(
        "pip install --upgrade 'pip==23.0.1' 'setuptools==65.5.1' 'wheel==0.38.4'",
    )
    # Install gym 0.21 with --no-build-isolation so it uses the pinned
    # setuptools above instead of PEP 517 grabbing the latest.
    .run_commands(
        "pip install --no-build-isolation gym==0.21.0",
    )
    # IMPORTANT: do NOT upgrade pip past 24.0. pip 24.1+ rejects gym 0.21's
    # malformed metadata (`opencv-python (>=3.)`), which kills every
    # subsequent install. Stay on 23.0.1 for the rest of the image build.
    # Core stack. numpy<1.24 because gym 0.21 uses np.bool_/np.int_ which
    # were removed in 1.24.
    .pip_install(
        "numpy<1.24",
        "scipy",
        "matplotlib",
        "pybullet>=3.2.5",
        "stable_baselines3==1.2.0",
        "tensorboard",
        "tensorboardX",
        "moviepy",
        "wandb",
        "pyaml",
        "opencv-python",
        "importlib-metadata<5.0",
        "imageio",
        "imageio-ffmpeg",
    )
    # Diffusion-policy deps. huggingface_hub<0.26 keeps the `hf_cache_home`
    # symbol that diffusers 0.20.0 imports (removed in hf_hub 0.26+).
    # torchvision==0.16.2 pairs with torch 2.1.2 (required by RGBObsEncoder
    # which uses torchvision.models.resnet18).
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "huggingface_hub<0.26",
        "diffusers==0.20.0",
        "einops",
    )
    # Mount the repo. Ignored heavy/irrelevant paths so the bundle stays small.
    .add_local_dir(
        ".",
        remote_path="/root/dedo",
        ignore=[
            "logs/**",
            ".git/**",
            "**/__pycache__/**",
            "**/*.pyc",
            "rendered/**",
            "**/*.pkl",
            "**/*.mp4",
            "**/*.egg-info/**",
        ],
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb")

REPO_ROOT = "/root/dedo"
DATA_ROOT = "/data"
DEMOS_PATH = f"{DATA_ROOT}/{DEMOS_SUBDIR}"
LOGS_PATH = f"{DATA_ROOT}/{LOGS_SUBDIR}"


# -----------------------------------------------------------------------------
# Sanity-check: cheap, runs in ~2 min. Confirms image builds, deps import,
# pybullet + dedo load. Run this BEFORE the long pipeline.
# -----------------------------------------------------------------------------
@app.function(image=image, cpu=2, memory=4096, timeout=600)
def verify_env():
    import sys
    print("=== imports ===")
    import torch
    import gym
    import pybullet
    print(f"  torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
    print(f"  gym={gym.__version__}")
    print(f"  pybullet={getattr(pybullet, '__version__', 'n/a (no version attr)')}")

    print("\n=== diffusers ===")
    import diffusers
    from diffusers.training_utils import EMAModel  # noqa: F401
    from diffusers.optimization import get_scheduler  # noqa: F401
    print(f"  diffusers={diffusers.__version__}")

    print("\n=== dedo ===")
    sys.path.insert(0, REPO_ROOT)
    import dedo  # noqa: F401
    from dedo.envs.deform_env import DeformEnv  # noqa: F401
    print(f"  dedo OK (loaded from {REPO_ROOT})")

    print("\n=== ready ===")


# -----------------------------------------------------------------------------
# Stage 1: collect demos. CPU-only — pybullet sim is single-threaded.
# 4 CPUs gives some headroom for the camera/PCD math even though sim itself
# won't parallelize.
# -----------------------------------------------------------------------------
@app.function(
    image=image,
    cpu=4,
    memory=16384,
    volumes={DATA_ROOT: volume},
    timeout=6 * 3600,  # 75-90 min expected; buffer for slow attempts
)
def collect_demos():
    import os
    import subprocess
    os.makedirs(LOGS_PATH, exist_ok=True)
    cmd = [
        "python", "experiments/hang_obs_exp/scripts/collect_bc_demos.py",
        "--demos_dir", DEMOS_PATH,
        "--n_demos", "1000",
        "--cam_resolution", "128",
        "--pcd_n_points", "2048",
        "--max_act_vel", "4.0",
        "--success_metric", "legacy",
        "--success_factor", "1.2",
        "--ctrl_freq", "15",
        "--max_episode_len", "200",
        "--episode_tail_frames", "5",
        "--randomize_goal_radius", "1.5",
        "--debug_viz_first_n", "3",
        "--debug_viz_every", "100",
        "--debug_viz_first_n_failed", "5",
        "--seed", "2026",
    ]
    log_path = f"{LOGS_PATH}/collect.log"
    print(f"writing log to {log_path}")
    with open(log_path, "w") as logf:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True,
                       stdout=logf, stderr=subprocess.STDOUT)
    # Flush writes so subsequent train functions see the demos.
    volume.commit()
    print(f"DONE collect — demos at {DEMOS_PATH}, log at {log_path}")


# -----------------------------------------------------------------------------
# Stage 2: train. One function per obs_mode so they get separate GPUs and run
# in parallel.
# -----------------------------------------------------------------------------
_TRAIN_BASE_CMD = [
    "python", "experiments/hang_obs_exp/scripts/train_diffusion_bc.py",
    "--demo_path", DEMOS_PATH,
    "--action_horizon", "4",
    "--success_metric", "legacy",
    "--success_factor", "1.2",
    "--num_epochs", "300",
    "--lr", "1e-4",
    "--num_workers", "2",
    "--eval_every_epochs", "20",
    "--n_eval_episodes", "30",
    "--n_final_eval_episodes", "100",
    "--save_every_epochs", "20",
    "--use_wandb",
    "--wandb_project", "hang_bc_diffusion",
    "--seed", "2026",
    "--logdir_root", f"{LOGS_PATH}/runs",
]


def _run_train(extra_args, log_name):
    import os
    import subprocess
    os.makedirs(LOGS_PATH, exist_ok=True)
    cmd = _TRAIN_BASE_CMD + extra_args
    log_path = f"{LOGS_PATH}/{log_name}"
    print(f"writing log to {log_path}")
    print(f"cmd: {' '.join(cmd)}")
    with open(log_path, "w") as logf:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True,
                       stdout=logf, stderr=subprocess.STDOUT)
    volume.commit()
    print(f"DONE {log_name}")


@app.function(
    image=image, gpu="A10G", cpu=4, memory=32768,
    volumes={DATA_ROOT: volume}, secrets=[wandb_secret],
    timeout=24 * 3600,
)
def train_state():
    _run_train(
        ["--obs_mode", "state", "--state_key", "hole_centroid",
         "--batch_size", "256"],
        "state.log",
    )


@app.function(
    image=image, gpu="A10G", cpu=4, memory=32768,
    volumes={DATA_ROOT: volume}, secrets=[wandb_secret],
    timeout=24 * 3600,
)
def train_rgb():
    _run_train(
        ["--obs_mode", "rgb", "--pretrained_rgb", "--batch_size", "64"],
        "rgb.log",
    )


@app.function(
    image=image, gpu="A10G", cpu=4, memory=32768,
    volumes={DATA_ROOT: volume}, secrets=[wandb_secret],
    timeout=24 * 3600,
)
def train_pcd():
    _run_train(
        ["--obs_mode", "pcd", "--batch_size", "128"],
        "pcd.log",
    )


# -----------------------------------------------------------------------------
# Re-eval an existing checkpoint with --eval_only. Reuses the train command
# (so all the [data] parity logic runs and uses the demos in the volume),
# but skips the training loop and just runs the final-eval block.
#
# Usage:
#   modal run --detach modal_app.py::eval_state \\
#       --ckpt /data/diffusion_bc/runs/state/<run_dir>/policy_best.pt
#
#   modal run --detach modal_app.py::eval_rgb --ckpt /data/.../policy_best.pt
#   modal run --detach modal_app.py::eval_pcd --ckpt /data/.../policy_best.pt
#
# Discover the run_dir name with:
#   modal volume ls hang-bc-data diffusion_bc/runs/state
#
# Eval results (videos + log) land in a NEW timestamped subdir under
# diffusion_bc/runs/<obs_mode>/. The original training artifacts are
# untouched.
# -----------------------------------------------------------------------------
def _run_eval(extra_args, ckpt: str, log_name: str):
    import os
    import subprocess
    os.makedirs(LOGS_PATH, exist_ok=True)
    cmd = _TRAIN_BASE_CMD + extra_args + [
        "--eval_only",
        "--resume", ckpt,
    ]
    log_path = f"{LOGS_PATH}/{log_name}"
    print(f"writing log to {log_path}")
    print(f"cmd: {' '.join(cmd)}")
    with open(log_path, "w") as logf:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True,
                       stdout=logf, stderr=subprocess.STDOUT)
    volume.commit()
    print(f"DONE {log_name}")


@app.function(
    image=image, gpu="A10G", cpu=4, memory=32768,
    volumes={DATA_ROOT: volume}, secrets=[wandb_secret],
    timeout=2 * 3600,
)
def eval_state(ckpt: str):
    _run_eval(
        ["--obs_mode", "state", "--state_key", "hole_centroid",
         "--batch_size", "256"],
        ckpt,
        "eval_state.log",
    )


@app.function(
    image=image, gpu="A10G", cpu=4, memory=32768,
    volumes={DATA_ROOT: volume}, secrets=[wandb_secret],
    timeout=2 * 3600,
)
def eval_rgb(ckpt: str):
    _run_eval(
        ["--obs_mode", "rgb", "--pretrained_rgb", "--batch_size", "64"],
        ckpt,
        "eval_rgb.log",
    )


@app.function(
    image=image, gpu="A10G", cpu=4, memory=32768,
    volumes={DATA_ROOT: volume}, secrets=[wandb_secret],
    timeout=2 * 3600,
)
def eval_pcd(ckpt: str):
    _run_eval(
        ["--obs_mode", "pcd", "--batch_size", "128"],
        ckpt,
        "eval_pcd.log",
    )


# -----------------------------------------------------------------------------
# Re-eval EVERY saved checkpoint (policy_ep*.pt) in a run dir, to recreate
# the mid-training eval curve under the updated eval-time code. Logs to
# wandb keyed by epoch.
#
# Usage:
#   modal run --detach modal_app.py::eval_all_state \\
#       --run-dir /data/diffusion_bc/runs/state/<run_dir>
# -----------------------------------------------------------------------------
def _run_eval_all(extra_args, run_dir: str, log_name: str):
    import os
    import subprocess
    os.makedirs(LOGS_PATH, exist_ok=True)
    cmd = _TRAIN_BASE_CMD + extra_args + [
        "--eval_all_in", run_dir,
    ]
    log_path = f"{LOGS_PATH}/{log_name}"
    print(f"writing log to {log_path}")
    print(f"cmd: {' '.join(cmd)}")
    with open(log_path, "w") as logf:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True,
                       stdout=logf, stderr=subprocess.STDOUT)
    volume.commit()
    print(f"DONE {log_name}")


@app.function(
    image=image, gpu="A10G", cpu=4, memory=32768,
    volumes={DATA_ROOT: volume}, secrets=[wandb_secret],
    timeout=6 * 3600,  # ~15 checkpoints × ~15-20 min eval each
)
def eval_all_state(run_dir: str):
    _run_eval_all(
        ["--obs_mode", "state", "--state_key", "hole_centroid",
         "--batch_size", "256"],
        run_dir,
        "eval_all_state.log",
    )


@app.function(
    image=image, gpu="A10G", cpu=4, memory=32768,
    volumes={DATA_ROOT: volume}, secrets=[wandb_secret],
    timeout=6 * 3600,
)
def eval_all_rgb(run_dir: str):
    _run_eval_all(
        ["--obs_mode", "rgb", "--pretrained_rgb", "--batch_size", "64"],
        run_dir,
        "eval_all_rgb.log",
    )


@app.function(
    image=image, gpu="A10G", cpu=4, memory=32768,
    volumes={DATA_ROOT: volume}, secrets=[wandb_secret],
    timeout=6 * 3600,
)
def eval_all_pcd(run_dir: str):
    _run_eval_all(
        ["--obs_mode", "pcd", "--batch_size", "128"],
        run_dir,
        "eval_all_pcd.log",
    )


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
@app.local_entrypoint()
def main(stage: str = "all"):
    """stage: 'collect' | 'train' | 'all'."""
    if stage not in ("collect", "train", "all"):
        raise ValueError(f"stage must be 'collect', 'train', or 'all'; got {stage!r}")

    if stage in ("collect", "all"):
        print(">>> stage 1: collecting demos")
        collect_demos.remote()  # blocks until done

    if stage in ("train", "all"):
        print(">>> stage 2: launching 3 parallel trainings on separate GPUs")
        h_state = train_state.spawn()
        h_rgb = train_rgb.spawn()
        h_pcd = train_pcd.spawn()
        # Block until all three finish (or error).
        h_state.get()
        h_rgb.get()
        h_pcd.get()

    print(">>> ALL DONE")
