import os
import sys

from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym

from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

# third-party
import numpy as np
import torch
import torch.nn as nn
import time

# ===== Modality CLI shim (must run BEFORE make_env) =====
def _pop_cli_arg(name, default=None):
    for i, a in enumerate(list(sys.argv)):
        if a == f"--{name}" and i + 1 < len(sys.argv):
            val = sys.argv[i + 1]
            del sys.argv[i:i+2]
            return val
    for i, a in enumerate(list(sys.argv)):
        prefix = f"--{name}="
        if a.startswith(prefix):
            val = a[len(prefix):]
            del sys.argv[i]
            return val
    return default

def apply_modality_class_defaults(modality: str):
    """
    Set class-level toggles on GO2RoughCfg BEFORE env creation.
    This must be called BEFORE task_registry.make_env().
    """
    from legged_gym.envs.go2.go2_config import GO2RoughCfg

    # Reset to P baseline (no vision, no heights)
    GO2RoughCfg.env.use_vision_in_actor  = False
    GO2RoughCfg.env.use_vision_in_critic = False
    GO2RoughCfg.env.detach_critic_vision = False
    GO2RoughCfg.env.camera_type = None
    GO2RoughCfg.terrain.measure_heights = False

    # Height observation dimension fallback (keep consistent with training)
    if not hasattr(GO2RoughCfg.env, "height_obs_dim"):
        GO2RoughCfg.env.height_obs_dim = 187

    # Privileged critic is not used here
    GO2RoughCfg.env.num_privileged_obs = None

    # Common sizes
    W, H = GO2RoughCfg.env.camera_res
    depth_dim = int(H) * int(W)
    P = GO2RoughCfg.env.proprio_dim
    Hdim = GO2RoughCfg.env.height_obs_dim

    # Apply preset
    if modality == "P":
        GO2RoughCfg.env.num_observations = P
    elif modality == "PH":
        GO2RoughCfg.terrain.measure_heights = True
        GO2RoughCfg.env.height_obs_dim = 187     # <-- force nonzero, match training
        Hdim = GO2RoughCfg.env.height_obs_dim
        GO2RoughCfg.env.use_vision_in_actor = False
        GO2RoughCfg.env.camera_type = None
        GO2RoughCfg.env.num_observations = P + Hdim
    elif modality == "PHD":
        GO2RoughCfg.terrain.measure_heights = True
        GO2RoughCfg.env.height_obs_dim = 187     # <-- force nonzero, match training
        Hdim = GO2RoughCfg.env.height_obs_dim
        GO2RoughCfg.env.use_vision_in_actor = True
        GO2RoughCfg.env.camera_type = "d"
        GO2RoughCfg.env.num_observations = P + Hdim + depth_dim
    elif modality == "PD":
        GO2RoughCfg.env.use_vision_in_actor = True
        GO2RoughCfg.env.camera_type = "d"
        GO2RoughCfg.env.num_observations = P + depth_dim
    elif modality == "PRIV":
        print("[WARN] PRIV is reserved; falling back to P for now.")
        GO2RoughCfg.env.num_observations = P
    else:
        raise ValueError(f"Unknown modality: {modality}")

    # Expose modality for debugging/logging
    GO2RoughCfg.env.modality = modality
    print(f"[MODALITY] {modality}  num_obs={GO2RoughCfg.env.num_observations}")
    if GO2RoughCfg.terrain.measure_heights:
        print(f"[HEIGHT] measure_heights=True height_obs_dim={Hdim}")
    else:
        print(f"[HEIGHT] measure_heights=False height_obs_dim=0")
# =======================================================

def export_policy_safe(policy_callable, out_dir: str, example_obs: torch.Tensor):
    """
    Trace only the pure inference callable: obs -> action.
    - Do NOT keep references to the actor_critic module to prevent JIT from recursing into it.
    - Use a real observation slice to derive the correct input shape per modality (P/PH/PD/PHD).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Infer device/dtype from example_obs
    dummy = example_obs[:1]
    device = dummy.device
    dtype = dummy.dtype

    # (Warm-up) Call once to ensure any lazy buffers are initialized.
    with torch.no_grad():
        _ = policy_callable(dummy)

    # Minimal wrapper that holds NO module references (prevents JIT recursion).
    class TraceOnly(torch.nn.Module):
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            # Call the provided pure callable; no module attributes captured.
            return policy_callable(obs)

    wrapper = TraceOnly().eval().to(device)

    # Stabilize JIT behavior
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)

    with torch.no_grad():
        ts = torch.jit.trace(wrapper, dummy, check_trace=False)

    out_path = os.path.join(out_dir, "policy_1.pt")
    ts.save(out_path)
    print(f"[EXPORT] TorchScript policy traced to: {out_path}  (obs_dim={int(dummy.shape[1])})")
    return out_path


def play(args, modality: str, run_suffix: str, eval_episodes: int = 0, eval_csv: str = ""):

    apply_modality_class_defaults(modality)

    # Get cfgs and mirror training run naming with modality tag
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # tag = modality + (f"-{run_suffix}" if run_suffix else "")
    # if hasattr(train_cfg.runner, "experiment_name"):
    #     train_cfg.runner.experiment_name = f"{train_cfg.runner.experiment_name}_{tag}"
    # if hasattr(train_cfg.runner, "run_name"):
    #     base = getattr(train_cfg.runner, "run_name", "")
    #     train_cfg.runner.run_name = f"{base}_{tag}" if base else tag

    # Light overrides for evaluation
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # -------------------------- addition of commands -----------------------------------
    # --- forward-only command lock (before make_env) ---


    # FWD = 0.5  # forward speed [m/s], 0.3~0.5 recommended

    # # 1) command resampling -> deactivate
    # if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "resampling_time"):
    #     env_cfg.commands.resampling_time = 1e9

    # # 2) fix the range as constant: x only for FWD, y/yaw=0
    # if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "ranges"):
    #     rng = env_cfg.commands.ranges
    #     if hasattr(rng, "lin_vel_x"):
    #         rng.lin_vel_x = [FWD, FWD]




        # if hasattr(rng, "lin_vel_y"):
        #     rng.lin_vel_y = [0.0, 0.0]
        # if hasattr(rng, "ang_vel_yaw"):
        #     rng.ang_vel_yaw = [0.0, 0.0]
        # if hasattr(rng, "heading"):
        #     rng.heading = [0.0, 0.0]

    # # 3) If this fork uses heading command, then turn it off
    # if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "heading_command"):
    #     env_cfg.commands.heading_command = False
    # -------------------------- addition of commands -----------------------------------

    # Build environment (the class-level modality toggles are already applied)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    print(f"[OBS] modality={modality} shape={tuple(obs.shape)}  num_obs(cfg)={env_cfg.env.num_observations}")


    # Load policy (resume the latest checkpoint under the tagged run)
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )

    # Use the real module for eval/export when needed
    actor_critic = ppo_runner.alg.actor_critic
    actor_critic.eval()             # set the actual module to eval mode
    torch.set_grad_enabled(False)   # global no-grad for rollout

    policy = ppo_runner.get_inference_policy(device=env.device)

    # --- debug init: policy is a callable, not an nn.Module ---
    policy_device = env.device
    t0 = time.time()

    # ---- Simple episode accumulators for eval-only logging ----
    use_eval = (eval_episodes > 0)
    if use_eval:
        # Per-env accumulators (on GPU to avoid host sync)
        ep_ret = torch.zeros(env.num_envs, device=env.device)            # sum of rewards
        ep_len = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        # Start x position to measure forward distance (x is index 0 in root state [x,y,z,...])
        ep_x0  = env.root_states[:, 0].clone()
        ep_collision = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        done_total = 0

        # CSV file (header)
        f = None
        if eval_csv:
            os.makedirs(os.path.dirname(eval_csv), exist_ok=True)
            f = open(eval_csv, "w")
            print("episode,mean_reward,length,distance,collision", file=f)
            f.flush()

    # --- rollout with timing breakdown (policy vs physx) ---
    steps_total = 10 * int(env.max_episode_length)

    # CUDA timing events (one-time setup)
    p_start = torch.cuda.Event(enable_timing=True)
    p_stop  = torch.cuda.Event(enable_timing=True)
    e_start = torch.cuda.Event(enable_timing=True)
    e_stop  = torch.cuda.Event(enable_timing=True)

    # simple accumulators
    policy_ms_acc = 0.0
    env_ms_acc = 0.0
    cnt = 0

    t0 = time.time()
    for i in range(steps_total):
        # --- policy forward (timed) ---
        p_start.record()
        with torch.no_grad():
            actions = policy(obs.detach())
        p_stop.record()

        # --- env.step (timed) ---
        e_start.record()
        step_out = env.step(actions)
        e_stop.record()

        # Support both 4- and 5-tuple returns:
        #  - 4-tuple: (obs, rewards, dones, infos)
        #  - 5-tuple: (obs, privileged_obs, rewards, dones, infos)
        if isinstance(step_out, (tuple, list)):
            if len(step_out) == 4:
                obs, rews, dones, infos = step_out
            elif len(step_out) == 5:
                obs, _priv, rews, dones, infos = step_out
            else:
                raise RuntimeError(f"Unexpected env.step() return length: {len(step_out)}")
        else:
            # Fallback: assume step_out is a 4-tuple-like object
            obs, rews, dones, infos = step_out

        # Wait for timings to be valid (CUDA events)
        torch.cuda.synchronize()

        if use_eval:
            # accumulate per-step reward and length
            ep_ret += rews
            ep_len += 1

            # accumulate per-episode collision flag:
            # if any penalized body has contact force > threshold at this step,
            # mark the episode as "collision happened".
            if hasattr(env, "contact_forces") and hasattr(env, "penalised_contact_indices"):
                # contact_forces: [num_envs, num_bodies, 3]
                cf = env.contact_forces[:, env.penalised_contact_indices, :]  # [N, K, 3]
                thr = getattr(env.cfg.rewards, "collision_contact_threshold", 1.0)
                hit_step = (torch.norm(cf, dim=-1) > thr).any(dim=1)          # [N] bool
                ep_collision |= hit_step                                      # OR accumulate over the episode

            # which envs finished this step
            done_idx = torch.nonzero(dones.view(-1), as_tuple=False).squeeze(-1)
            if done_idx.numel() > 0:
                # distance along +X: current x - starting x
                x_now = env.root_states[done_idx, 0]
                dist = x_now - ep_x0[done_idx]

                # mean reward per step within the episode
                mean_rew = ep_ret[done_idx] / torch.clamp(ep_len[done_idx].float(), min=1.0)

                # write rows (one row per finished env)
                if f is not None:
                    for j in range(done_idx.numel()):
                        env_id = int(done_idx[j].item())
                        done_total += 1
                        col_flag = int(ep_collision[env_id].item())  # 0 or 1 per episode
                        # CSV columns: episode,mean_reward,length,distance,collision
                        print(
                            f"{done_total},{float(mean_rew[j])},{int(ep_len[env_id])},{float(dist[j])},{col_flag}",
                            file=f,
                        )
                    f.flush()
                else:
                    done_total += int(done_idx.numel())

                # reset per-episode accumulators for those envs
                ep_ret[done_idx] = 0
                ep_len[done_idx] = 0
                ep_x0[done_idx]  = env.root_states[done_idx, 0]
                ep_collision[done_idx] = False

                # stop when enough episodes collected
                if done_total >= eval_episodes:
                    if f is not None:
                        f.close()
                    break

        # accumulate
        policy_ms = p_start.elapsed_time(p_stop)   # ms
        env_ms    = e_start.elapsed_time(e_stop)   # ms
        policy_ms_acc += policy_ms
        env_ms_acc += env_ms
        cnt += 1

        # every 200 steps: FPS + NaN + timing + memory + done ratio
        if (i + 1) % 200 == 0:
            has_nan_act = not torch.isfinite(actions).all()
            has_nan_obs = not torch.isfinite(obs).all()
            wall_fps = (i + 1) / (time.time() - t0)

            # memory snapshot (optional; may vary by driver)
            mem_alloc = torch.cuda.memory_allocated() / (1024**2)
            mem_resvd = torch.cuda.memory_reserved() / (1024**2)

            # done ratio (rough proxy of how many envs are resetting a lot)
            done_ratio = float(dones.float().mean().item()) if torch.is_tensor(dones) else 0.0

            avg_p = policy_ms_acc / max(cnt, 1)
            avg_e = env_ms_acc / max(cnt, 1)

            print(
                f"[PLAY] step={i+1}/{steps_total}  fps={wall_fps:.1f} "
                f"nan(act)={has_nan_act} nan(obs)={has_nan_obs} "
                f"avg_ms(policy)={avg_p:.2f} avg_ms(env)={avg_e:.2f} "
                f"mem(MB) alloc={mem_alloc:.1f}/resv={mem_resvd:.1f} "
                f"done_ratio={done_ratio:.2f}"
            )

            # (safety) break on NaN/Inf
            if has_nan_act or has_nan_obs:
                print("[PLAY] NaN/Inf detected -> breaking rollout early.")
                break

    # (optional) graceful shutdown / memory tidy
    try:
        env.close()
    except Exception:
        pass
    del env
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Close CSV if it is still open
    try:
        if use_eval and 'f' in locals() and f is not None and not f.closed:
            f.close()
    except Exception:
        pass

    # --- Optional export AFTER cleanup ---
    if EXPORT_POLICY:
        # Use the same device as observations/policy inference
        policy_device = obs.device  # safe and simple

        # Build a dummy observation with the exact runtime obs_dim
        obs_dim = int(getattr(env_cfg.env, "num_observations"))
        dummy = torch.zeros(1, obs_dim, device=policy_device, dtype=torch.float32)

        out_dir = os.path.join(
            LEGGED_GYM_ROOT_DIR, 'logs',
            train_cfg.runner.experiment_name,
            train_cfg.runner.run_name,
            'exported', 'policies'
        )
        # export_policy_safe must accept (policy_callable, out_dir, example_obs)
        export_policy_safe(policy, out_dir, example_obs=dummy)

        torch.cuda.synchronize()  # ensure export finishes on device


if __name__ == '__main__':
    # Runtime flags
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False

    # 1) Pop our custom CLI args BEFORE legged_gym's get_args()
    modality   = _pop_cli_arg("modality", default="P")
    run_suffix = _pop_cli_arg("run_suffix", default="")

    # Optional evaluation flags
    eval_episodes = int(_pop_cli_arg("eval_episodes", default="0"))
    eval_csv      = _pop_cli_arg("eval_csv",      default="")

    # 2) Parse standard legged_gym args
    args = get_args()

    # 3) Headless guards must be set before any windowing backend initializes
    if getattr(args, "headless", False):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        os.environ.pop("DISPLAY", None)

    # 4) Apply modality class-level defaults BEFORE env creation
    # apply_modality_class_defaults(modality)

    # 5) Run
    play(args, modality, run_suffix, eval_episodes=eval_episodes, eval_csv=eval_csv)
