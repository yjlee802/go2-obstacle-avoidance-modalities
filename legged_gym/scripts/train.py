import os
import sys
import argparse
from datetime import datetime

# ===== Modality CLI shim (consume custom flags before legged_gym argparse) =====
def _pop_cli_arg(name, default=None):
    """Remove --name or --name=VAL from sys.argv and return its value (string)."""
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

def parse_args():
    parser = argparse.ArgumentParser()
    # Task name as registered in task_registry (e.g., "go2")
    parser.add_argument("--task", type=str, default="go2")
    # Headless toggle
    parser.add_argument("--headless", action="store_true")
    # Modality preset: P, PH, PD, PHD  (PRIV reserved for later)
    parser.add_argument("--modality", type=str, default="P",
                        choices=["P", "PH", "PD", "PHD", "PRIV"],
                        help="P, PH, PD, PHD (PRIV = reserved; critic GT to be wired later)")
    # Optional: custom run name suffix to avoid checkpoint shape mismatch
    parser.add_argument("--run_suffix", type=str, default="")
    # Passthrough for other legged_gym args (seed, max_iterations, etc.) handled by get_args inside
    # but we keep this minimal and rely on internal arg parsing if needed.
    return parser.parse_known_args()[0]

def apply_modality_class_defaults(modality: str):
    from legged_gym.envs.go2.go2_config import GO2RoughCfg
    print(f"[HEIGHT] measure_heights={GO2RoughCfg.terrain.measure_heights} "
      f"height_obs_dim={GO2RoughCfg.env.height_obs_dim}")

    # Reset to P baseline
    GO2RoughCfg.env.use_vision_in_actor  = False
    GO2RoughCfg.env.use_vision_in_critic = False
    GO2RoughCfg.env.detach_critic_vision = False
    GO2RoughCfg.env.camera_type = None
    GO2RoughCfg.terrain.measure_heights = False

    # Common sizes
    W, H = GO2RoughCfg.env.camera_res
    depth_dim = int(W) * int(H)
    P = GO2RoughCfg.env.proprio_dim

    if modality == "P":
        GO2RoughCfg.env.num_observations = P

    elif modality == "PH":
        # Force height dim to the training-time constant
        GO2RoughCfg.terrain.measure_heights = True
        GO2RoughCfg.env.height_obs_dim = 187   # <-- critical line
        GO2RoughCfg.env.num_observations = P + GO2RoughCfg.env.height_obs_dim

    elif modality == "PD":
        GO2RoughCfg.env.use_vision_in_actor = True
        GO2RoughCfg.env.camera_type = "d"
        GO2RoughCfg.env.num_observations = P + depth_dim

    elif modality == "PHD":
        GO2RoughCfg.terrain.measure_heights = True
        GO2RoughCfg.env.height_obs_dim = 187   # <-- critical line
        GO2RoughCfg.env.use_vision_in_actor = True
        GO2RoughCfg.env.camera_type = "d"
        GO2RoughCfg.env.num_observations = P + GO2RoughCfg.env.height_obs_dim + depth_dim

    elif modality == "PRIV":
        print("[WARN] PRIV is reserved; falling back to P for now.")
        GO2RoughCfg.env.num_observations = P
    else:
        raise ValueError(f"Unknown modality: {modality}")

    GO2RoughCfg.env.modality = modality
    print(f"[MODALITY] {modality}  num_obs={GO2RoughCfg.env.num_observations}")
    if GO2RoughCfg.terrain.measure_heights:
        print(f"[HEIGHT] measure_heights=True height_obs_dim={GO2RoughCfg.env.height_obs_dim}")
    else:
        print(f"[HEIGHT] measure_heights=False height_obs_dim=0")



def main():
    # 1) Parse our minimal args first
    my = parse_args()

    # Remove our custom flags from sys.argv so gymutil.parse_arguments won't choke
    _pop_cli_arg("modality")
    _pop_cli_arg("run_suffix")

    # 2) Headless GUI guards (before any windowing backends initialize)
    if my.headless:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        os.environ.pop("DISPLAY", None)

    # 3) Apply class-level cfg overrides for the chosen modality BEFORE building the env
    apply_modality_class_defaults(my.modality)

    # (Optional) One-time height toggle/dim log for sanity check
    from legged_gym.envs.go2.go2_config import GO2RoughCfg
    print(f"[HEIGHT] measure_heights={GO2RoughCfg.terrain.measure_heights} "
          f"height_obs_dim={GO2RoughCfg.env.height_obs_dim}")

    # 4) Now import gym & registry (these will instantiate cfg/env using our class-level defaults)
    import isaacgym
    from legged_gym.utils import task_registry, get_args

    # 5) Merge legged_gym's own args (seed, iters, etc.). We keep our parsed args.
    args = get_args()
    # Keep task/headless consistent with our wrapper flags if provided
    if not hasattr(args, "task") or args.task is None:
        args.task = my.task
    if my.headless:
        args.headless = True

    # 4) Get cfg objects first (use our class-level defaults already applied)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # 5) Stamp names BEFORE runner is constructed (so dirs/checkpoints use the tag)
    tag = f"{my.modality}{('-' + my.run_suffix) if my.run_suffix else ''}"
    if hasattr(train_cfg.runner, "experiment_name"):
        train_cfg.runner.experiment_name = f"{getattr(train_cfg.runner, 'experiment_name', 'exp')}_{tag}"
    if hasattr(train_cfg.runner, "run_name"):
        base = getattr(train_cfg.runner, "run_name", '')
        train_cfg.runner.run_name = f"{base}_{tag}" if base else tag

    # 6) Build env with our env_cfg (class-level modality toggles are already in effect)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # 7) Build runner with the stamped train_cfg
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)

    # 8) Learn
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    main()

