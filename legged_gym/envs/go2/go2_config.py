from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# ----------------------------- (A) Shared constants/helpers -----------------------------
PROPRIO_DIM = 48  # size of proprioceptive observation (vel/angvel/gravity/cmd/joint states/last action)

def _depth_pixels(cfg) -> int:
    """Return number of raw depth pixels flattened into obs (W*H)."""
    w, h = cfg.env.camera_res[0], cfg.env.camera_res[1]
    return w * h

def _compute_num_obs(cfg, use_height: bool, use_depth: bool, height_obs_dim: int) -> int:
    """
    Compute the final observation length produced by the environment.
    NOTE: In the current pipeline, raw depth (W*H) is appended to obs and then encoded by the policy.
          Height adds 'height_obs_dim' scalars to the obs if enabled.
    """
    depth_dim = _depth_pixels(cfg) if use_depth else 0
    h_dim = height_obs_dim if use_height else 0
    return PROPRIO_DIM + h_dim + depth_dim

class GO2RoughCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
            # ===== Base =====
            num_envs = 30

            # Modality preset: one of {"P", "PH", "PD", "PHD", "PRIV"}
            modality = "P"

            # Observation/camera parameters (num_observations will be updated by apply_modality)
            num_observations = PROPRIO_DIM
            train_type = "none"
            camera_res = [424, 240]     # (W, H)
            camera_type = None          # "d"|"rgb"|None

            depth_near = 0.3
            depth_far  = 3.0
            follow_cam = False
            float_cam = False
            save_im = False

            # Subsample height measurement to reduce per-step cost (keep obs dim unchanged)
            height_update_interval = 2  # measure heights every 2 steps; keep last values otherwise

            # Toggles (set consistently by apply_modality)
            use_height = False                  # enable height measurements in obs
            use_vision_in_actor = False         # give depth to actor input
            use_vision_in_critic = False        # give depth to critic input (keep False for symmetric baselines)
            privileged_critic = False           # critic uses GT-only signals during training (Week5)
            detach_critic_vision = False

            # Size meta
            proprio_dim = PROPRIO_DIM
            cnn_output_size = 32                # output size of VisionEncoder (used inside the policy)
            height_obs_dim = 0                  # number of height samples appended by terrain (filled at runtime)
            num_privileged_obs = None           # size of critic-only GT obs (filled when PRIV is used)

    class terrain(LeggedRobotCfg.terrain):
        terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        # Terrain proportions by type (weights; internally converted to cumulative thresholds in make_terrain):
        # [0] Sloped plane (pyramid_sloped_terrain)
        #     - The first half of this bucket uses a negative slope (downhill), the second half positive (uphill).
        # [1] Sloped plane + height noise
        #     - pyramid_sloped_terrain + random_uniform_terrain overlay.
        # [2] Stairs (downstairs)
        #     - pyramid_stairs_terrain with negative step_height.
        # [3] Stairs (upstairs)
        #     - pyramid_stairs_terrain with positive step_height.
        # [4] Discrete obstacles (rectangles)
        #     - discrete_obstacles_terrain: ~20 rectangles, size ~1.0–2.0 m.
        # [5] Cell obstacles / random “walls”
        #     - discrete_obstacles_terrain_cells: dense rectangular cells (obs_scale-scaled size, height ~0.70–1.00).
        # [6] Stepping stones
        #     - stepping_stones_terrain with size/distance set by difficulty.
        # [7] Gaps
        #     - gap_terrain with gap_size based on difficulty.
        # [8] Pits
        #     - pit_terrain with depth based on difficulty.
        
        mesh_type = "trimesh"
        measure_heights = False # will be synced with env.use_height via apply_modality
        obstacle_x_cut_m = 20.0 # To remove the last part of terrain

        # Spawn a line of robots along the Y axis (facing +X), starting from the outside at x = 6.0
        spawn_axis = "y"           # 'y'면 긴 변이 y방향으로 가정, if 'y', treat the long edge as the Y direction
        spawn_min = -0.0           # 이 축의 시작(예: y 최소), start of the along-axis range (e.g., min y)
        spawn_max = +150.0         # 이 축의 끝(예: y 최대), end of the along-axis range (e.g., max y)
        spawn_ortho = 6.0          # 반대 축의 고정값(예: x = -5.6, 장애물 왼쪽 바깥), constant coordinate on the orthogonal axis (e.g., fixed x)
        spawn_edge_pad = 0.3       # 끝단 여유, padding at both ends of the along-axis range
        spawn_min_spacing = 0.6    # 한 줄에서 로봇 간 최소 간격, minimum spacing between robots on a single lane (meters)
        spawn_lane_spacing = 1.2   # 여러 줄로 넘어갈 때 레인 간 간격(노멀 방향), spacing between lanes when packing multiple rows (normal direction)
        spawn_lane_dir = +1.0      # 레인 늘릴 방향(+면 안쪽, -면 바깥쪽), lane growth direction (+ inward, - outward)



    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf"
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        collision_contact_threshold = 1.0  # [N] contact force norm threshold to count as collision
        
        class scales(LeggedRobotCfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0
            collision = -1.0   # negative scale (reward function returns +1 on collision)

    class termination:
        done_on_collision = True
        min_steps_before_termination = 50  # avoid killing episodes at t=0 on spawn noise

    # temporary
    class sim(LeggedRobotCfg.sim):
        class physx(LeggedRobotCfg.sim.physx):
            # ... existing PhysX params ...

            # Pre-allocate a large GPU contact pair buffer to avoid mid-run realloc spikes
            max_gpu_contact_pairs = 16 * 1024 * 1024  # 16M; safe upper bound for 100 envs
    # temporary

    # ------------------------------ (C) Preset application helper ------------------------------
    @staticmethod
    def apply_modality(cfg: "GO2RoughCfg"):
        """
        Apply a high-level modality preset to set all related flags and sizes consistently.

        cfg.env.modality ∈ {"P", "PH", "PD", "PHD", "PRIV"}:
          - "P"   : proprio only
          - "PH"  : proprio + height
          - "PD"  : proprio + raw depth (actor)
          - "PHD" : proprio + height + raw depth (actor)
          - "PRIV": privileged-critic training (critic-only GT); actor remains P by default
        """
        m = str(cfg.env.modality).upper()

        # 1) Decode preset
        use_h = ("H" in m)           # whether to append height measurements to obs
        use_d = ("D" in m)           # whether to append raw depth to obs (actor consumes it)
        is_priv = (m == "PRIV")      # privileged critic experiment

        # 2) Set feature toggles
        cfg.env.use_height = use_h
        cfg.terrain.measure_heights = use_h

        cfg.env.use_vision_in_actor = use_d
        cfg.env.camera_type = "d" if use_d else None

        # Keep critic symmetric for baselines unless PRIV is explicitly used
        cfg.env.use_vision_in_critic = False

        # Privileged critic: critic may receive GT-only features during training (Week5)
        cfg.env.privileged_critic = is_priv
        if is_priv:
            # If you plan to pass critic-only GT obs, set cfg.env.num_privileged_obs accordingly later.
            cfg.env.num_privileged_obs = cfg.env.num_privileged_obs or None

        # 3) Compute obs length (height_obs_dim can be updated later by the env once measured_heights is built)
        cfg.env.num_observations = _compute_num_obs(
            cfg, use_height=use_h, use_depth=use_d, height_obs_dim=cfg.env.height_obs_dim
        )

        # 4) One-shot debug log
        print(
            f"[CFG][MODALITY] {m} -> use_H={use_h} use_D={use_d} priv={is_priv} "
            f"num_obs={cfg.env.num_observations} "
            f"(Hdim={cfg.env.height_obs_dim}, depth_pixels={_depth_pixels(cfg) if use_d else 0})"
        )

# ---------------------------------------- (D) PPO config ----------------------------------------
class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "CustomActorCritic"
        run_name = ''
        experiment_name = 'rough_go2'

