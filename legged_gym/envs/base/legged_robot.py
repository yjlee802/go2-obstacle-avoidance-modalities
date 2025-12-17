from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from warnings import WarningMessage
import numpy as np
import os
import cv2
from torchvision.utils import save_image
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask

from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi
from legged_gym.utils.terrain import Terrain
from .legged_robot_config import LeggedRobotCfg
import time
import pickle
import math

SAVE_IMG = False
MAX_DEPTH = 10

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """

        self.count = 0
        self.num_calls = 0
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        self.noise_level = 0.0 # 새로 추가
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_eval_function()
        self._prepare_reward_function()
        self.init_done = True

        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rb_states = gymtorch.wrap_tensor(_rb_states)
        all_indices = torch.arange(len(rb_states))
        self.global_feet_indices = torch.cat(
            [
                all_indices[all_indices % 17 == 4],
                all_indices[all_indices % 17 == 8],
                all_indices[all_indices % 17 == 12],
                all_indices[all_indices % 17 == 16],
            ]
        )
        self.first_iter = True

    def make_handle_trans(self, res, env_num, trans, rot, hfov=None):
        # TODO Add camera sensors here?
        camera_props = gymapi.CameraProperties()
        # print("FOV: ", camera_props.horizontal_fov)
        # camera_props.horizontal_fov = 75.0
        # 1280 x 720
        width, height = res
        camera_props.width = width
        camera_props.height = height
        camera_props.enable_tensors = True
        if hfov is not None:
            camera_props.horizontal_fov = hfov
        # print("envs[i]", self.envs[i])
        # print("len envs: ", len(self.envs))
        camera_handle = self.gym.create_camera_sensor(
            self.envs[env_num], camera_props
        )
        # print("cam handle: ", camera_handle)

        local_transform = gymapi.Transform()
        # local_transform.p = gymapi.Vec3(75.0, 75.0, 30.0)
        # local_transform.r = gymapi.Quat.from_euler_zyx(0, 3.14 / 2, 3.14)
        x, y, z = trans
        local_transform.p = gymapi.Vec3(x, y, z)
        a, b, c = rot
        local_transform.r = gymapi.Quat.from_euler_zyx(a, b, c)

        return camera_handle, local_transform

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.extras["torque"] = self.torques  # gymtorch.unwrap_tensor(self.torques)
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)
            
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        self.compute_eval()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        
        if self.cfg.domain_rand.push_robots:
            self._push_robots()

        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """Check if environments need to be reset"""
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    # Modified: arrange robots and obstacles in a single-file layout along a long axis
    def reset_idx(self, env_ids):
        if len(env_ids) == 0: return

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        M = len(env_ids)
        ori = self.env_origins[env_ids, :].to(self.device)

        jmag = 0.05  # ~5 cm jitter; keep equal spacing along the long axis unchanged
        if getattr(self, "_spawn_long_axis", "y") == "y":
            x = ori[:,0] + (torch.rand(M, device=self.device)-0.5)*2*jmag  # jitter only in the normal (x) direction
            y = ori[:,1]                                                   # keep equal spacing along y
            yaw = torch.zeros(M, device=self.device)                       # face +x
        else:
            x = ori[:,0]
            y = ori[:,1] + (torch.rand(M, device=self.device)-0.5)*2*jmag  # jitter only in the normal (y) direction
            import math
            yaw = torch.full((M,), 0.5*math.pi, device=self.device)        # face +y

        # z safety margin
        try:
            xy = torch.stack([x, y], dim=1)
            z = self._height_at_xy(xy) + 0.30
        except Exception:
            z = ori[:,2] + 0.30

        # ── One-time debug: print ranges and a few samples ─────────────────────
        if not hasattr(self, "_dbg_reset_once"):
            M = len(env_ids)
            axis = getattr(self, "_spawn_long_axis", "?")
            print(f"[RESET] M={M} axis={axis}  x({x.min().item():.2f}..{x.max().item():.2f})  "
                f"y({y.min().item():.2f}..{y.max().item():.2f})")
            # print a few samples (first, middle, last)
            i0, im, il = 0, M // 2, M - 1
            print(f"[RESET] samples: 0=({float(x[i0]):.2f},{float(y[i0]):.2f})  "
                f"mid=({float(x[im]):.2f},{float(y[im]):.2f})  "
                f"last=({float(x[il]):.2f},{float(y[il]):.2f})")
            self._dbg_reset_once = True
        # ─────────────────────────────────────────────────────────
        
        self.root_states[env_ids, 0] = x
        self.root_states[env_ids, 1] = y
        self.root_states[env_ids, 2] = z

        zeros = torch.zeros_like(yaw)
        q = quat_from_euler_xyz(zeros, zeros, yaw)        # (M,4), xyzw
        self.root_states[env_ids, 3:7] = q

        # Apply the updated root states to the simulator for the selected envs
        self._set_actor_root_state_tensor_indexed(env_ids)

        # Continue with the rest of the reset logic
        self._resample_commands(env_ids)

        # reset buffers
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        if not hasattr(self, "_dbg_once"):
            print(f"[RESET] x({x.min():.2f}..{x.max():.2f})  y({y.min():.2f}..{y.max():.2f})  axis={getattr(self,'_spawn_long_axis','?')}")
            self._dbg_once = True


    # Original def reset_idx code before modification

    # def reset_idx(self, env_ids):
    #     """ Reset some environments.
    #         Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
    #         [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
    #         Logs episode info
    #         Resets some buffers

    #     Args:
    #         env_ids (list[int]): List of environment ids which must be reset
    #     """
    #     if len(env_ids) == 0:
    #         return
        
    #     # reset robot states
    #     self._reset_dofs(env_ids)
    #     self._reset_root_states(env_ids)

    #     self._resample_commands(env_ids)

    #     # reset buffers
    #     self.actions[env_ids] = 0.0
    #     self.last_actions[env_ids] = 0.0
    #     self.last_dof_vel[env_ids] = 0.0
    #     self.feet_air_time[env_ids] = 0.0
    #     self.episode_length_buf[env_ids] = 0
    #     self.reset_buf[env_ids] = 1
    #     # fill extras
    #     self.extras["episode"] = {}
    #     for key in self.episode_sums.keys():
    #         self.extras["episode"]["rew_" + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
    #         self.episode_sums[key][env_ids] = 0.
    #     if self.cfg.commands.curriculum:
    #         self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
    #     # send timeout info to the algorithm
    #     if self.cfg.env.send_timeouts:
    #         self.extras["time_outs"] = self.time_out_buf
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_eval(self):
        """Compute evals
        Reports eval function values, not used in training.
        """
        for i in range(len(self.eval_functions)):
            name = self.eval_names[i]
            rew = self.eval_functions[i]()
            self.eval_sums[name] += rew

    def compute_observations(self):
        """
        Build obs = [proprio | (height?) | (raw depth flattened?)] according to cfg toggles.
        - Height is appended only when terrain.measure_heights == True.
        - Raw depth (W*H) is appended only when env.use_vision_in_actor == True and camera_type == "d".
        - Noise is applied to proprio(+height) only (not to raw depth tail).
        """

        # -------------------- 1) Proprio (unchanged) --------------------
        obs = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )

        # -------------------- 2) Height (optional; with light caching) --------------------
        if getattr(self.cfg.terrain, "measure_heights", False):
            # One-time init: tick counter and cached height buffer
            if not hasattr(self, "_height_tick"):
                self._height_tick = 0
            if not hasattr(self, "_heights_buf"):
                # Compute once to infer the correct height dimension and init cache
                h_now = torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1.0, 1.0
                ) * self.obs_scales.height_measurements
                self._heights_buf = h_now.clone()

                # One-time sync back to cfg for consistent bookkeeping
                if hasattr(self.cfg.env, "height_obs_dim"):
                    self.cfg.env.height_obs_dim = int(self._heights_buf.shape[1])
                    use_d = bool(getattr(self.cfg.env, "use_vision_in_actor", False)) and (
                        getattr(self.cfg.env, "camera_type", None) == "d"
                    )
                    W, H = self.cfg.env.camera_res  # (W, H)
                    depth_dim = (H * W) if use_d else 0
                    self.cfg.env.num_observations = int(
                        self.cfg.env.proprio_dim + self.cfg.env.height_obs_dim + depth_dim
                    )

            # Update only every N steps (default 1 = update every step)
            interval = int(getattr(self.cfg.env, "height_update_interval", 1))
            if (self._height_tick % max(1, interval)) == 0:
                h_now = torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1.0, 1.0
                ) * self.obs_scales.height_measurements
                # In-place update to avoid new allocations
                self._heights_buf.copy_(h_now)
            self._height_tick += 1

            # Append cached heights to observation
            obs = torch.cat((obs, self._heights_buf), dim=-1)


        # -------------------- 3) Depth (optional; raw W*H appended only for depth cameras) --------------------
        want_depth = bool(getattr(self.cfg.env, "use_vision_in_actor", False)) and (
            getattr(self.cfg.env, "camera_type", None) == "d"
        )

        if want_depth and hasattr(self, "camera_handles") and len(self.camera_handles) > 0:
            # Debug: one-time print to verify camera resolution and env count
            if not hasattr(self, "_dbg_camres_once"):
                print(f"[OBS] camera_res(W,H)={self.cfg.env.camera_res}  num_envs={self.num_envs}")
                self._dbg_camres_once = True

            W, H = self.cfg.env.camera_res  # (W, H)
            width, height = int(W), int(H)
            image_buf = torch.zeros(self.num_envs, height, width, device=self.device)

            self.gym.start_access_image_tensors(self.sim)
            for i in range(self.num_envs):
                im = self.gym.get_camera_image_gpu_tensor(
                    self.sim,
                    self.envs[i],
                    self.camera_handles[i],
                    gymapi.IMAGE_DEPTH,
                )
                im = gymtorch.wrap_tensor(im)
                image_buf[i] = im

                if getattr(self.cfg.env, "save_im", False):
                    # Save a quick normalized preview (optional)
                    trans_im = im.detach().clone()
                    trans_im = -1.0 / trans_im
                    maxv = torch.max(trans_im)
                    if maxv > 0:
                        trans_im = trans_im / maxv
                    save_image(
                        trans_im.view((height, width, 1)).permute(2, 0, 1).float(),
                        f"images/dim/{getattr(self, 'count', 0):05}.png",
                    )

                # Optional raw PNG save if your globals exist (MAX_DEPTH/SAVE_IMG/cv2/np/time)
                if "SAVE_IMG" in globals() and SAVE_IMG:
                    img = torch.clamp(-im, 0, MAX_DEPTH) / MAX_DEPTH
                    img = np.uint8(img.cpu().numpy() * 255)
                    cv2.imwrite(f"images/{time.time()}.png", img)

            self.gym.end_access_image_tensors(self.sim)

            # Append raw depth (flattened)
            obs = torch.cat((obs, image_buf.view(self.num_envs, -1)), dim=-1)

        # -------------------- 4) Add noise (exclude raw depth part) --------------------
        if self.add_noise and self.noise_level > 0.0:
            W, H = self.cfg.env.camera_res
            depth_dim = (H * W) if want_depth else 0
            proprio_plus_height_len = obs.shape[1] - depth_dim

            if self.noise_scale_vec.shape[0] != obs.shape[1]:
                self.noise_scale_vec = torch.zeros(obs.shape[1], device=self.device)
                self.noise_scale_vec[:proprio_plus_height_len] = self.noise_level

            obs = obs + (2 * torch.rand_like(obs) - 1) * self.noise_scale_vec

        # -------------------- 5) Finalize --------------------
        self.obs_buf = obs

        # Optional defensive check: shape must match cfg.env.num_observations
        exp = int(getattr(self.cfg.env, "num_observations", obs.shape[1]))
        if obs.shape[1] != exp:
            print(
                f"[OBS][WARN] obs_dim={obs.shape[1]} != cfg.env.num_observations={exp}. "
                f"Check modality toggles and height_obs_dim initialization."
            )
        return self.obs_buf


    def create_sim(self):
        """Creates simulation, terrain and evironments"""
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ["heightfield", "trimesh"]:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]"
            )
        self._create_envs()

        if getattr(self, "viewer", None) is not None and not getattr(self, "_spawn_lines", False):
            color = gymapi.Vec3(0.0, 1.0, 0.0)
            gymutil.draw_lines([(self._debug_line_p0, self._debug_line_p1)], self.gym, self.viewer, self.envs[0], color)
            self._spawn_lines = True

    def set_camera(self, position, lookat):
        """Set camera position and direction"""
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(
                    friction_range[0], friction_range[1], (num_buckets, 1), device="cpu"
                )
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props

    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (
            (
                self.episode_length_buf
                % int(self.cfg.commands.resampling_time / self.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0
            )

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type == "V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0]
        if len(push_env_ids) == 0:
            return
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        
        env_ids_int32 = push_env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

   
    
    def update_command_curriculum(self, env_ids):
        """Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0. # previous actions

        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.torques = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.p_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.d_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device,
            requires_grad=False,
        )  # TODO change this
        self.feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
                             for name in self.reward_scales.keys()}
    def _prepare_eval_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales
        to_remove = []
        for name, scale in self.eval_scales.items():
            if not scale:
                to_remove.append(name)
        for name in to_remove:
            self.eval_scales.pop(name)

        # prepare list of functions
        self.eval_functions = []
        self.eval_names = []
        for name, scale in self.eval_scales.items():
            print(name, scale)
            if name == "termination":
                continue
            self.eval_names.append(name)
            name = "_eval_" + name
            self.eval_functions.append(getattr(self, name))

        # reward episode sums
        self.eval_sums = {
            name: torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            for name in self.eval_scales.keys()
        }

    def _create_ground_plane(self):
        """Adds a ground plane to the simulation, sets friction and restitution based on the cfg."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """Adds a heightfield terrain to the simulation, sets parameters based on the cfg."""
        hf_params = gymapi.HeightFieldProperties()
        hf_params.column_scale = self.terrain.horizontal_scale
        hf_params.row_scale = self.terrain.horizontal_scale
        hf_params.vertical_scale = self.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.border_size
        hf_params.transform.p.y = -self.terrain.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples)
            .view(self.terrain.tot_rows, self.terrain.tot_cols)
            .to(self.device)
        )

    def _create_trimesh(self):
        """Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        #"""
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(
            self.sim,
            self.terrain.vertices.flatten(order="C"),
            self.terrain.triangles.flatten(order="C"),
            tm_params,
        )
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples)
            .view(self.terrain.tot_rows, self.terrain.tot_cols)
            .to(self.device)
        )

    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = (
            self.cfg.asset.replace_cylinder_with_capsule
        )
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)

        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False
        )
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            # Purpose: single-file alignment
            # 1) Ensure origin is a (3,) torch tensor
            pos = self.env_origins[i, :].to(self.device).clone()  # (3,)

            # 2) Generate XY jitter with the 2D shape required by torch_rand_float
            xy_jitter = torch_rand_float(-0.20, 0.20, (2, 1), device=self.device).squeeze(1)  # -> (2,)

            # 3) Add jitter to XY only (columns 0:2)
            pos[0:2] += xy_jitter

            # 4) Set start pose (set yaw=0 to face +x, if desired)
            start_pose.p = gymapi.Vec3(float(pos[0]), float(pos[1]), float(pos[2]))
            start_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, 0.0)  # keep/remove as needed

            if i in (0, self.num_envs // 2, self.num_envs - 1):
                print(f"[CREATE] i={i} pos=({float(pos[0]):.2f},{float(pos[1]):.2f}) "
                    f"axis={getattr(self,'_spawn_long_axis','?')}")
            # Purpose: single-file alignment         

            # Original code before modification
            # pos = self.env_origins[i].clone()
            # pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(
            #     1
            # )
            # start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(
                rigid_shape_props_asset, i
            )
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                self.cfg.asset.name,
                i,
                self.cfg.asset.self_collisions,
                0,
            )
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle
            )
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, actor_handle, body_props, recomputeInertia=True
            )
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self._build_actor_root_ids()  # Added: robot (actor) index cache — used for single-file alignment (single-line addition)

        self.feet_indices = torch.zeros(
            len(feet_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i]
            )

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )


    # Added for single-file layout of obstacles and robots
    def _build_actor_root_ids(self):
        """각 env의 로봇 actor가 sim 전체에서 차지하는 root-state 인덱스를 int32 텐서로 보관."""
        """Store, as an int32 tensor, the SIM-domain root-state index of each env's robot actor."""
        ids = []
        for i in range(self.num_envs):
            env = self.envs[i]
            actor = self.actor_handles[i]
            actor_id = self.gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM)  # int
            ids.append(actor_id)
        self._actor_root_ids = torch.tensor(ids, dtype=torch.int32, device=self.device)

    def _set_actor_root_state_tensor_indexed(self, env_ids):
        """env_ids로 지정된 로봇들의 루트 상태만 GPU에 반영."""
        """Update on the GPU only the root states of the robots specified by env_ids."""
        assert hasattr(self, "_actor_root_ids"), "Call _build_actor_root_ids() after creating actors."
        actor_ids = self._actor_root_ids[env_ids].contiguous()  # (M,)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),          # (num_actors, 13)
            gymtorch.unwrap_tensor(actor_ids),
            actor_ids.numel(),
        )
    # Added for single-file layout of obstacles and robots

    # Added for single-file layout of obstacles and robots
    def _get_env_origins(self):
        try:
            tcfg = self.cfg.terrain
            print(f"[SPAWN_CFG] axis={getattr(tcfg,'spawn_axis',None)} "
                f"min={getattr(tcfg,'spawn_min',None)} "
                f"max={getattr(tcfg,'spawn_max',None)} "
                f"ortho={getattr(tcfg,'spawn_ortho',None)} "
                f"edge_pad={getattr(tcfg,'spawn_edge_pad',None)} "
                f"min_spacing={getattr(tcfg,'spawn_min_spacing',None)} "
                f"lane_spacing={getattr(tcfg,'spawn_lane_spacing',None)} "
                f"lane_dir={getattr(tcfg,'spawn_lane_dir',None)}")
        except Exception as e:
            print(f"[SPAWN_CFG] <ERR> {e}")
        """Trimesh: 긴 변 전체에 균등 배치 + 넘치면 노멀 방향으로 레인 패킹."""
        """Trimesh: uniformly distribute along the long edge; if capacity is exceeded,
        pack additional lanes in the normal direction."""

        self.custom_origins = True
        dev = self.device
        N   = int(self.num_envs)

        # --- Read ranges/spacings from config ---
        tcfg = self.cfg.terrain
        axis       = getattr(tcfg, "spawn_axis", "y")           # 'y' or 'x'
        a_min      = float(getattr(tcfg, "spawn_min", -5.0))
        a_max      = float(getattr(tcfg, "spawn_max",  5.0))
        ortho_const= float(getattr(tcfg, "spawn_ortho", -5.6))
        pad_edge   = float(getattr(tcfg, "spawn_edge_pad", 0.6))
        min_space  = float(getattr(tcfg, "spawn_min_spacing", 0.35))
        lane_space = float(getattr(tcfg, "spawn_lane_spacing", 0.60))
        lane_dir   = float(getattr(tcfg, "spawn_lane_dir", -1.0))  # +1 = inward, -1 = outward

        # Apply edge padding
        a0, a1 = a_min + pad_edge, a_max - pad_edge
        L = max(1e-6, a1 - a0)

        # Per-lane capacity (respecting minimum spacing)
        cap = max(1, int(math.floor(L / min_space)) + 1)        # maximum capacity per lane
        n_lanes = int(math.ceil(N / cap))                        # number of lanes required
        per_lane = int(math.ceil(N / n_lanes))                   # actual placements per lane (<= cap guaranteed)

        # 긴 변 방향 등분 좌표는 'per_lane' 개
        # Along-axis coordinates have 'per_lane' evenly spaced points
        if per_lane > 1:
            along = torch.linspace(a0, a1, steps=per_lane, device=dev)
        else:
            along = torch.tensor([(a0 + a1) * 0.5], device=dev)

        idx = torch.arange(N, device=dev)

        # Interleaving mapping (fill along the long edge first → then add lanes)
        lane_id = (idx % n_lanes).clamp_max(n_lanes - 1)
        pos_id  = (idx // n_lanes).clamp_max(per_lane - 1)       # Note: based on per_lane, not cap

        if axis == "y":
            ys = along[pos_id]
            xs = torch.full((N,), ortho_const, device=dev) + lane_dir * lane_id.float() * lane_space
            self._spawn_long_axis = "y"
        else:
            xs = along[pos_id]
            ys = torch.full((N,), ortho_const, device=dev) + lane_dir * lane_id.float() * lane_space
            self._spawn_long_axis = "x"

        zs = torch.zeros(N, device=dev)
        self.env_origins = torch.stack([xs, ys, zs], dim=1)
        # === Additional diagnostics (for train vs play comparison) ===
        try:
            mode = "PLAY" if getattr(self, "eval_mode", False) else "TRAIN"
            print(f"[ORIGIN] mode={mode} N={N} axis={self._spawn_long_axis}", flush=True)

            # Spacing stats along the long axis
            if self._spawn_long_axis == "y":
                d = torch.diff(ys)
            else:
                d = torch.diff(xs)
            if d.numel() > 0:
                print(f"[ORIGIN] along-spacing min/mean/max = "
                      f"{float(d.min()):.4f}/{float(d.mean()):.4f}/{float(d.max()):.4f}", flush=True)

            # Lane stats (orthogonal direction)
            if self._spawn_long_axis == "y":
                unique_lanes = torch.unique(xs)
            else:
                unique_lanes = torch.unique(ys)
            print(f"[ORIGIN] lanes={unique_lanes.numel()} lane_vals={unique_lanes[:5].tolist()}...", flush=True)

            # Print a few leading coordinate samples
            n = min(5, self.env_origins.shape[0])
            for i in range(n):
                ox, oy, oz = [float(v) for v in self.env_origins[i].tolist()]
                print(f"[ORIGIN] {i}: ({ox:.3f}, {oy:.3f}, {oz:.3f})", flush=True)

            # Overall range summary
            xs_min, xs_max = float(xs.min()), float(xs.max())
            ys_min, ys_max = float(ys.min()), float(ys.max())
            print(f"[ORIGIN] x-range=({xs_min:.3f},{xs_max:.3f})  y-range=({ys_min:.3f},{ys_max:.3f})", flush=True)
        except Exception as e:
            print(f"[ORIGIN] <ERR> {e}", flush=True)
         # === Additional diagnostics (for train vs play comparison) ===

        if not hasattr(self, "_dbg_spawn_cfg"):
            print(f"[SPAWN_CFG] axis={self._spawn_long_axis} min={self.cfg.terrain.spawn_min} "
                f"max={self.cfg.terrain.spawn_max} ortho={self.cfg.terrain.spawn_ortho} "
                f"edge_pad={self.cfg.terrain.spawn_edge_pad} min_spacing={self.cfg.terrain.spawn_min_spacing} "
                f"lane_spacing={self.cfg.terrain.spawn_lane_spacing} lane_dir={self.cfg.terrain.spawn_lane_dir}")
            self._dbg_spawn_cfg = True

        # ── At the end of _get_env_origins(): store only the two endpoints of the line
        x0 = float(self.env_origins[:,0].mean().item())  # if long axis is y, x is constant
        ymin = float(self.env_origins[:,1].min().item())
        ymax = float(self.env_origins[:,1].max().item())
        self._debug_line_p0 = gymapi.Vec3(x0, ymin, 0.05)
        self._debug_line_p1 = gymapi.Vec3(x0, ymax, 0.05)

        try:
            xs = self.env_origins[:, 0]
            ys = self.env_origins[:, 1]
            print(f"[PACK] axis={getattr(self,'_spawn_long_axis','?')}  "
                f"x-range=({xs.min().item():.2f},{xs.max().item():.2f})  "
                f"y-range=({ys.min().item():.2f},{ys.max().item():.2f})  "
                f"N={self.num_envs}")
            # Also print the endpoints
            print(f"[SPAWN-ORIGINS] first/last: "
                f"{self.env_origins[0,:2].tolist()} -> {self.env_origins[-1,:2].tolist()}")
        except Exception as e:
            print(f"[PACK] <ERR> {e}")
    # Added for single-file layout of obstacles and robots

    # Original def _get_env_origins code before modification
    # def _get_env_origins(self):
    #     """Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
    #     Otherwise create a grid.
    #     """
      
    #     self.custom_origins = False
    #     self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
    #     # create a grid of robots
    #     num_cols = np.floor(np.sqrt(self.num_envs))
    #     num_rows = np.ceil(self.num_envs / num_cols)
    #     xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
    #     spacing = self.cfg.env.env_spacing
    #     self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
    #     self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
    #     self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.eval_scales = class_to_dict(self.cfg.evals)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
     

        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
    def _init_height_points(self):
        """Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(
            self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False
        )
        x = torch.tensor(
            self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False
        )
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(
            self.num_envs,
            self.num_height_points,
            3,
            device=self.device,
            requires_grad=False,
        )
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == "plane":
            assert(False)
            return torch.zeros(
                self.num_envs,
                self.num_height_points,
                device=self.device,
                requires_grad=False,
            )
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = (
                quat_apply_yaw(
                    self.base_quat[env_ids].repeat(1, self.num_height_points),
                    self.height_points[env_ids],
                )
                + (self.root_states[env_ids, :3]).unsqueeze(1)
            )
        else:
            points = quat_apply_yaw(
                self.base_quat.repeat(1, self.num_height_points), self.height_points
            ) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        # print("Unique heights: ", torch.unique(self.height_samples) * self.terrain.cfg.vertical_scale)
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)


        heights = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

        return heights
    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.root_states[:, 2]
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    # def _reward_collision(self):
    #     # Penalize collisions on selected bodies
    #     return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_collision(self):
        """Return 1.0 if any penalized body collides (force > threshold), else 0.0."""
        # contact_forces: [num_envs, num_bodies, 3]
        # penalised_contact_indices: LongTensor with body indices to monitor
        cf = self.contact_forces[:, self.penalised_contact_indices, :]  # [N, K, 3]
        thr = getattr(self.cfg.rewards, "collision_contact_threshold", 1.0)
        hit_any = (torch.norm(cf, dim=-1) > thr).any(dim=1)             # [N] bool
        hit_f = hit_any.float()                                         # [N] 0/1

        # expose per-step 0/1 for logging; rsl-rl will pick this up in infos
        self.extras["collision"] = hit_f
        return hit_f
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

