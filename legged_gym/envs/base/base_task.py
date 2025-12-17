import cv2
import sys
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import torch
from torch import nn

# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()

        self.cfg = cfg

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type == "cuda" and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = "cpu"

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        # it seems like this line is turning off all rendering when headless, which prevents cameras.
        # if self.headless == True:
        #     self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(
            self.num_envs, self.num_obs, device=self.device, dtype=torch.float
        )
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.time_out_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(
                self.num_envs,
                self.num_privileged_obs,
                device=self.device,
                dtype=torch.float,
            )
        else:
            self.privileged_obs_buf = None

        # Allocate an optional per-step image buffer only when any vision path is enabled.
        use_any_vision = bool(getattr(cfg.env, "use_vision_in_actor", False) or
                            getattr(cfg.env, "use_vision_in_critic", False))
        cam_type = getattr(cfg.env, "camera_type", None)

        if use_any_vision and cam_type in ("d", "rgb"):
            self.image_buf = torch.zeros(
                self.num_envs,
                int(cfg.env.camera_res[1]),  # rows = height
                int(cfg.env.camera_res[0]),  # cols = width
                device=self.device,
                dtype=torch.float,
            )
        else:
            self.image_buf = None

            # self.num_privileged_obs = self.num_obs
        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync"
            )

            # Added: press 'C' to cycle to the next camera/env
            self.view_env = 0  # index of the env to display (default: robot 0)
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_C, "next_cam"
            )

            # Added: press 'N' to auto-select the env with the most nearby objects (multi-env)
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_N, "best_cam"
                )

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(
            torch.zeros(
                self.num_envs, self.num_actions, device=self.device, requires_grad=False
            )
        )
        return obs, privileged_obs

    def step(self, actions):
        raise NotImplementedError

    def render(self, sync_frame_time=True):
        # 0) Camera resolution / whether camera rendering is enabled
        # Render camera sensors only if any vision input is enabled and a valid camera type is set.
        use_cam = (
            bool(getattr(self.cfg.env, "use_vision_in_actor", False) or
                getattr(self.cfg.env, "use_vision_in_critic", False))
            and getattr(self.cfg.env, "camera_type", None) in ("d", "rgb")
)


        # 1) Handle viewer events
        if self.viewer:
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "next_cam" and evt.value > 0:
                    self.view_env = (self.view_env + 1) % self.num_envs
                    print(f"[VIEW] switched to env {self.view_env}")
                elif evt.action == "best_cam" and evt.value > 0:
                    self.view_env = self._pick_best_env()   # see helper below
                    print(f"[VIEW] best env -> {self.view_env}")
                

        # 2) Fetch sim results & step graphics
        if use_cam or self.viewer:
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)

        # 3) Draw the viewer
        if self.viewer:
            if self.enable_viewer_sync:
                self.gym.draw_viewer(self.viewer, self.sim, True)
            else:
                self.gym.poll_viewer_events(self.viewer)

        # 4) Render camera sensors here (right after drawing the viewer, before sync)
        if use_cam:
            self.gym.render_all_camera_sensors(self.sim)

            # (1) Check that camera handles exist
            if not hasattr(self, "camera_handles") or len(self.camera_handles) == 0:
                print("[CAM] no camera_handles")
            else:
                env_i = getattr(self, "view_env", 0)
                cam = self.camera_handles[env_i]

                # (2) Get resolution from cfg
                W, H = self.cfg.env.camera_res

                # COLOR (camera_type = "rgb")
                if self.cfg.env.camera_type == "rgb":
                    color = self.gym.get_camera_image(self.sim, self.envs[env_i], cam, gymapi.IMAGE_COLOR)
                    if isinstance(color, np.ndarray) and color.size > 0:
                        color = color.reshape(H, W, 4).copy()
                        bgr   = cv2.cvtColor(color[..., :3], cv2.COLOR_RGB2BGR)
                        if not self.headless:
                            cv2.imshow(f"[env {env_i}] HeadCam - Color", bgr)
                        if not hasattr(self, "_cam_dumped"):
                            cv2.imwrite("/tmp/headcam_color.png", bgr)
                            print("[CAM] saved /tmp/headcam_color.png", bgr.shape)
                            self._cam_dumped = True


                # DEPTH (camera_type = "d")
                if self.cfg.env.camera_type == "d":
                    depth_raw = self.gym.get_camera_image(self.sim, self.envs[env_i], cam, gymapi.IMAGE_DEPTH)
                    if isinstance(depth_raw, np.ndarray) and depth_raw.size > 0:
                        d = depth_raw.reshape(H, W).astype(np.float32)
                        d = -d  # Isaac: -Z → + distance
                        d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=np.inf)
                        finite_mask = np.isfinite(d)
                        if int(finite_mask.sum()) >= 50:
                            vals = d[finite_mask]
                            near = np.percentile(vals, 5); far = np.percentile(vals, 95)
                            if not np.isfinite(near): near = vals.min()
                            if not np.isfinite(far) or far - near < 1e-6: far = near + 1e-3
                            d_clip = np.clip(d, near, far)
                            d_norm = (d_clip - near) / (far - near + 1e-6)
                            d_vis_gray = ((1.0 - d_norm) * 255.0).astype(np.uint8)
                        else:
                            d_vis_gray = np.full((H, W), 127, np.uint8)

                        d_vis_color = cv2.applyColorMap(d_vis_gray, cv2.COLORMAP_TURBO)
                        if not self.headless:
                            cv2.imshow(f"[env {env_i}] HeadCam - Depth Gray",  d_vis_gray)
                            cv2.imshow(f"[env {env_i}] HeadCam - Depth Color", d_vis_color)

            if not self.headless: # Added to correct the error when play or train with '--headless'
                cv2.waitKey(1)

        if self.viewer and self.enable_viewer_sync and sync_frame_time:
            self.gym.sync_frame_time(self.sim)


    # Original def render code before modification

    # def render(self, sync_frame_time=True):
    #     if self.cfg.env.camera_res is not None:
    #         self.gym.render_all_camera_sensors(self.sim)

    #     if self.viewer:
    #         # check for window closed
    #         if self.gym.query_viewer_has_closed(self.viewer):
    #             sys.exit()

    #         # check for keyboard events
    #         for evt in self.gym.query_viewer_action_events(self.viewer):
    #             if evt.action == "QUIT" and evt.value > 0:
    #                 sys.exit()
    #             elif evt.action == "toggle_viewer_sync" and evt.value > 0:
    #                 self.enable_viewer_sync = not self.enable_viewer_sync

    #     if self.cfg.env.camera_res is not None or self.viewer:
    #         # fetch results
    #         if self.device != "cpu":
    #             self.gym.fetch_results(self.sim, True)
    #         self.gym.step_graphics(self.sim)

    #     if self.viewer:
    #         # step graphics
    #         if self.enable_viewer_sync:
    #             self.gym.draw_viewer(self.viewer, self.sim, True)
    #             if sync_frame_time:
    #                 self.gym.sync_frame_time(self.sim)
    #         else:
    #             self.gym.poll_viewer_events(self.viewer)

    def _pick_best_env(self, max_check=16):
        W, H = self.cfg.env.camera_res[0], self.cfg.env.camera_res[1]
        best_i, best_score = 0, -1.0
        for i, cam in enumerate(self.camera_handles[:min(self.num_envs, max_check)]):
            depth = self.gym.get_camera_image(self.sim, self.envs[i], cam, gymapi.IMAGE_DEPTH)
            if isinstance(depth, np.ndarray) and depth.size:
                d = -depth.reshape(H, W).astype(np.float32)
                d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=np.inf)
                score = np.isfinite(d).mean()
                if score > best_score:
                    best_score, best_i = score, i
        return best_i