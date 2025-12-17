import os
from isaacgym.torch_utils import *

import numpy as np
import cv2

from isaacgym import gymapi
from legged_gym.envs.base.legged_robot import LeggedRobot

from .go2_config import GO2RoughCfg


class GO2CameraMixin:
    def __init__(self, *args, **kwargs):
        self.follow_cam = None
        self.floating_cam = None
        super().__init__(*args, **kwargs)

    def init_aux_cameras(self, follow_cam=False, float_cam=False):
        if follow_cam:
            self.follow_cam, follow_trans = self.make_handle_trans(
                1920, 1080, 0, (1.0, -1.0, 0.0), (0.0, 0.0, 3 * 3.14 / 4)
            )
            body_handle = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], "base"
            )
            self.gym.attach_camera_to_body(
                self.follow_cam,  # camera_handle
                self.envs[0],
                body_handle,
                follow_trans,
                gymapi.FOLLOW_POSITION,
            )
            print(hasattr(gymapi, "FOLLOW_POSITION"), hasattr(gymapi, "CameraFollowMode"))

        if float_cam:
            self.floating_cam, _ = self.make_handle_trans(
                # 1280, 720, 0, (0, 0, 0), (0, 0, 0), hfov=50
                1920, 1080, 0, (0, 0, 0), (0, 0, 0)
            )
            camera_position = gymapi.Vec3(5, 5, 5)
            camera_target = gymapi.Vec3(0, 0, 0)
            self.gym.set_camera_location(
                self.floating_cam, self.envs[0], camera_position, camera_target
            )

    def make_handle_trans(self, width, height, i, trans, rot, hfov=None):
        # Create a camera with GPU tensor streaming enabled.
        camera_props = gymapi.CameraProperties()
        camera_props.width = width
        camera_props.height = height
        camera_props.enable_tensors = True             # required for GPU tensor access
        camera_props.near_plane = self.depth_near      # keep in sync with cfg.env.depth_near
        camera_props.far_plane  = self.depth_far       # keep in sync with cfg.env.depth_far
        if hfov is not None:
            camera_props.horizontal_fov = hfov
        camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(*trans)
        local_transform.r = gymapi.Quat.from_euler_zyx(*rot)
        return camera_handle, local_transform



class GO2(GO2CameraMixin, LeggedRobot):
    cfg: GO2RoughCfg

    def __init__(
        self, cfg, sim_params, physics_engine, sim_device, headless, record=False
    ):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self.depth_near = float(getattr(self.cfg.env, "depth_near", 0.3)) # 추가
        self.depth_far  = float(getattr(self.cfg.env, "depth_far",  3.0)) # 추가
        self.camera_handles = []
        self._dbg_saved = False
        self._cam_dbg = False  # (선택) CPU 디버그용

        print("GO2 INIT")

        self.init_aux_cameras(cfg.env.follow_cam, cfg.env.float_cam)

        # ---- Aux debug cams (optional UI cams)
        self.init_aux_cameras(cfg.env.follow_cam, cfg.env.float_cam)

        # ---- Create robot-mounted depth/rgb cameras only if vision is enabled ----
        use_any_vision = bool(getattr(cfg.env, "use_vision_in_actor", False) or
                            getattr(cfg.env, "use_vision_in_critic", False))
        cam_type = getattr(cfg.env, "camera_type", None)  # expected: "d" or "rgb" or None

        if use_any_vision and cam_type in ("d", "rgb"):
            # Debug print once
            if not hasattr(self, "_dbg_cfg_once"):
                print(f"[CFG] vision_on: actor={cfg.env.use_vision_in_actor} "
                    f"critic={cfg.env.use_vision_in_critic} type={cam_type} "
                    f"camera_res={cfg.env.camera_res}")
                self._dbg_cfg_once = True

            # Create one camera per env and attach to the robot base
            W, H = cfg.env.camera_res  # (W, H)
            self.camera_handles = []
            for i in range(self.num_envs):
                # Slightly forward & pitched down w.r.t. base
                cam_handle, cam_xform = self.make_handle_trans(
                    int(W), int(H),
                    i,
                    (0.35, 0.0, 0.0),     # local translation on base
                    (0.0, -3.14/6, 0.0)   # local rotation (ZYX Euler)
                )
                self.camera_handles.append(cam_handle)

                body_handle = self.gym.find_actor_rigid_body_handle(
                    self.envs[i], self.actor_handles[i], "base"
                )
                self.gym.attach_camera_to_body(
                    cam_handle,
                    self.envs[i],
                    body_handle,
                    cam_xform,
                    gymapi.FOLLOW_TRANSFORM,   # keep fixed transform on the base
                )

            # Sanity check: ensure we created as many handles as envs
            if len(self.camera_handles) != self.num_envs:
                print(f"[CAM][WARN] camera_handles={len(self.camera_handles)} != num_envs={self.num_envs}")
        else:
            # Vision disabled or unspecified type -> keep empty handles list
            self.camera_handles = []
            print(f"[CAM] Vision disabled (actor={getattr(cfg.env,'use_vision_in_actor',False)}, "
                f"critic={getattr(cfg.env,'use_vision_in_critic',False)}, type={cam_type}). "
                "Skipping camera creation.")


    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # Additionaly empty actuator network hidden states
        self.sea_hidden_state_per_env[:, env_ids] = 0.0
        self.sea_cell_state_per_env[:, env_ids] = 0.0

    def _init_buffers(self):
        super()._init_buffers()
        # Additionally initialize actuator network hidden state tensors
        self.sea_input = torch.zeros(
            self.num_envs * self.num_actions,
            1,
            2,
            device=self.device,
            requires_grad=False,
        )
        self.sea_hidden_state = torch.zeros(
            2,
            self.num_envs * self.num_actions,
            8,
            device=self.device,
            requires_grad=False,
        )
        self.sea_cell_state = torch.zeros(
            2,
            self.num_envs * self.num_actions,
            8,
            device=self.device,
            requires_grad=False,
        )
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(
            2, self.num_envs, self.num_actions, 8
        )
        self.sea_cell_state_per_env = self.sea_cell_state.view(
            2, self.num_envs, self.num_actions, 8
        )

    def _compute_torques(self, actions):
        return super()._compute_torques(actions)
