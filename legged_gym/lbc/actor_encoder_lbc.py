import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from rsl_rl.modules.models.simple_cnn import SimpleCNN

# ------------------------- Constants -------------------------
PROPRIO_SIZE = 48  # canonical proprio length used across the project


# ------------------------- CNN encoder wrappers -------------------------
class CNNEncoder(nn.Module):
    def __init__(self, image_size, cnn_out_size):
        """
        image_size: [H, W, C], e.g., [240, 424, 1]
        The SimpleCNN expects a dict with channels-last images under key "depth".
        """
        super().__init__()
        self.image_size = image_size

        # Minimal replacement for gym.spaces.Box to satisfy SimpleCNN interface.
        class DummyBox:
            def __init__(self, low, high, shape, dtype):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

        class ObservationSpace:
            def __init__(self):
                # Training path feeds depth as float32 in [0, 1]
                self.spaces = {
                    "depth": DummyBox(low=0.0, high=1.0, shape=image_size, dtype=torch.float32)
                }

        self.cnn = SimpleCNN(ObservationSpace(), cnn_out_size)

    def forward(self, depth):
        """
        depth: [B, 1, H, W] or [B, H, W, 1]
        Returns a feature vector of size cnn_out_size.
        """
        if not isinstance(depth, torch.Tensor):
            depth = depth["depth"]  # dict-style input

        # Normalize layout to channels-last (B,H,W,C) for SimpleCNN
        if depth.dim() == 4 and depth.shape[1] in (1, 3):          # [B,C,H,W]
            images_cl = depth.permute(0, 2, 3, 1).contiguous()     # -> [B,H,W,C]
        elif depth.dim() == 4 and depth.shape[-1] in (1, 3):       # [B,H,W,C]
            images_cl = depth.contiguous()
        else:
            raise RuntimeError(f"Unexpected depth shape {tuple(depth.shape)}")

        if not hasattr(self, "_dbg_cnn_once"):
            print("[ENC] pass to SimpleCNN:", tuple(images_cl.shape))
            self._dbg_cnn_once = True

        out = self.cnn({"depth": images_cl})  # SimpleCNN will permute internally as needed
        return out


class VisionEncoder(nn.Module):
    def __init__(self, image_size=[240, 424, 1], cnn_output_size=32):
        super(VisionEncoder, self).__init__()
        self.encoder = CNNEncoder(image_size, cnn_output_size)
        print(f"(Student) ENCODER CNN: {self.encoder}")

    def forward(self, depth_tensor):
        """Forward the (preprocessed) depth tensor through the CNN encoder."""
        return self.encoder(depth_tensor)


# ------------------------- Actor -------------------------
class Actor(nn.Module):
    """
    Actor that can consume:
      - proprio only (P),
      - proprio + height (PH),
      - proprio + depth features (PD),
      - proprio + height + depth features (PHD).
    Depth is preprocessed to [0,1], optionally downsampled, and then encoded by a CNN.
    """
    is_recurrent = False

    def __init__(
        self,
        num_actions,
        num_actor_obs=None, # kept for compatibility; not used
        actor_hidden_dims=[512, 256, 128],
        image_size=[240, 424, 1],
        cnn_output_size=32,            # encoded depth feature dim
        proprio_dim=PROPRIO_SIZE,
        camera_res=None,               # (W, H)
        train_type="lbc",
        depth_near: float = 0.3,
        depth_far: float = 3.0,
        activation="elu",
        init_noise_std=1.0,
        encoder_res=(84, 84),          # (H, W) fed to the CNN after downsampling
        # ---- New toggles/size metas (read from cfg in CustomActorCritic) ----
        use_vision_in_actor=False,     # if True, expect raw depth in obs
        use_height=False,              # if True, expect height scalars in obs
        height_obs_dim=0,              # number of height scalars appended to obs
        **kwargs,
    ):
        super(Actor, self).__init__()
        if kwargs:
            print("Actor.__init__ ignored extra keys:", list(kwargs.keys()))

        act_fn = get_activation(activation)

        # --- Store flags/sizes ---
        self.train_type = train_type
        self.use_vision = bool(use_vision_in_actor)
        self.use_height = bool(use_height)

        self.proprio_dim = int(proprio_dim)
        self.height_obs_dim = int(height_obs_dim)
        self.enc_dim = int(cnn_output_size)

        # Depth intrinsics
        self.depth_near = float(depth_near)
        self.depth_far = float(depth_far)
        print(f"[CFG][Actor] depth_near={self.depth_near} depth_far={self.depth_far}")

        # Camera size (raw from simulator)
        if camera_res is None:
            H, W = image_size[0], image_size[1]
        else:
            W, H = camera_res
        self.H, self.W = int(H), int(W)
        self.D = int(self.H * self.W) if self.use_vision else 0  # number of raw depth pixels when enabled

        # Downsampled resolution before CNN
        self.enc_H, self.enc_W = int(encoder_res[0]), int(encoder_res[1])
        if not hasattr(self, "_dbg_actor_once"):
            print(f"[Actor] raw depth HW={(self.H, self.W)}, enc HW={(self.enc_H, self.enc_W)}")
            self._dbg_actor_once = True

        # Vision encoder is instantiated only if depth is used
        self.vision_encoder = VisionEncoder([self.enc_H, self.enc_W, 1], cnn_output_size) if self.use_vision else None

        # --- Build MLP with dynamic input dim ---
        # final input = proprio [P] + (height [Hdim]) + (encoded depth [E])

        mlp_input_dim = self.proprio_dim
        if self.use_height and self.height_obs_dim > 0:
            mlp_input_dim += self.height_obs_dim
        if self.use_vision:
            mlp_input_dim += self.enc_dim

        layers = [nn.Linear(mlp_input_dim, actor_hidden_dims[0]), act_fn]
        for l in range(len(actor_hidden_dims) - 1):
            layers += [nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]), act_fn]
        layers += [nn.Linear(actor_hidden_dims[-1], num_actions)]
        self.actor = nn.Sequential(*layers)

        print(f"(Student) Flags: use_height={self.use_height} height_dim={self.height_obs_dim} "
              f"use_vision={self.use_vision} enc_dim={self.enc_dim}")
        print(f"(Student) MLP input = {mlp_input_dim} (P={self.proprio_dim}"
              f"{' + H='+str(self.height_obs_dim) if self.use_height else ''}"
              f"{' + E='+str(self.enc_dim) if self.use_vision else ''})")

        print(f"(Student) Actor MLP: {self.actor}")
        print("(Student) Train Type: ", self.train_type)

        print(f"(Student) Actor MLP: {self.actor}")
        print("(Student) Train Type: ", self.train_type)

        # Action noise / distribution
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    # ---------- Utilities ----------
    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        """Update the Normal policy with the provided observations (already concatenated)."""
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    # ---------- Observation parsing ----------
    def parse_observations(self, obs: torch.Tensor):
        """
        obs layout (fixed order):
          [0 : P) -> proprio
          [P : P+H) -> height scalars (optional, if use_height)
          [P+H : P+H+D) -> raw depth flattened (optional, if use_vision)
        Returns:
          proprio: [B, P]
          depth  : [B, 1, enc_H, enc_W] or None
          height : [B, Hdim] or None
        """
        B, T = obs.shape
        idx = 0

        # 1) Proprio
        proprio = obs[:, idx : idx + self.proprio_dim]
        idx += self.proprio_dim

        # 2) Height (optional)
        height = None
        if self.use_height and self.height_obs_dim > 0:
            height = obs[:, idx : idx + self.height_obs_dim]
            idx += self.height_obs_dim

        # 3) Depth (optional)
        depth = None
        if self.use_vision:
            expected = self.D
            depth_flat = obs[:, idx : idx + expected]  # safe slicing even if expected=0
            if depth_flat.numel() != B * expected:
                raise RuntimeError(
                    f"Depth slice mismatch: expected {expected} per sample but got {depth_flat.shape[1] if depth_flat.ndim==2 else '??'}. "
                    f"Check env.num_observations / flags: use_vision_in_actor={self.use_vision}"
                )

            # Restore to (B,1,H,W)
            depth = depth_flat.view(B, 1, self.H, self.W).contiguous()

            # --- Training-time preprocessing ---
            # Isaac depth uses -Z -> negate to positive distance
            d = (-depth[:, 0, :, :]).contiguous()

            # Replace invalid values with 'far'
            d.nan_to_num_(posinf=self.depth_far, neginf=self.depth_far)

            # Clip to [near, far]
            d.clamp_(min=self.depth_near, max=self.depth_far)

            # Normalize to [0,1] (near=1, far=0)
            scale = (self.depth_far - self.depth_near) + 1e-6
            d.sub_(self.depth_near).mul_(1.0 / scale)
            d.mul_(-1.0).add_(1.0)
            d.clamp_(0.0, 1.0)

            # Downsample to encoder input size
            depth = d.unsqueeze(1)                                  # [B,1,H,W]
            depth = F.interpolate(depth, size=(self.enc_H, self.enc_W),
                                  mode="bilinear", align_corners=False)  # [B,1,enc_H,enc_W]

            # Debug (one-shot)
            if not hasattr(self, "_dbg_slice_once"):
                print(f"[OBS] P={self.proprio_dim} Hdim={self.height_obs_dim} raw(H,W)=({self.H},{self.W}) "
                      f"enc(H,W)=({self.enc_H},{self.enc_W}) T={T}")
                print(f"[OBS] proprio={tuple(proprio.shape)} "
                      f"height={tuple(height.shape) if height is not None else None} "
                      f"depth={tuple(depth.shape) if depth is not None else None}")
                self._dbg_slice_once = True

        if self.use_height and not self.use_vision and not hasattr(self, "_dbg_ph_once"):
            print(f"[OBS-PH] P={self.proprio_dim} Hdim={self.height_obs_dim}")
            print(f"[OBS-PH] proprio={tuple(proprio.shape)} "
                  f"height={tuple(height.shape) if height is not None else None}")
            self._dbg_ph_once = True

        return proprio, depth, height

    # ---------- Policy forward helpers ----------
    def act(self, proprio, enc_depth=None, height=None):
        """
        Concatenate available features and produce action mean.
        enc_depth: encoded depth features (B, E) or None
        height   : raw height scalars (B, Hdim) or None
        """
        if enc_depth is not None and enc_depth.dim() == 1:
            enc_depth = enc_depth.unsqueeze(0)

        if height is not None and height.dim() == 1:
            height = height.unsqueeze(0)

        feats = [proprio]
        if height is not None:
            feats.append(height)
        if enc_depth is not None:
            feats.append(enc_depth)

        x = torch.cat(feats, dim=-1)

        first_linear = next(m for m in self.actor if isinstance(m, nn.Linear))
        assert x.shape[-1] == first_linear.in_features, \
            f"Actor input {x.shape[-1]} != expected {first_linear.in_features}"
        
        return self.actor(x)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, proprio, enc_depth=None, height=None):
        """Return the mean action via the exact same path as in training."""
        return self.act(proprio, enc_depth, height)


# ------------------------- Activation helper -------------------------
def get_activation(act_name):
    table = {
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "relu": nn.ReLU(),
        "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }
    return table.get(str(act_name).lower(), nn.ReLU())


# ------------------------- Generic MLP builder (kept) -------------------------
def construct_mlp_base(input_size, hidden_sizes):
    layers = []
    prev_size = input_size
    for out_size in hidden_sizes:
        layers.append(nn.Linear(int(prev_size), int(out_size)))
        layers.append(nn.ReLU())
        prev_size = out_size
    mlp = nn.Sequential(*layers) if len(layers) > 1 else layers[0]
    return mlp
