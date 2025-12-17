from legged_gym.lbc.actor_encoder_lbc import Actor, VisionEncoder
import torch
import torch.nn as nn
from torch.distributions import Normal


class CustomActorCritic(nn.Module):

    """
    PPO-compatible Actor-Critic with modality toggles:
      - P        : proprio only
      - PH       : proprio + height scalars
      - PD       : proprio + depth (encoded by CNN)
      - PHD      : proprio + height + depth
      - PRIV     : privileged-critic (handled at runner/env level; interface-ready)
    The class shares a vision encoder between actor and critic when needed.
    """
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        depth_near: float = 0.3,
        depth_far: float = 3.0,
        **kwargs,
    ):
        super().__init__()

        # 0) 참조 이전에 기본 속성 먼저 만들어 AttributeError 차단
        self.use_vision_in_actor  = False
        self.use_vision_in_critic = False
        self.detach_critic_vision = False
        self.use_height           = False
        self.height_obs_dim       = 0
        self.privileged_critic    = False

        # 1) GO2RoughCfg 값을 kwargs 기본값으로 주입 (반드시 어떤 참조보다 먼저)
        from legged_gym.envs.go2.go2_config import GO2RoughCfg as _Cfg

        kwargs.setdefault("camera_res",        tuple(getattr(_Cfg.env,     "camera_res", (424, 240))))
        kwargs.setdefault("proprio_dim",       int(getattr(_Cfg.env,      "proprio_dim", 48)))
        kwargs.setdefault("cnn_output_size",   int(getattr(_Cfg.env,   "cnn_output_size", 32)))
        _enc = getattr(_Cfg.env, "encoder_res", (84, 84))
        kwargs.setdefault("encoder_res",       (int(_enc[0]), int(_enc[1])))

        kwargs.setdefault("use_vision_in_actor",  bool(getattr(_Cfg.env,     "use_vision_in_actor",  False)))
        kwargs.setdefault("use_vision_in_critic", bool(getattr(_Cfg.env,     "use_vision_in_critic", False)))
        kwargs.setdefault("detach_critic_vision", bool(getattr(_Cfg.env,     "detach_critic_vision", False)))
        kwargs.setdefault("use_height",           bool(getattr(_Cfg.terrain, "measure_heights",      False)))
        kwargs.setdefault("height_obs_dim",       int(getattr(_Cfg.env,      "height_obs_dim",       0)))
        kwargs.setdefault("privileged_critic",    bool(getattr(_Cfg.env,     "privileged_critic",    False)))

        # 2) 이제 안전하게 self.*로 pop
        self.camera_res        = tuple(kwargs.pop("camera_res"))
        self.proprio_dim       = int(kwargs.pop("proprio_dim"))
        self.cnn_out           = int(kwargs.pop("cnn_output_size"))
        enc_h, enc_w           = kwargs.pop("encoder_res")
        self.encoder_res       = (int(enc_h), int(enc_w))

        self.use_vision_in_actor  = bool(kwargs.pop("use_vision_in_actor"))
        self.use_vision_in_critic = bool(kwargs.pop("use_vision_in_critic"))
        self.detach_critic_vision = bool(kwargs.pop("detach_critic_vision"))
        self.use_height           = bool(kwargs.pop("use_height"))
        self.height_obs_dim       = int(kwargs.pop("height_obs_dim"))
        self.privileged_critic    = bool(kwargs.pop("privileged_critic"))

        # (이 아래는 기존 코드 그대로 유지)
        act_fn = get_activation(activation)
        print(f"[CFG][AC] depth_near={depth_near} depth_far={depth_far}")
        # ... Actor/ Critic 생성 등 기존 흐름 계속 ...


        # --- (C) ACTOR 생성: 위에서 확정한 '지역 변수'만 전달 ---
        self.actor = Actor(
            num_actor_obs=0,                 # 호환용(내부에서 사용 안 함)
            num_actions=num_actions,
            proprio_dim=self.proprio_dim,
            cnn_output_size=self.cnn_out,
            camera_res=self.camera_res,
            encoder_res=self.encoder_res,
            depth_near=depth_near,
            depth_far=depth_far,
            activation=activation,
            init_noise_std=init_noise_std,
            use_vision_in_actor=self.use_vision_in_actor,
            use_height=self.use_height,           # ← 지역 변수
            height_obs_dim=self.height_obs_dim,       # ← 지역 변수
        )



# # ------------------------------ Custom Actor-Critic ------------------------------
# class CustomActorCritic(nn.Module):
#     """
#     PPO-compatible Actor-Critic with modality toggles:
#       - P        : proprio only
#       - PH       : proprio + height scalars
#       - PD       : proprio + depth (encoded by CNN)
#       - PHD      : proprio + height + depth
#       - PRIV     : privileged-critic (handled at runner/env level; interface-ready)
#     The class shares a vision encoder between actor and critic when needed.
#     """
#     is_recurrent = False

#     def __init__(
#         self,
#         num_actor_obs,
#         num_critic_obs,
#         num_actions,
#         actor_hidden_dims=[256, 256, 256],
#         critic_hidden_dims=[256, 256, 256],
#         activation="elu",
#         init_noise_std=1.0,
#         depth_near: float = 0.3,
#         depth_far: float = 3.0,
#         **kwargs,
#     ):
#         super().__init__()

#         # ---- Read cfg/kwargs (contract with go2_config.apply_modality) ----
#         # Camera & sizes
#         self.camera_res = tuple(kwargs.pop("camera_res", (424, 240)))   # (W, H)
#         self.proprio_dim = int(kwargs.pop("proprio_dim", 48))
#         self.cnn_out = int(kwargs.pop("cnn_output_size", 32))
#         enc_h, enc_w = kwargs.pop("encoder_res", (84, 84))
#         self.encoder_res = (int(enc_h), int(enc_w))

#         # Toggles (actor/critic vision; height usage)
#         self.use_vision_in_actor  = bool(kwargs.pop("use_vision_in_actor",  False))
#         self.use_vision_in_critic = bool(kwargs.pop("use_vision_in_critic", False))
#         self.detach_critic_vision = bool(kwargs.pop("detach_critic_vision", False))
#         self.use_height           = bool(kwargs.pop("use_height", False))
#         self.height_obs_dim       = int(kwargs.pop("height_obs_dim", 0))

#         # Privileged critic flag may be managed by the runner/env; kept here for completeness
#         self.privileged_critic    = bool(kwargs.pop("privileged_critic", False))

#         W, H = self.camera_res
#         act_fn = get_activation(activation)

#         print(f"[CFG][AC] depth_near={depth_near} depth_far={depth_far}")

#         # ---- Instantiate ACTOR (handles parsing of [proprio | height | depth_flat]) ----
#         # Actor internally creates a VisionEncoder only if use_vision_in_actor=True.
#         self.actor = Actor(
#             num_actor_obs=0,                      # kept for compat; not used inside Actor
#             num_actions=num_actions,
#             proprio_dim=self.proprio_dim,
#             cnn_output_size=self.cnn_out,
#             camera_res=(W, H),                    # raw depth slice H*W is based on this
#             encoder_res=self.encoder_res,         # downsampled size given to CNN
#             depth_near=depth_near,
#             depth_far=depth_far,
#             activation=activation,                # pass string; Actor resolves to module
#             init_noise_std=init_noise_std,
#             use_vision_in_actor=self.use_vision_in_actor,
#             use_height=self.use_height,
#             height_obs_dim=self.height_obs_dim,
#             **kwargs,
#         )




        # ---- Shared Vision Encoder ----
        # Prefer the encoder created by the Actor (if any). If actor is blind but critic wants vision,
        # create a dedicated encoder for the critic.
        self.vision_encoder = getattr(self.actor, "vision_encoder", None)
        if self.vision_encoder is None and self.use_vision_in_critic:
            H_enc, W_enc = self.encoder_res
            self.vision_encoder = VisionEncoder(image_size=[H_enc, W_enc, 1], cnn_output_size=self.cnn_out)

        # ---- Build CRITIC with dynamic input dims ----
        # critic input = proprio [P] + (height [Hdim]) + (encoded depth [E] if enabled)
        critic_in_dim = self.proprio_dim
        if self.use_height and self.height_obs_dim > 0:
            critic_in_dim += self.height_obs_dim
        if self.use_vision_in_critic:
            critic_in_dim += self.cnn_out

        critic_layers = [nn.Linear(critic_in_dim, critic_hidden_dims[0]), act_fn]
        for l in range(len(critic_hidden_dims) - 1):
            critic_layers += [nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]), act_fn]
        critic_layers += [nn.Linear(critic_hidden_dims[-1], 1)]
        self.critic = nn.Sequential(*critic_layers)

        # ---- Action distribution (diagonal Gaussian) ----
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

        # ---- Debug summary (one-shot) ----
        if not hasattr(self, "_shape_log_once"):
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"[CHECK] params total={total} trainable={trainable}")
            self._shape_log_once = True

    # ------------------------------ Forward: Actor ------------------------------
    def forward_actor(self, obs: torch.Tensor):
        """
        Parse obs -> (proprio, depth, height), encode depth if enabled,
        and run the actor MLP to get the mean action.
        """
        proprio, depth, height = self.actor.parse_observations(obs)

        # Encode depth when enabled; otherwise no depth features
        enc_depth = None
        if self.use_vision_in_actor:
            if depth is None or self.vision_encoder is None:
                raise RuntimeError("Actor expects depth but it is not provided. Check modality flags and num_observations.")
            enc_depth = self.vision_encoder(depth)

        mean = self.actor.act(proprio, enc_depth=enc_depth, height=height)
        return mean

    # ------------------------------ Forward: Critic ------------------------------
    def forward_critic(self, obs: torch.Tensor):
        """
        Parse obs in the same layout as the actor. If critic vision is enabled,
        concatenate encoded depth as well (optionally detached for stability).
        """
        proprio, depth, height = self.actor.parse_observations(obs)

        feats = [proprio]
        if self.use_height and height is not None:
            feats.append(height)

        if self.use_vision_in_critic:
            if depth is None or self.vision_encoder is None:
                raise RuntimeError("Critic expects depth but it is not provided. Check modality flags and critic_obs layout.")
            enc = self.vision_encoder(depth)
            if self.detach_critic_vision:
                enc = enc.detach()
            feats.append(enc)

        critic_in = torch.cat(feats, dim=-1)
        return self.critic(critic_in)

    # ------------------------------ PPO interface ------------------------------
    def update_distribution(self, observations):
        mean = self.forward_actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    # ===== RSL-RL compatibility properties (must exist on the class) =====
    @property
    def action_mean(self):
        if self.distribution is None:
            raise RuntimeError("distribution is None. Call act()/update_distribution() first.")
        return self.distribution.mean

    @property
    def action_std(self):
        if self.distribution is None:
            raise RuntimeError("distribution is None. Call act()/update_distribution() first.")
        return self.distribution.stddev

    @property
    def entropy(self):
        if self.distribution is None:
            raise RuntimeError("distribution is None. Call act()/update_distribution() first.")
        return self.distribution.entropy().sum(dim=-1)
    # ====================================================================

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations, **kwargs):
        """
        Critic forward path. For privileged-critic setups, the runner may pass a
        different critic_observations layout; in that case, adapt this method
        (or provide a separate evaluate_privileged) accordingly.
        """
        return self.forward_critic(critic_observations)

    # Deterministic policy head for rollouts/play
    @torch.jit.export
    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return self.forward_actor(observations)

    # RSL-RL compatibility; no recurrent state used here
    def reset(self, dones=None):
        try:
            if hasattr(self.vision_encoder, "reset") and callable(self.vision_encoder.reset):
                self.vision_encoder.reset(dones)
        except Exception:
            pass
        try:
            if hasattr(self.actor, "reset") and callable(self.actor.reset):
                self.actor.reset(dones)
        except Exception:
            pass
        return


# ------------------------------ Activation helper ------------------------------
def get_activation(act_name):
    """Return an activation module by name (default: ReLU)."""
    table = {
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "relu": nn.ReLU(),
        "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }
    return table.get(str(act_name).lower(), nn.ReLU())
