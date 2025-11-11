##########################################################
# Copyright (c) 2025 Lara Bergmann, Bielefeld University #
##########################################################

from pathlib import Path

import gymnasium as gym
import magbotsim
import numpy as np
import torch as th
from stable_baselines3 import SAC

gym.register_envs(magbotsim)

# NOTE: 
# Stable-Baselines3 processes dictionary observations and sorts the keys alphabetically. Ensure that your observations are correctly sorted in TwinCAT!

# init env
env_id = "LongHorizonGlobalTrajectoryPlanningEnv"
num_movers = 3
num_cycles = 120
learn_jerk = False
collision_params = {
    "shape": "box",
    "size": np.array([0.09, 0.09]),
    "offset": 0.0,
    "offset_wall": 0.0,
}

env_kwargs = {
    "layout_tiles": np.ones((4, 3)),
    "num_movers": num_movers,
    "show_2D_plot": False,
    "num_cycles": num_cycles,
    "learn_jerk": learn_jerk,
}


class OnnxablePolicy(th.nn.Module):
    def __init__(self, latent_pi: th.nn.Module, mu: th.nn.Linear):
        super().__init__()
        self.latent_pi = latent_pi
        self.mu = mu

    def forward(self, observation: th.Tensor) -> th.Tensor:
        # NOTE: You may have to postprocess (unnormalize) actions
        # to the correct bounds (see commented code below)
        x = self.latent_pi(observation)
        mu = th.tanh(self.mu(x))
        return mu

# load trained SAC policy
env = gym.make(env_id + "-v0", **env_kwargs)

model = SAC.load(
    path=Path(__file__).parent.joinpath("best_model.zip"), # TODO: update path to your model
    env=env,
    device="cpu",
)

# to onnx
onnxable_model = OnnxablePolicy(
    model.policy.actor.latent_pi,
    model.policy.actor.mu,
)

dummy_input = th.randn(1, 8 * num_movers if learn_jerk else 6 * num_movers)

onnx_path = Path("best_model.onnx") # TODO: update path
th.onnx.export(
    onnxable_model,
    dummy_input,
    onnx_path,
    opset_version=17,
    input_names=["input"],
)
