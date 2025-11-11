############################################################
# Copyright (c) 2025 Cedric Grothues, Bielefeld University #
############################################################

from pathlib import Path

import gymnasium as gym
import hydra
import magbotsim
import torch
from omegaconf import DictConfig
from torch import nn
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP

from utils import create_envs # TODO: insert your code for creating environments
from utils import create_sac_agent # TODO: insert your code for creating sac agents

gym.register_envs(magbotsim)

class DeterministicPolicy(torch.nn.Module):
    """Use the mean of the learned Gaussian policy for deterministic inference."""

    def __init__(self, mlp: MLP):
        super().__init__()
        self.mlp = mlp

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = self.mlp(observation)
        return torch.tanh(x)


def extract_policy_mlp(model: nn.ModuleList) -> MLP:
    return model[0].module[0].module[0].module  # type: ignore


def split_mu_logstd_layer(mlp: MLP, mu_dim: int = 6) -> None:
    """Trim the last layer to only output mu.

    Our actor outputs both mu and log_std from a single linear layer.
    For deterministic inference, we only need mu. More importantly, if we
    export the full layer and slice in ONNX, it creates operations unsupported
    by TwinCAT. By trimming the layer to only output mu, we get a cleaner ONNX
    graph that TwinCAT can execute.
    """
    layer = mlp[-1]

    if not isinstance(layer, nn.Linear):
        raise ValueError(f"Expected nn.Linear, got {type(layer)}.")

    mu_logstd_dim = mu_dim * 2
    if layer.out_features != mu_dim * 2:
        raise ValueError(
            f"Expected layer with {mu_logstd_dim}, got {layer.out_features}."
        )

    mu_layer = nn.Linear(layer.in_features, mu_dim)
    mu_layer.weight.data = layer.weight.data[:mu_dim, :]
    mu_layer.bias.data = layer.bias.data[:mu_dim]

    mlp[-1] = mu_layer


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    state_dict = torch.load(Path("frames=3000000.pt"))

    _, eval_env = create_envs(cfg, None)
    model, _ = create_sac_agent(cfg, eval_env)
    model.load_state_dict(state_dict["model"])

    mlp = extract_policy_mlp(model)
    mlp.requires_grad_(False)
    mlp.eval()

    split_mu_logstd_layer(mlp)

    fake_td = eval_env.base_env.fake_tensordict()  # type: ignore
    # NOTE: the keys are sorted alphabetically. Ensure that your observations are correctly sorted in TwinCAT!
    dummy_input = torch.cat(
        [
            fake_td["achieved_goal"].float(),
            fake_td["desired_goal"].float(),
            fake_td["observation"].float(),
        ]
    )

    policy = DeterministicPolicy(mlp)

    with set_exploration_type(ExplorationType.DETERMINISTIC):
        torch.onnx.export(
            policy,
            (dummy_input,),
            Path("exported_policy.onnx"),
            input_names=["observation"],
            output_names=["action"],
        )


if __name__ == "__main__":
    main()
