"""
PyTorch checkpoint loader.

Loads a state dict into a user-provided nn.Module and wraps it as a BaseModel.

load_checkpoint()/CheckpointModel are the generic path for a net whose forward() returns the
(dx, dy) actions directly. For this project's QMIX checkpoints use load_my_policy_checkpoint()
below.

Checkpoint format
-----------------
Both formats are supported:

  # Raw state dict
  torch.save(model.state_dict(), "model.pt")

  # Dict with a named key (e.g. from a trainer that saves extra metadata)
  torch.save({"policy_state": model.state_dict(), "step": 1000}, "model.pt")
"""

from __future__ import annotations

from typing import Any, Type

import numpy as np
import torch
import torch.nn as nn

from .base import BaseModel


class CheckpointModel(BaseModel):
    """
    Wraps a loaded nn.Module as a BaseModel.
    Batches all agents' observations together for a single forward pass.
    """

    def __init__(self, net: nn.Module, device: torch.device):
        self._net = net
        self._device = device

    def act(
        self,
        obs: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        agents = list(obs.keys())

        graphics = torch.tensor(
            np.stack([obs[a]["graphic"] for a in agents]),
            dtype=torch.float32,
            device=self._device,
        )
        # graphics from env: [B, H, W, C] → model expects [B, C, H, W]
        graphics = graphics.permute(0, 3, 1, 2)

        team_states = torch.tensor(
            np.stack([obs[a]["team_state"] for a in agents]),
            dtype=torch.float32,
            device=self._device,
        )
        agent_states = torch.tensor(
            np.stack([obs[a]["agent_states"] for a in agents]),
            dtype=torch.float32,
            device=self._device,
        )

        with torch.no_grad():
            raw: torch.Tensor = self._net(graphics, team_states, agent_states)

        actions = torch.clamp(raw, -1.0, 1.0).cpu().numpy()
        return {agent: actions[i] for i, agent in enumerate(agents)}


def load_checkpoint(
    model_class: Type[nn.Module],
    checkpoint_path: str,
    state_dict_key: str | None = "policy_state",
    device: str | torch.device = "cpu",
    **model_kwargs: Any,
) -> CheckpointModel:
    """
    Instantiate a model class and load weights from a checkpoint file.

    Parameters
    ----------
    model_class : Type[nn.Module]
        The nn.Module subclass to instantiate.
    checkpoint_path : str
        Path to the .pt checkpoint file.
    state_dict_key : str | None
        Key to extract the state dict from the checkpoint dict.
        Pass None if the checkpoint file is a raw state dict.
        Defaults to "policy_state".
    device : str | torch.device
        Device to load the model onto.
    **model_kwargs
        Keyword arguments forwarded to model_class().

    Returns
    -------
    CheckpointModel
        A BaseModel-compatible wrapper ready for inference.

    Examples
    --------
    `UserPolicy` is your own nn.Module (the README's competition `policy.py`), not this repo's
    MyPolicy; extra keyword arguments go to its constructor.

        # Checkpoint saved as raw state dict
        model = load_checkpoint(UserPolicy, "model.pt", state_dict_key=None,
                                n_graphic_channels=13, agent_state_size=12)

        # Checkpoint saved as {"policy_state": state_dict, ...}
        model = load_checkpoint(UserPolicy, "model.pt", n_graphic_channels=13, agent_state_size=12)
    """
    device = torch.device(device)
    net = model_class(**model_kwargs)

    raw = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if state_dict_key is not None:
        if not isinstance(raw, dict) or state_dict_key not in raw:
            raise KeyError(
                f"Key '{state_dict_key}' not found in checkpoint. "
                f"Available keys: {list(raw.keys()) if isinstance(raw, dict) else 'N/A'}. "
                f"Pass state_dict_key=None if the file is a raw state dict."
            )
        state_dict = raw[state_dict_key]
    else:
        state_dict = raw

    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    return CheckpointModel(net, device)


def load_my_policy_checkpoint(
    checkpoint_path: str,
    device: str | torch.device = "cpu",
    **model_kwargs: Any,
) -> Any:
    """
    Loads a QMIXTrainer checkpoint (train/qmix_trainer.py's `save()` format) into a fresh
    MyModel + MyPolicy, ready for run_match()/run_series() via the BaseModel.act() interface.

    load_checkpoint()/CheckpointModel above assume the wrapped net's forward() output IS the
    action tensor directly (`torch.clamp(net(...), -1, 1)`) — the right contract for a model
    that regresses continuous actions directly. MyModel is architecturally different: it
    returns a 5-tuple (q_values, quantile_values, tau, vision_latent, global_latent) of
    discrete per-unit IQN Q-values, not an action tensor, so CheckpointModel can't wrap it.
    MyPolicy already does the right conversion (per-unit Q-row selection by unit index ->
    argmax -> direction vector, see my_policy.py) but its constructor takes an already-built
    MyModel rather than a checkpoint path, so it doesn't fit load_checkpoint()'s generic
    "instantiate model_class, load state dict" flow either — this is the missing bridge
    between a trained checkpoint and the repo's standard deployment/competition path.

    Trainer checkpoints also carry mixer/optimizer/SPR state used only for training; none of
    that is relevant for inference, so only `policy_state` (the online net, not the EMA or
    target copy) is read here.

    `model_kwargs` go straight to MyModel(), whose constructor default hidden_size (256) is NOT
    the training config (QMIXConfig.hidden_size = 128). When hidden_size isn't passed it is read
    from the checkpoint's token-type embedding, so Run 11 checkpoints load without extra
    arguments. The returned MyPolicy has mask_walls=False, matching the default
    QMIXConfig.action_masking.
    """
    from .my_model import MyModel
    from .my_policy import MyPolicy

    device = torch.device(device)
    raw = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(raw, dict) or "policy_state" not in raw:
        raise KeyError(
            "Expected a QMIXTrainer checkpoint with a 'policy_state' key. "
            f"Available keys: {list(raw.keys()) if isinstance(raw, dict) else 'N/A'}."
        )

    # token_type_emb is nn.Embedding(4, hidden_size), so its weight's width is the model width.
    type_emb = raw["policy_state"].get("token_type_emb.weight")
    if "hidden_size" not in model_kwargs and type_emb is not None:
        model_kwargs["hidden_size"] = int(type_emb.shape[1])
    net = MyModel(**model_kwargs)
    net.load_state_dict(raw["policy_state"])
    net.to(device)
    net.eval()

    return MyPolicy(net, device=device)
