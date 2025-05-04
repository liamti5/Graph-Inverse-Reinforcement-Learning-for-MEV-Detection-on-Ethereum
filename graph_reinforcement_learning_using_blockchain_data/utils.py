import os
import pickle
from datetime import datetime
from typing import Dict, Any, Union, Tuple

import mlflow
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
from imitation.rewards import reward_nets
from stable_baselines3.common import base_class
from stable_baselines3.common.logger import KVWriter


def save_model(
    learner: base_class.BaseAlgorithm,
    reward_net: reward_nets,
    stats: dict,
    path: str,
    ts: datetime,
) -> None:
    """
    Saves the model to the specified path.

    Args:
        learner: Learner policy.
        reward_net: Reward network.
        stats: Training statistics.
        path: Path to save the model to.
        ts: Timestamp to include in the file names.
    """
    # Create the directory if it does not exist
    os.makedirs(path, exist_ok=True)

    # Save the learner
    learner.save(f"{path}/{ts}_learner")

    # Save the reward net
    th.save(reward_net, f"{path}/{ts}_reward_nn")

    # Save the training statistics
    with open(f"{path}/{ts}_stats.pkl", "wb") as f:
        pickle.dump(stats, f)


def pad_features(x: torch.tensor, max_length: int) -> torch.tensor:
    """
    Pads features (x) of a graph to a max length.

    :param x:
    :param max_length:
    :return:
    """
    current_dim = x.size(1)
    if current_dim < max_length:
        pad_amount = max_length - current_dim

        x = F.pad(x, (0, pad_amount), "constant", 0)
    return x


class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:

        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):

            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)
