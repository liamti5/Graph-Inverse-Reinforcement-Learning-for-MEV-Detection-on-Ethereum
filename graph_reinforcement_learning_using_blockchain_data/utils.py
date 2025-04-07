import os
import pickle
from datetime import datetime, timedelta
from random import uniform

import torch as th
from imitation.rewards import reward_nets
from stable_baselines3.common import base_class
from stable_baselines3.ppo import PPO


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