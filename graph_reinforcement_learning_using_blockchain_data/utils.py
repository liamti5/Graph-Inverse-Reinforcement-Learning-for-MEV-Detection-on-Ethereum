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
import pandas as pd


def save_model(
    learner: base_class.BaseAlgorithm,
    reward_net: reward_nets,
    stats: dict,
    path: str,
    ts: datetime,
) -> None:
    """
    Saves the model components required for later restoration or evaluation.

    This function saves the given components of a model to the specified path
    to allow for persistence of training processes and results. It stores the
    learner, the reward network, and training statistics in files within the
    provided directory. If the directory does not exist, it is created.

    :param learner: The primary model that implements a base algorithm
    :param reward_net: The neural network that represents the reward function
    :param stats: Dictionary containing training statistics
    :param path: Directory path where components should be saved
    :param ts: Timestamp to label the saved components
    :return: None
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
    Pads a tensor to a specified maximum length along the second dimension. If the current
    dimension is less than the maximum length, it will pad the tensor with zeros until the
    maximum length is reached. Padding is applied to the end of the tensor.

    :param x: A tensor of arbitrary shape with at least two dimensions. Padding will be
        applied to the second dimension.
    :param max_length: The desired maximum length along the second dimension.
    :return: A tensor padded to the specified maximum length along the second dimension.
    :rtype: torch.tensor
    """
    current_dim = x.size(1)
    if current_dim < max_length:
        pad_amount = max_length - current_dim

        x = F.pad(x, (0, pad_amount), "constant", 0)
    return x


class MLflowOutputFormat(KVWriter):
    """
    Manages writing key-value pairs as metrics to MLflow, with specific handling
    for excluded keys and step tracking. This class inherits from `KVWriter` and
    extends its functionality to integrate with the MLflow tracking framework.
    It allows logging of scalar values to MLflow and ignores keys with specified
    exclusions involving "mlflow".
    """

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:
        """
        Logs metrics to MLflow, filtering out specific keys based on exclusions and conditions.
        Exclusion check ensures that metrics associated with keys marked as excluded for "mlflow"
        are skipped. Metrics are only logged if they are scalar types and not strings, ensuring
        numeric metrics are processed correctly.

        :param key_values: Dictionary of key-value pairs where the key is the metric name
            and the value is the metric value that needs to be logged.
            Expected values in the dictionary include scalar types (e.g., int, float) and strings.
        :param key_excluded: Dictionary of exclusion criteria for metrics. The key relates
            to the metric name, and the value defines exclusion rules for the corresponding
            metric. For the context of this method, a value containing "mlflow" prevents
            the logging of that metric.
        :param step: An integer representing the step or epoch index timestamp associated
            with the metrics being logged. Default value is set to 0.
        :return: This method does not return any value.
        """
        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):

            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)


def load_dataframes(file_paths: list) -> pd.DataFrame:
    dataframes_dict = {}
    for path in file_paths:
        key = str(path).split("/")[-1].split(".")[0]
        dataframes_dict[key] = pd.read_csv(path)

    return dataframes_dict
