import argparse
import ast
import os
from typing import Any, Dict, Tuple

import gymnasium as gym
import mlflow
import numpy as np
import pandas as pd
import torch
from imitation.algorithms.adversarial.airl import AIRL
from imitation.data.rollout import flatten_trajectories
from imitation.data.types import Trajectory, Transitions
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util import logger as imitation_logger
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecCheckNan
from stable_baselines3.ppo import MlpPolicy

import graph_reinforcement_learning_using_blockchain_data as grl
from graph_reinforcement_learning_using_blockchain_data import config

config.load_dotenv()
mlflow.set_tracking_uri(uri=config.MLFLOW_TRACKING_URI)


class AirlTrainer:
    """
    Trainer class for Adversarial Inverse Reinforcement Learning (AIRL) using PPO as the generator.
    """

    def __init__(
        self,
        venv: gym.Env,
        trajectories: Transitions,
        device: torch.device = torch.device("mps"),
        **kwargs: Dict[str, Any],
    ):
        """
        Initializes the AIRL trainer with the given environment, expert trajectories, and hyperparameters.

        :param venv: The vectorized environment.
        :param trajectories: Expert trajectories for imitation learning.
        :param device: Torch device to use for training.
        :param kwargs: Additional keyword arguments for PPO and AIRL.
        """
        hier_logger = imitation_logger.configure()
        hier_logger.default_logger.output_formats.append(grl.MLflowOutputFormat())

        self.n_steps = kwargs.get("n_steps")
        self.batch_size = kwargs.get("batch_size")
        self.total_timesteps = kwargs.get("total_timesteps")

        self.learner = PPO(
            env=venv,
            policy=MlpPolicy,
            policy_kwargs=kwargs.get("policy_kwargs"),
            learning_rate=kwargs.get("learning_rate"),
            n_steps=kwargs.get("n_steps"),
            batch_size=kwargs.get("batch_size"),
            n_epochs=kwargs.get("n_epochs"),
            gamma=kwargs.get("gamma"),
            gae_lambda=kwargs.get("gae_lambda"),
            clip_range=kwargs.get("clip_range"),
            clip_range_vf=kwargs.get("clip_range_vf"),
            normalize_advantage=kwargs.get("normalize_advantage"),
            ent_coef=kwargs.get("ent_coef"),
            vf_coef=kwargs.get("vf_coef"),
            max_grad_norm=kwargs.get("max_grad_norm"),
            use_sde=kwargs.get("use_sde"),
            sde_sample_freq=kwargs.get("sde_sample_freq"),
            verbose=kwargs.get("verbose"),
            seed=42,
            device=device,
        )
        self.reward_net = BasicShapedRewardNet(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            normalize_input_layer=RunningNorm,
        )
        self.trainer = AIRL(
            demonstrations=trajectories,
            demo_batch_size=kwargs.get("demo_batch_size"),
            demo_minibatch_size=kwargs.get("demo_minibatch_size"),
            n_disc_updates_per_round=kwargs.get("n_disc_updates_per_round"),
            gen_train_timesteps=kwargs.get("gen_train_timesteps"),
            gen_replay_buffer_capacity=kwargs.get("gen_replay_buffer_capacity"),
            venv=venv,
            gen_algo=self.learner,
            reward_net=self.reward_net,
            allow_variable_horizon=kwargs.get("allow_variable_horizon"),
            disc_opt_kwargs=kwargs.get("disc_opt_kwargs"),
            custom_logger=hier_logger,
        )

    def train(self, experiment_name: str) -> Tuple[AIRL, BasicShapedRewardNet, PPO]:
        """
        Trains the AIRL model and logs artifacts to MLflow.

        :param experiment_name: Name of the MLflow experiment.
        :return: Tuple containing the AIRL trainer, reward network, and PPO learner.
        """
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            mlflow.log_param("n_steps", self.n_steps)
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("total_timesteps", self.total_timesteps)

            self.trainer.train(total_timesteps=self.total_timesteps)

            learner_dir = config.MODELS_DIR / "learner"
            reward_net_dir = config.MODELS_DIR / "reward_net"

            self.learner.save(learner_dir)
            torch.save(self.reward_net, reward_net_dir)

            mlflow.log_artifact(str(learner_dir) + ".zip")
            mlflow.log_artifact(reward_net_dir)

            os.remove(str(learner_dir) + ".zip")
            os.remove(reward_net_dir)

            mlflow.end_run()

        return self.trainer, self.reward_net, self.learner


def calibrate(
    validation_trajectories: Transitions, reward_net: BasicShapedRewardNet
) -> Tuple[float, float]:
    """
    Calibrates the reward network outputs to have zero mean and unit variance.

    :param validation_trajectories: Trajectories to use for calibration.
    :param reward_net: The reward network to calibrate.
    :return: Tuple (alpha, beta) for scaling and shifting the reward outputs.
    """
    states, obs, next_states, dones = (
        validation_trajectories.obs,
        validation_trajectories.acts,
        validation_trajectories.next_obs,
        validation_trajectories.dones,
    )

    outs = reward_net.predict(states, obs, next_states, dones)
    mean, std = outs.mean(), outs.std()
    target_mean, target_std = 0.0, 1.0
    alpha = target_std / std
    beta = target_mean - alpha * mean
    return alpha, beta


def register_envs(
    df_dict: Dict[str, pd.DataFrame],
    mflow_gnn_path: str,
    label: str,
    name: str = None,
    device: torch.device = torch.device("mps"),
) -> None:
    """
    Registers custom Gym environments for each dataframe in the dictionary.

    :param df_dict: Dictionary mapping keys to dataframes.
    :param mflow_gnn_path: Path to the GNN model in MLflow.
    :param name: Optional name for the environment.
    :param label: Label for the environment.
    :param device: Torch device to use.
    """
    for k, df in df_dict.items():
        gym_id = f"gymnasium_env/TransactionGraphEnv_{k}{name}_v2"
        gym.envs.register(
            id=gym_id,
            entry_point=grl.TransactionGraphEnvV2,
            kwargs={
                "df": df,
                "device": device,
                "label": int(label),
                "model_uri": mflow_gnn_path,
                "observation_space_dim": 128,
            },
        )


def make_venvs(df_dict: Dict[str, pd.DataFrame], name: str = None) -> Dict[str, gym.Env]:
    """
    Creates vectorized environments for each dataframe.

    :param df_dict: Dictionary mapping keys to dataframes.
    :param name: Optional name for the environment.
    :return: Dictionary mapping keys to vectorized Gym environments.
    """
    venvs = {}
    for k, df in df_dict.items():
        env_id = f"gymnasium_env/TransactionGraphEnv_{k}{name}_v2"
        venv = make_vec_env(
            env_id,
            rng=np.random.default_rng(42),
            n_envs=1,
            post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
            parallel=False,
        )
        VecCheckNan(venv, raise_exception=True)
        venv.reset()
        venvs[k] = venv
    return venvs


def extract_trajectories(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Transitions]:
    """
    Extracts and flattens trajectories from the provided dataframes.

    :param dataframes: Dictionary mapping keys to dataframes.
    :return: Dictionary mapping keys to flattened trajectory transitions.
    """
    trajectories = {}
    for k, df in dataframes.items():
        traj_list = []
        for account, group in df.groupby("from"):
            group = group.sort_values("blockNumber")
            obs_list = group["embeddings"].tolist() + [np.zeros(128, dtype=np.float32)]
            trajectory_dict = {
                "obs": np.stack(obs_list),
                "acts": np.array(group["action"].tolist()),
                "label": group["label"].iloc[0],
            }
            trajectory = Trajectory(
                obs=trajectory_dict["obs"], acts=trajectory_dict["acts"], infos=None, terminal=True
            )
            traj_list.append(trajectory)

        trajectories[k] = flatten_trajectories(traj_list)

    return trajectories


def run_airl_pipeline(
    data_class: str,
    embeddings: str,
    experiment_name: str,
    mlflow_gnn_path: str,
    kwargs: Dict[str, Any],
) -> AirlTrainer:
    """
    Run the Adversarial Inverse Reinforcement Learning (AIRL) pipeline.

    This function loads data, processes embeddings, extracts trajectories,
    registers environments, creates vectorized environments, and trains
    an AIRL model.

    :param data_class: The class of data to use ("0" or "1").
    :param embeddings: Path to the embeddings file (relative to PROCESSED_DATA_DIR/AIRL).
    :param experiment_name: Name of the MLflow experiment for logging.
    :param mlflow_gnn_path: Path to the GNN model in MLflow.
    :param kwargs: Dictionary containing hyperparameters for the AIRL trainer.
    :return: The trained AIRL trainer instance.
    """
    logger.info("Reading data ...")
    file_paths_eth_data = [
        config.PROCESSED_DATA_DIR / "AIRL" / f"airl_{data_class}_train.csv",
    ]

    embeddings = pd.read_csv(config.PROCESSED_DATA_DIR / "AIRL" / embeddings)
    embeddings["embeddings"] = embeddings["embeddings"].apply(
        lambda x: np.array(ast.literal_eval(x), dtype=np.float32)
    )

    dataframes = grl.load_dataframes(file_paths_eth_data)

    for k, df in dataframes.items():
        dataframes[k] = pd.merge(
            df, embeddings, how="inner", left_on="transaction_hash", right_on="transactionHash"
        )

    logger.success("Read data successfully.")

    trajectories = extract_trajectories(dataframes)

    register_envs(dataframes, mlflow_gnn_path, data_class)

    venvs = make_venvs(dataframes)

    airl_trainer = AirlTrainer(
        venvs[f"airl_{data_class}_train"],
        trajectories[f"airl_{data_class}_train"],
        torch.device("mps"),
        **kwargs,
    )
    airl_trainer.train(experiment_name)

    return airl_trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_class", choices=["0", "1"])
    parser.add_argument(
        "--embeddings",
    )
    parser.add_argument("--experiment_name")
    parser.add_argument("--mlflow_gnn_path")

    args = parser.parse_args()
    assert args.data_class, "Please specify the data class."
    assert args.embeddings, "Please specify the embeddings file to use."
    assert args.experiment_name, "Please specify the experiment name."
    assert args.mlflow_gnn_path, "Please specify the mlflow path to the gnn model."

    kwargs_class1 = {
        # PPO hyper-params
        "learning_rate": 2e-4,  # Learning rate can be a function of progress
        "batch_size": 256,  # Mini batch size for each gradient update
        "n_epochs": 10,  # N of epochs when optimizing the surrogate loss
        "gamma": 0.99,  # Discount factor
        "gae_lambda": 0.95,  # Generalized advantage estimation
        "clip_range": 0.25,  # Clipping parameter
        "ent_coef": 0.05,  # Entropy coefficient for the loss calculation
        "vf_coef": 1.0,  # Value function coef. for the loss calculation
        "max_grad_norm": 0.5,  # The maximum value for the gradient clipping
        "verbose": 0,  # Verbosity level: 0 no output, 1 info, 2 debug
        "normalize_advantage": False,  # Whether to normalize or not the advantage
        "use_sde": False,  # Use State Dependent Exploration
        "sde_sample_freq": -1,  # SDE - noise matrix frequency (-1 = disable)
        "policy_kwargs": {"use_expln": True},
        # PPO roll-out length
        "gen_train_timesteps": 2048,  # N steps in the environment per one round
        "n_steps": 2048,
        # AIRL discriminator
        "n_disc_updates_per_round": 8,  # N discriminator updates per one round
        "disc_opt_kwargs": {
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
        "demo_minibatch_size": 128,  # N samples in minibatch for one discrim. update
        "demo_batch_size": 640,  # N samples in the batch of expert data (batch)
        "allow_variable_horizon": True,
        "gen_replay_buffer_capacity": None,
        # total training
        "total_timesteps": 2048 * 100,
    }

    kwargs_class0 = {
        # PPO hyper-params
        "learning_rate": 5e-4,  # Learning rate can be a function of progress
        "batch_size": 256,  # Mini batch size for each gradient update
        "n_epochs": 10,  # N of epochs when optimizing the surrogate loss
        "gamma": 0.9,  # Discount factor
        "gae_lambda": 0.8,  # Generalized advantage estimation
        "clip_range": 0.2,  # Clipping parameter
        "ent_coef": 0.005,  # Entropy coefficient for the loss calculation
        "vf_coef": 0.5,  # Value function coef. for the loss calculation
        "max_grad_norm": 0.5,  # The maximum value for the gradient clipping
        "verbose": 0,  # Verbosity level: 0 no output, 1 info, 2 debug
        "normalize_advantage": True,  # Whether to normalize or not the advantage
        "use_sde": False,  # Use State Dependent Exploration
        "sde_sample_freq": -1,  # SDE - noise matrix frequency (-1 = disable)
        "policy_kwargs": {"use_expln": True},
        # PPO roll-out length
        "gen_train_timesteps": 2048,  # N steps in the environment per one round
        "n_steps": 1024,
        # AIRL discriminator
        "n_disc_updates_per_round": 6,  # N discriminator updates per one round
        "disc_opt_kwargs": {
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
        "demo_minibatch_size": 128,  # N samples in minibatch for one discrim. update
        "demo_batch_size": 512,  # N samples in the batch of expert data (batch)
        "allow_variable_horizon": True,
        "gen_replay_buffer_capacity": None,
        # total training
        "total_timesteps": 1024 * 200,
    }

    kwargs_map = {
        "0": kwargs_class0,
        "1": kwargs_class1,
    }

    run_airl_pipeline(
        args.data_class,
        args.embeddings,
        args.experiment_name,
        args.mlflow_gnn_path,
        kwargs_map[args.data_class],
    )


if __name__ == "__main__":
    main()
