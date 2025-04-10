import os
import sys
from typing import Dict, Any, Union, Tuple

import gymnasium as gym
import mlflow
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from graph_reinforcement_learning_using_blockchain_data import config, utils

mlflow.set_tracking_uri(uri=config.MLFLOW_TRACKING_URI)

np.random.seed(42)

loggers = Logger(
    folder=None,
    output_formats=[HumanOutputFormat(sys.stdout), utils.MLflowOutputFormat()],
)


class BlockchainEnv(gym.Env):
    def __init__(
        self,
        df_trx: pd.DataFrame,
        max_num_addresses: int,
        embedding_dim: int,
        feature_cols: list,
        alpha: float,
    ):
        self.df = df_trx.sort_values(by=["blockNumber"])
        self.max_num_addresses = max_num_addresses
        self.embedding_dim = embedding_dim
        self.feature_cols = feature_cols
        self.feature_dim = len(feature_cols)
        self.alpha = alpha
        self.observation_space = gym.spaces.Dict(
            {
                "states": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(max_num_addresses, embedding_dim),
                    dtype=np.float32,
                ),
                "mask": gym.spaces.Box(low=0, high=1, shape=(max_num_addresses,), dtype=np.int8),
            }
        )
        self.action_space = gym.spaces.MultiBinary(max_num_addresses)
        self.address_embeddings = {}
        self.blocks = self.df["blockNumber"].unique()
        self.current_block_index = 0
        self.current_active_addresses = None
        self.current_ground_truth = None
        self.current_block_df = None

    def reset(self, seed=42, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_block_index = 0
        self.address_embeddings = {}
        return self._get_observation(), {}

    def _get_observation(self):
        if self.current_block_index >= len(self.blocks):
            return None
        block_num = self.blocks[self.current_block_index]
        df_block = self.df[self.df["blockNumber"] == block_num]

        active_addresses = df_block["from"].unique()
        active_embeddings = []
        for addr in active_addresses:
            if addr not in self.address_embeddings:
                self.address_embeddings[addr] = np.full(
                    (self.embedding_dim,), -1.0, dtype=np.float32
                )
            active_embeddings.append(self.address_embeddings[addr])
        active_embeddings = np.array(active_embeddings)  # shape: (num_active, embedding_dim)
        num_active = active_embeddings.shape[0]

        padded_states = np.zeros((self.max_num_addresses, self.embedding_dim), dtype=np.float32)
        padded_states[:num_active] = active_embeddings
        mask = np.zeros((self.max_num_addresses,), dtype=np.int8)
        mask[:num_active] = 1

        self.current_active_addresses = active_addresses

        ground_truth = []
        for addr in active_addresses:
            addr_df = df_block[df_block["from"] == addr]
            label = int((addr_df["label"] == 1).any())
            ground_truth.append(label)
        self.current_ground_truth = np.array(ground_truth, dtype=np.int8)
        self.current_block_df = df_block

        return {"states": padded_states, "mask": mask}

    def _aggregate_features(self, addr):
        addr_df = self.current_block_df[self.current_block_df["from"] == addr]
        features = addr_df[self.feature_cols].mean().values.astype(np.float32)
        return features

    def step(self, action):
        num_active = len(self.current_active_addresses)
        valid_actions = action[:num_active]
        # Compare agent's predictions with ground truth.
        rewards = (valid_actions == self.current_ground_truth).astype(np.float32)
        total_reward = float(np.sum(rewards))

        # For addresses with a correct prediction, update their embedding.
        for i, addr in enumerate(self.current_active_addresses):
            if rewards[i] == 1:
                agg_features = self._aggregate_features(addr)  # shape: (feature_dim,)
                # Project the aggregated features into the embedding space.
                # Here we use a fixed random projection.
                if not hasattr(self, "proj"):
                    self.proj = np.random.randn(self.feature_dim, self.embedding_dim).astype(
                        np.float32
                    )
                projected_features = agg_features.dot(self.proj)  # shape: (embedding_dim,)
                # Soft update: blend the new projected features with the current embedding.
                self.address_embeddings[addr] = (
                    self.alpha * projected_features
                    + (1 - self.alpha) * self.address_embeddings[addr]
                )
        # Move to the next block.
        self.current_block_index += 1
        done = self.current_block_index >= len(self.blocks)
        obs = self._get_observation() if not done else None
        return obs, total_reward, done, False, {}


# class PPOActor(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim=64, action_dim=2):
#         super(PPOActor, self).__init__()
#         self.fc1 = nn.Linear(embedding_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, action_dim)
#
#     def forward(self, states):
#         # states: shape (max_addresses, embedding_dim)
#         x = F.relu(self.fc1(states))
#         logits = self.fc2(x)  # shape: (max_addresses, action_dim)
#         return logits
#
#
# # Critic network to estimate state values.
# class PPOCritic(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim=64):
#         super(PPOCritic, self).__init__()
#         self.fc1 = nn.Linear(embedding_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, 1)
#
#     def forward(self, states):
#         x = F.relu(self.fc1(states))
#         values = self.fc2(x)  # shape: (max_addresses, 1)
#         return values
#
#
# # Combined PPO Agent that uses both the actor and the critic.
# class PPOAgent(nn.Module):
#     def __init__(self, embedding_dim, action_dim=2):
#         super(PPOAgent, self).__init__()
#         self.actor = PPOActor(embedding_dim, hidden_dim=64, action_dim=action_dim)
#         self.critic = PPOCritic(embedding_dim, hidden_dim=64)
#
#     def act(self, observation):
#         """
#         observation: a dict with keys 'states' and 'mask'
#         'states': numpy array of shape (max_addresses, embedding_dim)
#         'mask': numpy array of shape (max_addresses,) indicating which rows are active.
#         """
#         states = torch.tensor(
#             observation["states"], dtype=torch.float32
#         )  # shape: (max_addresses, embedding_dim)
#         mask = torch.tensor(observation["mask"], dtype=torch.float32)  # shape: (max_addresses,)
#
#         # Compute logits for each address.
#         logits = self.actor(states)  # shape: (max_addresses, action_dim)
#         action_probs = F.softmax(logits, dim=-1)  # probabilities for each address.
#         dist = torch.distributions.Categorical(action_probs)
#         actions = dist.sample()  # shape: (max_addresses,)
#
#         # Zero out actions for inactive addresses.
#         actions = actions * mask.long()
#         log_probs = dist.log_prob(actions)
#         entropy = dist.entropy()
#         # Critic returns state-value estimates for each address.
#         values = self.critic(states)  # shape: (max_addresses, 1)
#         return actions, log_probs, entropy, values


def run(
    vec_train_env,
    vec_test_env,
    max_num_addresses,
    experiment_name,
    n_steps,
    batch_size,
    total_timesteps,
    embedding_dim,
    alpha,
):
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_param("max_num_addresses", max_num_addresses)
        mlflow.log_param("embedding_dim", embedding_dim)
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("n_steps", n_steps)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("total_timesteps", total_timesteps)

        model = PPO(
            "MultiInputPolicy",
            vec_train_env,
            verbose=1,
            device="mps",
            n_steps=n_steps,
            batch_size=batch_size,
        )

        model.set_logger(loggers)

        model.learn(total_timesteps=total_timesteps, progress_bar=True, log_interval=1)

        accuracy = test(vec_test_env, model)
        print(f"Custom Test Accuracy: {accuracy:.3f}")
        mlflow.log_metric("test_accuracy", accuracy)

        model_path = f"{experiment_name}.zip"
        model.save(model_path)
        mlflow.log_artifact(model_path)
        os.remove(model_path)


def test(vec_test_env, agent):
    total_correct = 0
    total_predictions = 0
    obs = vec_test_env.reset()  # vec_test_env.reset() returns only the observations
    print(obs)
    done = False

    while not done:
        actions, _ = agent.predict(obs, deterministic=True)

        mask = obs[0]["mask"] if isinstance(obs, (list, tuple)) else obs["mask"]
        total_predictions += mask.sum()

        obs, rewards, done, _ = vec_test_env.step(actions)
        total_correct += rewards[0]

    accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    return accuracy


def main():
    df = pd.read_csv(config.FLASHBOTS_Q2_DATA_DIR / "features_edges_multiocc.csv")
    max_addresses = df.groupby("blockNumber")["from"].nunique().max()

    unique_blocks = np.sort(df["blockNumber"].unique())
    split_index = int(0.8 * len(unique_blocks))
    train_blocks = unique_blocks[:split_index]
    test_blocks = unique_blocks[split_index:]

    df_train = df[df["blockNumber"].isin(train_blocks)]
    df_test = df[df["blockNumber"].isin(test_blocks)]

    print(f"Train blocks: {len(train_blocks)}, Test blocks: {len(test_blocks)}")

    max_num_addresses = max_addresses
    embedding_dim = 128
    feature_cols = [
        "gasUsed",
        "cumulativeGasUsed",
        "effectiveGasPrice",
        "status",
        "fee",
        "num_logs",
        "dummy_0xd78ad95f",
        "dummy_0xe1fffcc4",
        "dummy_0x908fb5ee",
        "dummy_0xe9149e1b",
        "dummy_0x1c411e9a",
        "dummy_0x9d9af8e3",
        "dummy_0x19b47279",
        "dummy_0x8201aa3f",
        "dummy_0xc42079f9",
        "dummy_0xddf252ad",
        "dummy_0x17307eab",
        "dummy_0xddac4093",
        "dummy_0x8c5be1e5",
        "dummy_0x7fcf532c",
    ]
    alpha = 0.3

    train_env = BlockchainEnv(df_train, max_num_addresses, embedding_dim, feature_cols, alpha)
    test_env = BlockchainEnv(df_test, max_num_addresses, embedding_dim, feature_cols, alpha)

    # train_env_flat = gym.wrappers.FlattenObservation(train_env)
    # test_env_flat = gym.wrappers.FlattenObservation(test_env)

    train_env = Monitor(train_env)
    test_env = Monitor(test_env)

    check_env(train_env)
    check_env(test_env)

    # Wrap your environment in a DummyVecEnv to make it compatible with SB3.
    vec_train_env = DummyVecEnv([lambda: train_env])
    vec_test_env = DummyVecEnv([lambda: test_env])

    # Set up MLflow experiment.
    mlflow.set_experiment("RL PPO soft-update embeddings v1")
    n_steps = 2048
    batch_size = 256
    total_timesteps = len(train_blocks) * 5

    with mlflow.start_run():
        # Log parameters.
        mlflow.log_param("max_num_addresses", max_num_addresses)
        mlflow.log_param("embedding_dim", embedding_dim)
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("n_steps", n_steps)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("total_timesteps", total_timesteps)

        # Create the PPO model using a standard MLP policy.
        model = PPO(
            "MultiInputPolicy",
            vec_train_env,
            verbose=1,
            device="mps",
            n_steps=n_steps,
            batch_size=batch_size,
        )

        model.set_logger(loggers)

        # Train the model for the desired number of timesteps.
        model.learn(total_timesteps=len(train_blocks) * 5, progress_bar=True, log_interval=1)

        total_correct = 0
        total_predictions = 0
        obs = vec_test_env.reset()  # vec_test_env.reset() returns only the observations
        done = False

        while not done:
            # For DummyVecEnv with one environment, obs is a list (or dict) with one element.
            actions, _ = model.predict(obs, deterministic=True)

            # Retrieve the mask. This depends on how DummyVecEnv returns your observation.
            # Here we assume obs is a list of dicts.
            mask = obs[0]["mask"] if isinstance(obs, (list, tuple)) else obs["mask"]
            total_predictions += mask.sum()

            obs, rewards, done, _ = vec_test_env.step(actions)
            # rewards is an array; for a single env, we use the first element.
            total_correct += rewards[0]

        accuracy = total_correct / total_predictions if total_predictions > 0 else 0
        print(f"Custom Test Accuracy: {accuracy:.3f}")
        mlflow.log_metric("test_accuracy", accuracy)

        model_path = "ppo_soft_embeddings_v1.zip"
        model.save(model_path)
        mlflow.log_artifact(model_path)
        os.remove(model_path)


if __name__ == "__main__":
    main()
