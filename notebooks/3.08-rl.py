import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from graph_reinforcement_learning_using_blockchain_data import config


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


def main():
    df = pd.read_csv(config.FLASHBOTS_Q2_DATA_DIR / "features_edges_multiocc.csv")
    max_addresses = df.groupby("blockNumber")["from"].nunique().max()

    env = BlockchainEnv(
        df,
        max_num_addresses=max_addresses,
        embedding_dim=128,
        feature_cols=[
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
        ],
        alpha=0.3,
    )
    flat_env = gym.wrappers.FlattenObservation(env)
    #
    # # Run a simple loop over blocks.
    # agent = PPOAgent(embedding_dim=128, action_dim=2)
    # optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    #
    # num_episodes = 5
    # for episode in range(num_episodes):
    #     obs = env.reset()  # obs is a dict with 'states' and 'mask'
    #     done = False
    #     total_reward = 0
    #     total_predictions = 0  # Total number of active address predictions made
    #     while not done:
    #         # Count predictions from the current observation using the mask.
    #         total_predictions += np.sum(obs["mask"])
    #
    #         # Agent acts on the current observation.
    #         actions, log_probs, entropy, values = agent.act(obs)
    #         # Convert actions to numpy array (env.step requires numpy actions).
    #         actions_np = actions.numpy().astype(np.int8)
    #
    #         # Environment step returns next observation and reward (number of correct predictions in block)
    #         next_obs, reward, done, _ = env.step(actions_np)
    #         total_reward += reward
    #
    #         obs = next_obs
    #         # For demonstration, we run one step per block.
    #     # Compute accuracy for this episode.
    #     accuracy = total_reward / total_predictions if total_predictions > 0 else 0
    #     print(
    #         f"Episode {episode + 1} Reward: {total_reward}, Total Predictions: {total_predictions}, Accuracy: {accuracy:.3f}"
    #     )
    check_env(flat_env)

    # Wrap your environment in a DummyVecEnv to make it compatible with SB3.
    vec_env = DummyVecEnv([lambda: env])

    # Create the PPO model using a standard MLP policy.
    model = PPO("MultiInputPolicy", vec_env, verbose=1)

    # Train the model for the desired number of timesteps.
    model.learn(total_timesteps=10)

    # Save the trained model.
    model.save("ppo_blockchain_model")

    # Testing the trained agent on your environment:
    obs = env.reset()
    done = False
    while not done:
        # Use the trained model to predict an action.
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print("Reward for this block:", reward)


if __name__ == "__main__":
    main()
