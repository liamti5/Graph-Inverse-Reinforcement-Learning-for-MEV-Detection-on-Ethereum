from typing import Tuple, Optional, Any, Dict

import gymnasium as gym
import mlflow
import numpy as np
import pandas as pd
import requests
import torch
from deprecated import deprecated
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import DeepGraphInfomax

import graph_reinforcement_learning_using_blockchain_data as grl
from graph_reinforcement_learning_using_blockchain_data import config

config.load_dotenv()
mlflow.set_tracking_uri(uri=config.MLFLOW_TRACKING_URI)


class TransactionGraphEnvV2(gym.Env):
    """
    A Gymnasium environment for transaction graph reinforcement learning.

    This environment simulates an agent interacting with transaction data,
    aiming to learn optimal actions based on graph embeddings.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label: int,
        model_uri: str,
        observation_space_dim: int,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initializes the TransactionGraphEnvV2.

        :param df: DataFrame containing transaction data.
        :param label: Label associated with the transaction data (1 for arbitrage, 0 for non-arbitrage). Only required to satisfy the create_group_transaction_graph interface.
        :param model_uri: MLflow URI of the pre-trained GNN model.
        :param observation_space_dim: Dimensionality of the observation space.
        :param device: PyTorch device to use for model computations.
        """
        try:
            response = requests.get(config.MLFLOW_TRACKING_URI, timeout=3)
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Could not connect to the MLflow tracking server at {config.MLFLOW_TRACKING_URI}. Have you started it? Original error: {e}"
            )
        model_uri = model_uri
        model = mlflow.pytorch.load_model(model_uri)

        if isinstance(model, DeepGraphInfomax):
            self.gnn = model.encoder
            self.case = "unsupervised"
        else:
            self.gnn = model
            self.case = "supervised"

        self.device = device
        self.gnn.to(device)
        self.label = label
        self.window_size = 10

        self.df = df.copy()
        self.accounts = df["from"].unique()
        self.account_index = None
        self.transaction_index = None
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_space_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(2)
        self.current_trajectory = None
        self.current_account_transactions = None

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to an initial state.

        :param seed: Optional seed for the random number generator.
        :param options: Optional dictionary of environment-specific options.
        :return: A tuple containing the initial observation and an info dictionary.
        """
        super().reset(seed=seed)
        self.account_index = self.np_random.integers(0, len(self.accounts))
        self.transaction_index = 0
        self.current_trajectory = []
        self.current_account_transactions = self.df[
            self.df["from"] == self.accounts[self.account_index]
        ]
        self.current_account_transactions = self.current_account_transactions.sort_values(
            by=["blockNumber"]
        )
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        """
        Generates the current observation.

        :return: The current observation as a NumPy array.
        """
        if self.transaction_index == 0:
            initial_obs = self.current_account_transactions.iloc[0]["embeddings"]
            return np.array(initial_obs, dtype=np.float32)

        if self.transaction_index >= len(self.current_account_transactions):
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        start_idx = max(0, self.transaction_index - self.window_size)
        current_trxs = self.current_account_transactions.iloc[
            start_idx : self.transaction_index
        ].copy()

        graph = grl.create_group_transaction_graph(current_trxs, self.label)[-1]
        graph.x = grl.pad_features(graph.x, 49)
        data = Batch.from_data_list([graph])
        with torch.no_grad():
            data = data.to(self.device)

            if self.case != "supervised":
                z_nodes = self.gnn(data)
                embeddings = global_mean_pool(z_nodes, data.batch)
            else:
                _, embeddings = self.gnn(data, return_embeddings=True)

        return embeddings.cpu().detach().numpy().squeeze(0)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes one time step within the environment.

        :param action: The action taken by the agent.
        :return: A tuple containing the next observation, reward, done flag, truncated flag, and info dictionary.
        """
        current_trx = self.current_account_transactions.iloc[self.transaction_index].copy()
        previous_trx = (
            self.current_account_transactions.iloc[self.transaction_index - 1].copy()
            if self.transaction_index > 0
            else current_trx
        )

        expert_action = current_trx["action"]
        # reward isn't used in AIRL training, but useful for later evaluation of learner
        reward = 1.0 if action == expert_action else 0.0

        if previous_trx["eth_balance"] <= 0:
            previous_trx["eth_balance"] = current_trx["eth_balance"]

        if action != expert_action:
            current_trx["action"] = action
            # priority gas fee, so we add a std to the median gas price
            if action == 1:
                current_trx["eth_balance"] = previous_trx["eth_balance"] - current_trx[
                    "gasUsed"
                ] * (current_trx["median_gas_price"] + current_trx["std_gas_price"])

            else:
                # low gas fee, so we subtract a std to the median gas price
                current_trx["eth_balance"] = previous_trx["eth_balance"] - current_trx[
                    "gasUsed"
                ] * (current_trx["median_gas_price"] - current_trx["std_gas_price"])

            self.current_account_transactions.iloc[self.transaction_index] = current_trx

        else:
            # learner got the correct action, but we still have to recalculate eth balance since we don't know what happened in previous transitions. However, we just use the effective gas price
            current_trx["eth_balance"] = (
                previous_trx["eth_balance"]
                - current_trx["gasUsed"] * current_trx["effectiveGasPrice"]
            )

            self.current_account_transactions.iloc[self.transaction_index] = current_trx

        self.transaction_index += 1
        next_obs = self._obs()
        self.current_trajectory.append((next_obs, action))

        # if we've exhausted all transactions for an account, we can signal done to trigger an env reset
        done = self.transaction_index >= len(self.current_account_transactions)
        info = {}
        # check if we are done with this account
        if done:
            # metadata
            last_trx = self.current_account_transactions.iloc[self.transaction_index - 1]
            info = {
                "from": self.accounts[self.account_index],
                "num_transactions": self.transaction_index,
                "last_block": last_trx["blockNumber"],
                "trajectory": self.current_trajectory,
            }

        truncated = False

        return next_obs, reward, done, truncated, info


@deprecated(reason="Use TransactionGraphEnvV2 instead.")
class TransactionGraphEnv(gym.Env):
    def __init__(
        self,
        df: pd.DataFrame,
        alpha: float,
        label: int,
        device: torch.device = torch.device("cpu"),
    ):
        self.df = df.copy()
        self.accounts = df["from"].unique()
        self.account_index = None
        self.transaction_index = None
        self.alpha = alpha
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(128,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(2)
        self.current_trajectory = None
        self.current_account_transactions = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.account_index = self.np_random.integers(0, len(self.accounts))
        self.transaction_index = 0
        self.current_trajectory = []
        self.current_account_transactions = self.df[
            self.df["from"] == self.accounts[self.account_index]
        ]
        self.current_account_transactions = self.current_account_transactions.sort_values(
            by=["blockNumber"]
        )
        return self._get_observation(), {}

    def _get_observation(self):
        if self.transaction_index >= len(self.current_account_transactions):
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        obs = self.current_account_transactions.iloc[self.transaction_index]["embeddings"]
        obs = np.array(obs, dtype=np.float32)
        return obs

    def step(self, action):
        obs = self._get_observation()
        # If this is the first transaction, there's no previous action to compare.
        if self.transaction_index == 0:
            reward = 0  # or some default reward
        else:
            # Compute reward based on the previous transaction's expert action.
            current_tx = self.current_account_transactions.iloc[self.transaction_index - 1]
            # print(current_tx["transaction_hash"])
            # print(current_tx["from"])
            actual_action = current_tx["action"]
            reward = 1.0 if action == actual_action else -1.0

        self.current_trajectory.append((obs, action))
        self.transaction_index += 1

        done = False
        info = {}
        if self.transaction_index >= len(self.current_account_transactions):
            done = True

            # metadata
            last_tx = self.current_account_transactions.iloc[self.transaction_index - 1]
            info = {
                "from": self.accounts[self.account_index],
                "num_transactions": self.transaction_index,
                "last_block": last_tx["blockNumber"],
                "trajectory": self.current_trajectory,
            }

            # move to next account
            self.account_index += 1
            self.transaction_index = 0
            if self.account_index < len(self.accounts):
                self.current_account_transactions = self.df[
                    self.df["from"] == self.accounts[self.account_index]
                ]
                self.current_account_transactions = self.current_account_transactions.sort_values(
                    by=["blockNumber"]
                )
            else:
                done = True

        next_obs = self._get_observation()
        return next_obs, reward, done, False, info
