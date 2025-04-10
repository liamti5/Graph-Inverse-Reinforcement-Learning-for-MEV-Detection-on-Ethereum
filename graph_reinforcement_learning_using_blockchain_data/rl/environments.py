import gymnasium as gym
import mlflow
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch



import graph_reinforcement_learning_using_blockchain_data as grl
from graph_reinforcement_learning_using_blockchain_data import config

config.load_dotenv()
mlflow.set_tracking_uri(uri=config.MLFLOW_TRACKING_URI)


class TransactionGraphEnv(gym.Env):
    def __init__(
        self,
        df: pd.DataFrame,
        alpha: float,
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
        self.account_index = config.RNG.integers(0, len(self.accounts))
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


class TransactionGraphEnvV2(gym.Env):
    def __init__(
        self,
        df: pd.DataFrame,
        alpha: float,
        label: int,
        device: torch.device = torch.device("cpu"),
    ):
        model_uri = (
            "mlflow-artifacts:/748752183556303764/465d6f94d00b4fe2bef4d8885ed7be39/artifacts/model"
        )
        self.gnn = mlflow.pytorch.load_model(model_uri)
        self.device = device
        self.gnn.to(device)
        self.label = label

        self.window_size = 10

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
        self.account_index = config.RNG.integers(0, len(self.accounts))
        self.transaction_index = 0
        self.current_trajectory = []
        self.current_account_transactions = self.df[
            self.df["from"] == self.accounts[self.account_index]
        ]
        self.current_account_transactions = self.current_account_transactions.sort_values(
            by=["blockNumber"]
        )
        return self._get_observation(), {}

    def _get_observation(self, action=None):
        if self.transaction_index == 0 or action is None:
            initial_obs = self.current_account_transactions.iloc[0]["embeddings"]
            return np.array(initial_obs, dtype=np.float32)

        if self.transaction_index >= len(self.current_account_transactions):
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        previous_trx = self.current_account_transactions.iloc[self.transaction_index - 1].copy()
        current_trx = self.current_account_transactions.iloc[self.transaction_index].copy()
        if action != current_trx["action"]:
            current_trx["action"] = action
            current_trx["eth_balance"] = previous_trx["eth_balance"] - (
                current_trx["median_gas_price"] + current_trx["std_gas_price"]
            )

            start_idx = max(0, self.transaction_index - 1 - self.window_size)
            current_trxs = self.current_account_transactions.iloc[start_idx:self.transaction_index - 1]
            current_trxs = pd.concat([current_trxs, pd.DataFrame([previous_trx])], ignore_index=True)

            graph = grl.create_group_transaction_graph(current_trxs, self.label)[-1]
            graph.x = grl.pad_features(graph.x, 51)
            data = Batch.from_data_list([graph])
            with torch.no_grad():
                data = data.to(self.device)
                _, embeddings = self.gnn(data, return_embeddings=True)

            obs = embeddings.cpu().detach().numpy().squeeze(0)

        else:
            obs = self.current_account_transactions.iloc[self.transaction_index]["embeddings"]
            obs = np.array(obs, dtype=np.float32)

        return obs

    def step(self, action):
        obs = self._get_observation(action)
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

        next_obs = self._get_observation(action)
        return next_obs, reward, done, False, info
