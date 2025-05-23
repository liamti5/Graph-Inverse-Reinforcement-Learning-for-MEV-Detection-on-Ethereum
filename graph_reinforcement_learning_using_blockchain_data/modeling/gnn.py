from typing import Dict, Tuple, Union, Any

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import DeepGraphInfomax
from tqdm import tqdm

from graph_reinforcement_learning_using_blockchain_data import config

config.load_dotenv()
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
app = typer.Typer()


class GraphSAGE(nn.Module):
    """
    GraphSAGE model for graph classification.
    """

    def __init__(self, num_node_features: int, hidden_channels: int, num_classes: int) -> None:
        """
        Initializes the GraphSAGE model.

        :param num_node_features: The number of features for each node.
        :param hidden_channels: The number of hidden channels in the SAGEConv layers.
        :param num_classes: The number of output classes for classification.
        :return: None
        """
        torch.manual_seed(42)
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        self.fc1 = nn.Linear(hidden_channels, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(
        self, data: Union[Data, Batch], return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the GraphSAGE model.

        :param data: The input graph data (either a single Data object or a Batch).
        :param return_embeddings: If True, returns both the output and the embeddings.
        :return: The output tensor, or a tuple of (output tensor, embedding tensor)
                 if return_embeddings is True.
        """
        # 1. embeddings
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))

        # 2. readout
        x = global_mean_pool(x, data.batch, size=data.num_graphs)

        # 3. final classifier
        embeddings = self.fc1(x)
        x = F.dropout(embeddings, p=0.5, training=self.training)
        out = self.fc2(x)

        return (out, embeddings) if return_embeddings else out


# DGI
class SAGEEncoder(nn.Module):
    """
    SAGEEncoder module used within the Deep Graph Infomax (DGI) framework.
    This encoder generates node embeddings.
    """

    def __init__(self, num_node_features: int, hidden: int) -> None:
        """
        Initializes the SAGEEncoder.

        :param num_node_features: The number of input features for each node.
        :param hidden: The number of hidden units in the SAGEConv layers.
        :return: None
        """
        super().__init__()
        self.convs = nn.ModuleList(
            [
                SAGEConv(num_node_features, hidden),
                SAGEConv(hidden, hidden),
                SAGEConv(hidden, hidden),
                SAGEConv(hidden, hidden),
            ]
        )

    def forward(self, data: Union[Data, Batch]) -> torch.Tensor:
        """
        Forward pass for the SAGEEncoder.

        :param data: The input graph data (either a single Data object or a Batch).
        :return: Node embeddings tensor of shape (num_nodes, hidden).
        """
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x  # (num_nodes, hidden)


class GraphSAGEClassifier(nn.Module):
    """
    A GraphSAGE based classifier that uses a pre-trained SAGEEncoder.
    """

    def __init__(self, encoder: SAGEEncoder, hidden: int, num_classes: int) -> None:
        """
        Initializes the GraphSAGEClassifier.

        :param encoder: A pre-trained SAGEEncoder instance.
        :param hidden: The number of hidden units in the encoder's output (and input to the classifier).
        :param num_classes: The number of output classes for classification.
        :return: None
        """
        super().__init__()
        self.encoder = encoder  # ← weights already trained
        self.fc1 = nn.Linear(hidden, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(
        self, data: Union[Data, Batch], return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for the GraphSAGEClassifier.

        :param data: The input graph data (either a single Data object or a Batch).
        :param return_embeddings: If True, returns both the output and the embeddings.
        :return: The output tensor, or a tuple of (output tensor, embedding tensor)
                 if return_embeddings is True.
        """
        x = self.encoder(data)
        x = global_mean_pool(x, data.batch, size=data.num_graphs)
        embeddings = self.fc1(x)
        x = F.dropout(embeddings, p=0.5, training=self.training)
        out = self.fc2(x)
        return (out, embeddings) if return_embeddings else out


def pretrain_dgi(
    loader: DataLoader, dgi: DeepGraphInfomax, device: torch.device, epochs: int = 10
) -> SAGEEncoder:
    """
    Pre-trains a SAGEEncoder using the Deep Graph Infomax (DGI) method.

    :param loader: DataLoader for the training data.
    :param dgi: The DeepGraphInfomax model instance.
    :param device: The torch device (e.g., 'cuda' or 'cpu') to train on.
    :param epochs: The number of epochs to train for.
    :return: The pre-trained SAGEEncoder.
    """
    mlflow.set_experiment("DGI")
    mlflow.pytorch.autolog()
    with mlflow.start_run():
        opt = torch.optim.Adam(dgi.parameters(), lr=1e-3)
        for epoch in tqdm(range(epochs)):
            dgi.train()
            tot = 0
            for data in loader:
                data = data.to(device)
                opt.zero_grad()
                pos_z, neg_z, summary = dgi(data)
                loss = dgi.loss(pos_z, neg_z, summary)
                loss.backward()
                opt.step()
                tot += loss.item()

            mlflow.log_metric("loss", tot / len(loader), step=epoch)
            print(f"[DGI] epoch {epoch + 1:02d}  loss={tot / len(loader):.4f}")

        mlflow.pytorch.log_model(dgi, "model")
        return dgi.encoder  # pretrained encoder ready for downstream


def train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Trains the given model for one epoch.

    :param model: The neural network model to train.
    :param loader: DataLoader for the training data.
    :param optimizer: The optimizer to use for training.
    :param criterion: The loss function.
    :param device: The torch device (e.g., 'cuda' or 'cpu') to train on.
    :return: The average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    for data in loader:
        # data.x = torch.cat([data.x, data.global_features[data.batch].unsqueeze(1)], dim=-1)
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


def test(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    return_embeddings: bool,
) -> Tuple[float, float, Dict[Any, torch.Tensor]]:
    """
    Evaluates the model on the test set.

    :param model: The neural network model to test.
    :param loader: DataLoader for the test data.
    :param criterion: The loss function.
    :param device: The torch device (e.g., 'cuda' or 'cpu') to test on.
    :param return_embeddings: If True, computes and returns embeddings for the test data.
    :return: A tuple containing:
             - Average test loss.
             - Test accuracy.
             - A dictionary of embeddings if return_embeddings is True, otherwise an empty dictionary.
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    embeddings = {}

    with torch.no_grad():
        for data in loader:
            # data.x = torch.cat([data.x, data.global_features[data.batch].unsqueeze(1)], dim=-1)
            data = data.to(device)
            if return_embeddings:
                out, emb = model(data, return_embeddings)
                mapping = {trx_id: emb for trx_id, emb in zip(data.trx_id, emb)}
                embeddings.update(mapping)
            else:
                out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.num_graphs
    return total_loss / len(loader.dataset), correct / total, embeddings


@app.command()
def run_experiment(
    experiment_name: str,
    num_epochs: int,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    return_embeddings: bool = False,
) -> Tuple[nn.Module, Dict[Any, torch.Tensor]]:
    """
    Run a complete model training experiment with MLflow tracking.

    This function trains a graph neural network model for a specified number of epochs,
    evaluates it on a test set, and logs metrics to MLflow. It supports returning node
    embeddings for downstream tasks.

    :param experiment_name: Name for the MLflow experiment.
    :param num_epochs: Number of training epochs to run.
    :param model: Neural network model to train.
    :param train_loader: DataLoader containing the training dataset.
    :param test_loader: DataLoader containing the test dataset.
    :param optimizer: PyTorch optimizer for model training.
    :param criterion: Loss function for training.
    :param device: Torch device to use for computation (CPU/GPU).
    :param return_embeddings: Whether to compute and return node embeddings.
    :return: A tuple containing:
             - The trained model.
             - A dictionary mapping transaction IDs to their embeddings (if return_embeddings=True)
               or an empty dictionary (if return_embeddings=False).
    """
    mlflow.set_experiment(experiment_name)
    mlflow.pytorch.autolog()

    with mlflow.start_run():
        for epoch in tqdm(range(1, num_epochs + 1)):
            print(f"Epoch {epoch} starts")
            train_loss = train(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc, embeddings = test(
                model, test_loader, criterion, device, return_embeddings
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("test_loss", test_loss, step=epoch)
            mlflow.log_metric("test_accuracy", test_acc, step=epoch)

            print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}")

        mlflow.pytorch.log_model(model, "model")
        return model, embeddings


if __name__ == "__main__":
    pass
