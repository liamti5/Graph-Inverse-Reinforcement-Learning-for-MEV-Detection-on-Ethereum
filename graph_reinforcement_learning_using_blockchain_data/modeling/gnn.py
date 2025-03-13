import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from torch.nn import Linear
from torch_geometric.nn import GATv2Conv, SAGEConv
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

from graph_reinforcement_learning_using_blockchain_data import config

config.load_dotenv()
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
app = typer.Typer()


class GraphSAGE(nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        torch.manual_seed(42)
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        self.fc1 = nn.Linear(hidden_channels, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, data, return_embeddings=False):
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
        x = self.fc2(x)
        if return_embeddings:
            return x, embeddings
        else:
            return x


class GAT(torch.nn.Module):
    def __init__(self, input_features, hidden_channels, num_classes, edge_attr_dim):
        super(GAT, self).__init__()
        torch.manual_seed(42)
        self.conv1 = GATv2Conv(input_features, hidden_channels, edge_dim=edge_attr_dim)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, edge_dim=edge_attr_dim)
        self.conv3 = GATv2Conv(hidden_channels, hidden_channels, edge_dim=edge_attr_dim)
        self.conv4 = GATv2Conv(hidden_channels, hidden_channels, edge_dim=edge_attr_dim)
        self.lin = Linear(hidden_channels, 256)
        self.lin2 = Linear(256, num_classes)
        self.batchnorm = nn.BatchNorm1d(256)

    def forward(self, data):
        # 1. Obtain node embeddings
        x = self.conv1(data.x, data.edge_index, data.edge_attr)
        x = x.relu()
        x = self.conv2(x, data.edge_index, data.edge_attr)
        x = x.relu()
        x = self.conv3(x, data.edge_index, data.edge_attr)

        # 2. Readout layer
        x = global_mean_pool(x, data.batch, size=data.num_graphs)

        # 3. Apply a final classifier
        x = self.lin(x)

        # x = self.batchnorm(x)

        x = x.relu()
        x = self.lin2(x)
        return x


class GINE(torch.nn.Module):
    def __init__(self, input_features, hidden_channels, num_classes, edge_attr_dim):
        super(GINE, self).__init__()
        torch.manual_seed(42)
        mlp1 = nn.Sequential(
            nn.Linear(input_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.conv1 = GINEConv(mlp1, edge_dim=edge_attr_dim)

        mlp2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.conv2 = GINEConv(mlp2, edge_dim=edge_attr_dim)

        mlp3 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.conv3 = GINEConv(mlp3, edge_dim=edge_attr_dim)

        self.lin = Linear(hidden_channels, 256)
        self.lin2 = Linear(256, num_classes)
        self.batchnorm = nn.BatchNorm1d(256)

    def forward(self, data):
        # 1. Obtain node embeddings
        # edge_attr = data.edge_attr.unsqueeze(-1)  # Now shape: [num_edges, 1]

        x = self.conv1(data.x, data.edge_index, data.edge_attr)
        x = x.relu()
        x = self.conv2(x, data.edge_index, data.edge_attr)
        x = x.relu()
        x = self.conv3(x, data.edge_index, data.edge_attr)

        # 2. Readout layer
        x = global_mean_pool(x, data.batch, size=data.num_graphs)

        # 3. Apply a final classifier
        x = self.lin(x)

        # x = self.batchnorm(x)

        x = x.relu()
        x = self.lin2(x)
        return x


class GINC(torch.nn.Module):
    def __init__(self, input_features, hidden_channels, num_classes):
        super(GINC, self).__init__()
        torch.manual_seed(42)
        mlp1 = nn.Sequential(
            nn.Linear(input_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.conv1 = GINConv(mlp1)

        mlp2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.conv2 = GINConv(mlp2)

        mlp3 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.conv3 = GINConv(mlp3)

        self.lin = Linear(hidden_channels, 256)
        self.lin2 = Linear(256, num_classes)
        self.batchnorm = nn.BatchNorm1d(256)

    def forward(self, data):
        # 1. Obtain node embeddings
        x = self.conv1(data.x, data.edge_index)
        x = x.relu()
        x = self.conv2(x, data.edge_index)
        x = x.relu()
        x = self.conv3(x, data.edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, data.batch, size=data.num_graphs)

        # 3. Apply a final classifier
        x = self.lin(x)

        # x = self.batchnorm(x)

        x = x.relu()
        x = self.lin2(x)
        return x


def train(model, loader, optimizer, criterion, device):
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


def test(model, loader, criterion, device, return_embeddings):
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
    experiment_name,
    num_epochs,
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    device,
    return_embeddings=False,
):
    mlflow.set_experiment(experiment_name)
    mlflow.pytorch.autolog()

    with mlflow.start_run():
        for epoch in tqdm(range(1, num_epochs + 1)):
            print(f"Epoch {epoch} starts")
            train_loss = train(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc, embeddings = test(
                model, test_loader, criterion, device, return_embeddings
            )

            # Log metrics manually (per epoch)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("test_loss", test_loss, step=epoch)
            mlflow.log_metric("test_accuracy", test_acc, step=epoch)

            print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}")

        # Optionally log the final model artifact
        mlflow.pytorch.log_model(model, "model")
        return embeddings

if __name__ == "__main__":
    pass
