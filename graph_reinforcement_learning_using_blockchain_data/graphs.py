import ast

import networkx as nx
import numpy as np
import torch
import typer
from torch_geometric.data import Data

app = typer.Typer()


class Graph:
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def compute_graph_metrics(self):
        simple_G = nx.DiGraph(self.graph)
        density = nx.density(simple_G)

        avg_degree = (
            sum(dict(simple_G.degree()).values()) / simple_G.number_of_nodes()
            if simple_G.number_of_nodes() > 0
            else 0
        )

        clustering_coeff = nx.average_clustering(simple_G.to_undirected())

        components = list(nx.weakly_connected_components(simple_G))
        largest_component_size = max((len(c) for c in components), default=0)
        num_isolated_nodes = len(list(nx.isolates(simple_G)))

        degree_distribution = {}
        for node, degree in simple_G.degree():
            degree_distribution[degree] = degree_distribution.get(degree, 0) + 1

        return {
            "size": simple_G.number_of_nodes(),
            "density": density,
            "average_degree": avg_degree,
            "clustering_coefficient": clustering_coeff,
            "largest_component_size": largest_component_size,
            "num_isolated_nodes": num_isolated_nodes,
            "degree_distribution": degree_distribution,
        }


def _extract_transfer_addresses(topics):
    from_address = "0x" + topics[1][-40:]
    to_address = "0x" + topics[2][-40:]
    return from_address, to_address


def create_group_transaction_graph(group_df, label):
    group_account_mapping = {}
    group_balances = {}
    node_counter = 0
    cumulative_edges = []
    cumulative_edge_attrs = []
    graphs = []

    for _, row in group_df.iterrows():
        from_addr_main, to_addr_main = row["from"].lower(), row["to"].lower()
        if from_addr_main not in group_account_mapping:
            group_account_mapping[from_addr_main] = node_counter
            node_counter += 1
        if to_addr_main not in group_account_mapping:
            group_account_mapping[to_addr_main] = node_counter
            node_counter += 1
        cumulative_edges.append(
            [group_account_mapping[from_addr_main], group_account_mapping[to_addr_main]]
        )

        if from_addr_main not in group_balances:
            group_balances[from_addr_main] = {}

        if to_addr_main not in group_balances:
            group_balances[to_addr_main] = {}

        ethereum_placeholder_addr = "0x0000000000000000000000000000000000000000"
        if ethereum_placeholder_addr not in group_balances[from_addr_main]:
            amount = int(row["eth_balance"])
            group_balances[from_addr_main][ethereum_placeholder_addr] = (
                np.sign(amount) * np.log1p(np.abs(float(amount))) if amount != 0 else 0
            )

        logs = ast.literal_eval(row["logs"])
        for log in logs:
            log_address = log["address"]
            topics = log["topics"]

            ERC20_TRANSFER_SIG = (
                "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
            )

            if not topics or topics[0] != ERC20_TRANSFER_SIG or len(topics) < 3:
                continue

            from_addr, to_addr = _extract_transfer_addresses(topics)
            if from_addr not in group_balances:
                group_balances[from_addr] = {}
            if to_addr not in group_balances:
                group_balances[to_addr] = {}

            amount = int(log["data"], 16) if log["data"] != "0x" else 0
            if log_address not in group_balances[from_addr]:
                group_balances[from_addr][log_address] = 0
            if log_address not in group_balances[to_addr]:
                group_balances[to_addr][log_address] = 0
            group_balances[from_addr][log_address] -= (
                np.sign(amount) * np.log1p(np.abs(float(amount))) if amount != 0 else 0
            )
            group_balances[to_addr][log_address] += (
                np.sign(amount) * np.log1p(np.abs(float(amount))) if amount != 0 else 0
            )

            if from_addr not in group_account_mapping:
                group_account_mapping[from_addr] = node_counter
                node_counter += 1
            if to_addr not in group_account_mapping:
                group_account_mapping[to_addr] = node_counter
                node_counter += 1

            cumulative_edges.append(
                [group_account_mapping[from_addr], group_account_mapping[to_addr]]
            )

        num_nodes = node_counter
        degree = [0] * num_nodes
        for src, dst in cumulative_edges:
            degree[src] += 1
            degree[dst] += 1
        degree = torch.tensor(degree, dtype=torch.float).unsqueeze(1)

        # --- Add balances to the node features ---
        # Create a reverse mapping: node id -> account address.
        reverse_mapping = {v: k for k, v in group_account_mapping.items()}
        # Determine all tokens seen in the balances (fixed order).
        all_tokens = sorted({token for bal in group_balances.values() for token in bal.keys()})
        node_features = []
        for i in range(num_nodes):
            account = reverse_mapping[i]
            # Start with the node's degree.
            deg = degree[i].item()
            # Get the balance dictionary for this account (if any) and produce a vector.
            balances = group_balances.get(account, {})
            token_features = [balances.get(token, 0) for token in all_tokens]
            # Concatenate degree with balance features.
            node_features.append([deg] + token_features)

        node_features_tensor = torch.tensor(node_features, dtype=torch.float)

        edge_index_tensor = torch.tensor(cumulative_edges, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(cumulative_edge_attrs, dtype=torch.int)

        y = torch.tensor(label, dtype=torch.int)
        trx_id = row["transaction_hash"]
        block_number = row["blockNumber"]
        from_addr = row["from"].lower()

        data = Data(
            x=node_features_tensor, y=y, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor
        )
        data.account_mapping = group_account_mapping.copy()
        data.trx_id = trx_id
        data.block_number = block_number
        data.from_addr = from_addr
        graphs.append(data)

    return graphs
