# Graph Inverse Reinforcement Learning for MEV Detection on Ethereum

[![CCDS](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)
[![Poetry](https://img.shields.io/badge/poetry-1.8.5-blue?logo=poetry&label=poetry)](https://python-poetry.org/)
[![MLflow](https://img.shields.io/badge/mlflow-v2.20.2-orange?logo=mlflow)](https://mlflow.org/)
[![Stable-Baselines3](https://img.shields.io/badge/stable_baselines3-v2.2.1-green)](https://stable-baselines3.readthedocs.io/en/master/)
[![Imitation](https://img.shields.io/badge/imitation-v1.0.1-brightgreen)](https://imitation.readthedocs.io/en/latest/)
[![PyTorch Geometric](https://img.shields.io/pypi/v/torch-geometric?label=PyG&color=blue)](https://pytorch-geometric.readthedocs.io/en/latest/index.html)

This repository contains the full implementation of a research framework developed for a master’s thesis at 
the [University of Zurich (UZH)](https://www.uzh.ch/en.html), conducted within the 
[Blockchain and Distributed Ledger Technologies (BDLT)](https://www.ifi.uzh.ch/en/bdlt/index.html) group. 
The project focuses on detecting MEV arbitrage transactions on the Ethereum blockchain by combining Graph Neural Networks (GNNs) 
with Adversarial Inverse Reinforcement Learning (AIRL). It includes baseline classifiers, GNN-based embedding models, and AIRL training 
pipelines that jointly learn reward functions and policies from transaction graph data. The code supports supervised, unsupervised, and 
hybrid learning setups, enabling both classification and policy inference. It is intended to support reproducible research in MEV detection, 
transaction classification, and reinforcement learning for blockchain analytics.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://gitlab.uzh.ch/liam.tessendorf/graph-reinforcement-learning-using-blockchain-data.git
    cd graph-reinforcement-learning-using-blockchain-data
    ```

2.  **Ensure Python and Poetry are installed:**
    This project uses Poetry for dependency management. If you don't have Poetry installed, you can install it by following the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation).

3.  **Install dependencies:**
    Navigate to the project root directory (where `pyproject.toml` is located) and run:
    ```bash
    poetry install
    ```
    This will create a virtual environment and install all the necessary dependencies.

4.  **Activate the virtual environment:**
    To activate the virtual environment created by Poetry, run:
    ```bash
    poetry shell
    ```
    
Now you should have all the dependencies installed and are ready to run files in this project.

> [!IMPORTANT]  
> ML-Flow is used for tracking experiments, models, and metrics. Make sure to start the ML-Flow server to load artifacts. See below.

## ML-Flow

To start the mlflow server, run 
```bash
mlflow server --host 127.0.0.1 --port 8080
```
then you can visit http://127.0.0.1:8080 to see all experiments. 

## Usage

### AIRL evaluation

The main evaluation notebook for AIRL can be found in `notebooks/4.01-airl-evaluation.ipynb`. It contains the evaluation of all three 
different AIRL models, including:
- Classification performance metrics like accuracy and F1-score as well as confusion matrices on unseen trajectories
- Learner (PPO) performance with learned reward functions on unseen trajectories

### AIRL training

AIRL was trained using the `graph_reinfocement_learning_with_blockchain_data/rl/airl.py` script. It takes 4 arguments:
1. `--data_class`: class 0 or 1 (non-arbitrage or arbitrage)
2. `--embeddings`: which embeddings dataframe to use (GraphSAGE, DGI, or semi-supervised GraphSAGE)
3. `--experiment_name`: name for ML-Flow experiment
4. `--mlflow_gnn_path`: ML-Flow path to the GNN to use for state encoding (used in `graph_reinfocement_learning_with_blockchain_data/rl/environments.py`)

Example usage:
```bash
poetry run python3 graph_reinforcement_learning_using_blockchain_data/rl/airl.py --data_class 1 --embeddings state_embeddings_pre_trained_128.csv --experiment_name "AIRLv2 DGI semi-supervised" --mlflow_gnn_path mlflow-artifacts:/132032870842317128/7559d28e50674e629ce8042ea64902de/artifacts/model 
```

### GNN training

The GNNs were trained using the `graph_reinfocement_learning_with_blockchain_data/modeling/gnn.py` script. It contains definitions of:
- GraphSAGE model
- SAGEEncoder
- GraphSAGEClassifier
as well as helper functions for training. You can use the `run_experiment` function for training. 

Example usage:
```python
import graph_reinforcement_learning_using_blockchain_data as grl

grl.run_experiment(
    "Graph SAGE", 20, model_GNNSAGE, train_loader, test_loader, optimizer, criterion, device
)
```

### Random Forest training

The Random Forest (RF) models were trained using the `graph_reinfocement_learning_with_blockchain_data/modeling/random_forest.py` script. 

Example usage:
```python
import graph_reinforcement_learning_using_blockchain_data as grl

rf_trainer = grl.RandomForestTrainer()
grid_search = rf_trainer.grid_search(features_to_scale)
rf_trainer.train(X_train, X_test, y_train, y_test, grid_search, "Edge Classification")
```
### Data

The `graph_reinforcement_learning_using_blockchain_data/raw_ethereum_data.py` script is used to fetch and process raw Ethereum 
blockchain data, such as transaction receipts and ETH balances. It requires an Alchemy API URL to be set as an environment 
variable `ALCHEMY_API_URL`.

The script accepts the following command-line arguments:
-   `--data`: Specifies the type of data to process.
    -   `receipts0`: Fetch transaction receipts for non-arbitrage transactions (class 0).
    -   `receipts1`: Fetch transaction receipts for arbitrage transactions (class 1).
    -   `eth_balances`: Fetch ETH balances for accounts that are present in a given CSV file.
-   `--rows` (optional, for `receipts0`, `receipts1`): Number of rows to sample from the source arbitrage dataset. Defaults to -1 (all rows).
-   `--output_filename`: Name of the output CSV file.
-   `--input_filename` (optional, for `eth_balances`): Name of the input CSV file containing accounts and block numbers.

Example usage:

Fetching transaction receipts for class 0 (non-arbitrage):
```bash
poetry run python graph_reinforcement_learning_using_blockchain_data/raw_ethereum_data.py --data receipts0 --rows 1000 --output_filename sampled_receipts_class0.csv
```

Fetching ETH balances for accounts from `processed_receipts_class0.csv`:
```bash
poetry run python graph_reinforcement_learning_using_blockchain_data/raw_ethereum_data.py --data eth_balances --input_filename receipts_class0.csv --output_filename eth_balances_class0.csv
```

## Project Organization

`0.01-pjb-data-source-1.ipynb`

- 0.01 - Helps keep work in chronological order. The structure is PHASE.NOTEBOOK. NOTEBOOK is just the Nth notebook in that phase to be created:
  - 0 - Data exploration - often just for exploratory work
  - 1 - Data cleaning and feature creation - often writes data to data/processed or data/interim
  - 2 - Visualizations - often writes publication-ready viz to reports
  - 3 - Modeling - training machine learning models
  - 4 - Publication - Notebooks that get turned directly into reports
- data-source-1 - A description of what the notebook covers

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         graph_reinforcement_learning_using_blockchain_data and configuration for tools like black
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── setup.cfg          <- Configuration file for flake8
│
└── graph_reinforcement_learning_using_blockchain_data   <- Source code for use in this project.
    ├── __init__.py          <- Makes graph_reinforcement_learning_using_blockchain_data a Python module
    ├── config.py            <- Stores configuration variables and paths
    ├── graphs.py            <- Functions for graph creation and manipulation
    ├── modeling             <- Contains code related to model training and architecture
    │   ├── __init__.py      <- Makes modeling a Python submodule
    │   ├── gnn.py           <- Graph Neural Network model implementations
    │   └── random_forest.py <- Random Forest model implementations
    ├── plots.py             <- Functions for creating visualizations
    ├── raw_ethereum_data.py <- Scripts for fetching and processing raw Ethereum data
    ├── rl                   <- Contains code related to Reinforcement Learning
    │   ├── __init__.py      <- Makes rl a Python submodule
    │   ├── airl.py          <- Adversarial Inverse Reinforcement Learning implementations
    │   ├── environments.py  <- Custom RL environment definitions
    │   └── rl.py            <- General RL algorithm implementations and utilities
    └── utils.py             <- Utility functions used across the project
```

--------

