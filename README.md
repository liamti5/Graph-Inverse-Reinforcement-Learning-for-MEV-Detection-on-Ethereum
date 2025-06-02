# Graph Inverse Reinforcement Learning for MEV Detection on Ethereum

[![CCDS](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)
[![Poetry](https://img.shields.io/badge/poetry-1.8.5-blue?logo=poetry&label=poetry)](https://python-poetry.org/)
[![MLflow](https://img.shields.io/badge/mlflow-v2.20.2-orange?logo=mlflow)](https://mlflow.org/)
[![Stable-Baselines3](https://img.shields.io/badge/stable_baselines3-v2.2.1-green)](https://stable-baselines3.readthedocs.io/en/master/)
[![Imitation](https://img.shields.io/badge/imitation-v1.0.1-brightgreen)](https://imitation.readthedocs.io/en/latest/)
[![PyTorch Geometric](https://img.shields.io/pypi/v/torch-geometric?label=PyG&color=blue)](https://pytorch-geometric.readthedocs.io/en/latest/index.html)

This repository contains the full implementation of a research framework developed for a master’s thesis at the [University of Zurich (UZH)](https://www.uzh.ch/en.html), conducted within the [Blockchain and Distributed Ledger Technologies (BDLT)](https://www.ifi.uzh.ch/en/bdlt/index.html) group. The project focuses on detecting MEV arbitrage transactions on the Ethereum blockchain by combining Graph Neural Networks (GNNs) with Adversarial Inverse Reinforcement Learning (AIRL). It includes baseline classifiers, GNN-based embedding models, and AIRL training pipelines that jointly learn reward functions and policies from transaction graph data. The code supports supervised, unsupervised, and hybrid learning setups, enabling both classification and policy inference. It is intended to support reproducible research in MEV detection, transaction classification, and reinforcement learning for blockchain analytics.

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
    
Now you should have all the dependencies installed and are ready to run the project.

> [!IMPORTANT]  
> ML-Flow is used for tracking experiments, models, and metrics. Make sure to start the ML-Flow server to load artifacts. See below.

## ML-Flow
To start the mlflow server, run 
```
mlflow server --host 127.0.0.1 --port 8080
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
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
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

