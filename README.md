# graph-reinforcement-learning-using-blockchain-data

[![CCDS](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)
[![Poetry](https://img.shields.io/badge/poetry-1.8.5-blue?logo=poetry&label=poetry)](https://python-poetry.org/)
[![MLflow](https://img.shields.io/badge/mlflow-v2.20.2-orange?logo=mlflow)](https://mlflow.org/)
[![Stable-Baselines3](https://img.shields.io/badge/stable_baselines3-v2.2.1-green)](https://stable-baselines3.readthedocs.io/en/master/)
[![Imitation](https://img.shields.io/badge/imitation-v1.0.1-brightgreen)](https://imitation.readthedocs.io/en/latest/)
![PyTorch Geometric](https://img.shields.io/pypi/v/torch-geometric?label=PyG&color=blue)

A short description of the project.

## ML-Flow
To start the mlflow server, run 
```
mlflow server --host 127.0.0.1 --port 8080
```

## Project Organization

`0.01-pjb-data-source-1.ipynb`

- 0.01 - Helps keep work in chronological order. The structure is PHASE.NOTEBOOK. NOTEBOOK is just the Nth notebook in that phase to be created. For phases of the project, we generally use a scheme like the following, but you are welcome to design your own conventions:
  - 0 - Data exploration - often just for exploratory work
  - 1 - Data cleaning and feature creation - often writes data to data/processed or data/interim
  - 2 - Visualizations - often writes publication-ready viz to reports
  - 3 - Modeling - training machine learning models
  - 4 - Publication - Notebooks that get turned directly into reports
- pjb - Your initials; this is helpful for knowing who created the notebook and prevents collisions from people working in the same notebook.
- data-source-1 - A description of what the notebook covers

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         graph_reinforcement_learning_using_blockchain_data and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
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
    │
    ├── __init__.py             <- Makes graph_reinforcement_learning_using_blockchain_data a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

