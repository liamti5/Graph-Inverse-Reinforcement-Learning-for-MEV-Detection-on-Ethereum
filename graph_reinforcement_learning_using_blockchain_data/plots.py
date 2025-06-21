import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_hist(df: pd.DataFrame, label_name: str) -> None:
    """
    Generates and displays a histogram of class labels from a DataFrame.

    :param df: The input DataFrame containing the data.
    :param label_name: The name of the column in the DataFrame that contains the class labels.
    sns.set(style="whitegrid")
    """

    plt.figure(figsize=(8, 6))
    ax = sns.histplot(df[label_name], bins=2, discrete=True)

    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
        )

    ax.set_xticks([0, 1])
    ax.set_xlabel("Class Label")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of Class Labels")

    plt.show()
