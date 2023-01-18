from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(42)


def generate_simulation_data(simulation: int):

    N = 1000
    x0 = np.random.uniform(-1, 1, N)
    x2 = np.random.binomial(1, 0.5, N)

    if simulation == 1:
        x1 = np.random.uniform(-1, 1, N)

    elif simulation == 2:
        # X1はX2に依存する
        x1 = np.where(
            x2 == 1,
            np.random.uniform(-0.5, 1, N),
            np.random.uniform(-1, 0.5, N)
        )

    epsilon = np.random.normal(0, 0.1, N)

    X = np.column_stack((x0, x1, x2))

    # モデル
    y = x0 - 5*x1 + 10*x1*x2 + epsilon

    return train_test_split(X, y, test_size=0.2, random_state=42)


def plot_scatter(
    x,
    y,
    group=None,
    title=None,
    xlabel=None,
    ylabel=None,
    simulation=None
):
    """散布図を作成する"""
    fig, ax = plt.subplots()

    if simulation == 2:
        sns.scatterplot(x, y, style=group, hue=group, alpha=0.5, ax=ax)
    elif simulation == 1:
        ax.scatter(x, y, alpha=0.3)

    ax.set(xlabel=xlabel, ylabel=ylabel)
    fig.suptitle(title)

    fig.show()
