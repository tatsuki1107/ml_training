from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
np.random.seed(42)


def generate_simulation_data(simulation):

    N = 1000
    epsilon = np.random.normal(0, 0.1, N)

    if simulation == 1:
        beta = np.array([1])
        X = np.random.uniform(0, 1, [N, 1])  # 一様分布から特徴量作成
        y = X@beta + epsilon

    elif simulation == 2:
        X = np.random.uniform(-2*np.pi, 2*np.pi, [N, 2])
        y = 10*np.sin(X[:, 0]) + X[:, 1] + epsilon

    return train_test_split(X, y, test_size=0.2, random_state=42)


def plot_scatter(x, y, xlabel="X", ylabel="y", title=None):
    """散布図作成"""

    fig, ax = plt.subplots()
    sns.scatterplot(x, y, ci=None, alpha=0.3, ax=ax)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    fig.suptitle(title)
    fig.show()


def plot_scatters(X, y, var_names, title=None):
    """simulation2用の散布図作成"""
    J = X.shape[1]
    fig, axes = plt.subplots(nrows=1, ncols=J, figsize=(4*J, 4))

    for j, ax in enumerate(axes):
        sns.scatterplot(X[:, j], y, ci=None, alpha=0.3, ax=ax)
        ax.set(
            xlabel=var_names[j],
            ylabel="Y",
            xlim=(X.min()*1.1, X.max()*1.1)
        )
    fig.suptitle(title)
    fig.show()


def regression_metrics(estimator, X, y):
    """回帰精度の評価指標をまとめて返す関数"""

    # テストデータで予測
    y_pred = estimator.predict(X)

    # 評価指標をデータフレームにまとめる
    df = pd.DataFrame(
        data={
            "RMSE": [mean_squared_error(y, y_pred, squared=False)],
            "R2": [r2_score(y, y_pred)],
        }
    )

    return df


def get_coef(estimator, var_names):
    """特徴量名と回帰係数が対応したデータフレームを作成する"""

    # 切片含む回帰係数と特徴量の名前を抜き出してデータフレームにまとめる
    df = pd.DataFrame(
        data={"coef": [estimator.intercept_] + estimator.coef_.tolist()},
        index=["intercept"] + var_names
    )

    return df


def counter_factual_prediction(
    estimator,
    X,
    idx_to_replace,
    value_to_replace
):
    """ある特徴量の値を置き換えたときの予測値を求める。

    Args:
      estimator: 学習済みモデル
      X: 特徴量
      idx_to_replace: 値を置き換える特徴量のインデックス
      value_to_replace: 置き換える値
    """

    X_replaced = X.copy()
    X_replaced[:, idx_to_replace] = value_to_replace
    y_pred = estimator.predict(X_replaced)

    return y_pred


def plot_line(x, y, xlabel="X", ylabel="Y", title=None):
    """特徴量の値を変化させたときの値の変化を可視化"""

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    fig.suptitle(title)
    fig.show()
