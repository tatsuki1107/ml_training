from dataclasses import dataclass
from typing import Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class PartialDependence:
    """Partial Dependence (PD)

    Args:
      estimator: 学習済みモデル
      X: 特徴量
      var_names: 特徴量の名前
    """

    estimator: Any
    X: np.ndarray
    var_names: List[str]

    def _counterfactual_prediction(
        self,
        idx_to_replace: int,
        value_to_replace: float
    ) -> np.ndarray:
        """ある特徴量の値を置き換えたときの予測値を求める。

        Args:
          estimator: 学習済みモデル
          X: 特徴量
          idx_to_replace: 値を置き換える特徴量のインデックス
          value_to_replace: 置き換える値
        """

        X_replaced = self.X.copy()
        X_replaced[:, idx_to_replace] = value_to_replace
        y_pred = self.estimator.predict(X_replaced)

        return y_pred

    def partial_dependence(
        self,
        var_name: str,
        n_grid: int = 50
    ) -> None:
        """PDを求める

        Args:
          var_name: PDを計算したい特徴量の名前
          n_grid: 分割の数
        """

        self.target_var_name = var_name
        var_index = self.var_names.index(var_name)

        value_range = np.linspace(
            self.X[:, var_index].min(),
            self.X[:, var_index].max(),
            num=n_grid
        )

        average_prediction = np.array(
            [self._counterfactual_prediction(
                var_index, x).mean() for x in value_range]
        )

        self.df_partial_dependence = pd.DataFrame(
            data={var_name: value_range, "avg_pred": average_prediction}
        )

    def plot(
        self,
        ylim: Optional[List[float]] = None
    ) -> None:
        """PDを可視化

        Args:
          ylim: Y軸の範囲
        """

        fig, ax = plt.subplots()
        ax.plot(
            self.df_partial_dependence[self.target_var_name],
            self.df_partial_dependence["avg_pred"]
        )
        ax.set(
            xlabel=self.target_var_name,
            ylabel="Average Prediction",
            ylim=ylim
        )
        fig.suptitle(f'Partial Dependence Plot ({self.target_var_name})')

        fig.show()
