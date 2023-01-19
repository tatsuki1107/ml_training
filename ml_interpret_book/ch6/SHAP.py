from scipy.special import factorial
from itertools import combinations
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
pd.options.display.float_format = "{:.2f}".format


@dataclass
class ShapleyAdditiveExplanations:
    """SHAP

    Args:
      estimators: 学習済みモデル
      X: SHAPの計算に使う特徴量
      var_names: 特徴量の名前
    """

    estimator: Any
    X: np.ndarray
    var_names: List[str]

    def __post_init__(self) -> None:

        self.baseline = self.estimator.predict(self.X).mean()
        self.J = self.X.shape[1]

        # あり得る特徴量のすべての組み合わせ
        self.subsets = [
            s
            for j in range(self.J + 1)
            for s in combinations(range(self.J), j)
        ]

    def _get_expected_value(self, subset: Tuple[int, ...]) -> np.ndarray:
        """特徴量の組み合わせを指定し予測値を求める

        Args:
          subset: 特徴量の組み合わせ
        """

        _X = self.X.copy()

        if subset is not None:
            _s = list(subset)
            _X[:, _s] = _X[self.i, _s]

        return self.estimator.predict(_X).mean()

    def _calc_weighted_marginal_contribution(
        self,
        j: int,
        s_union_j: Tuple[int, ...]
    ) -> float:
        """限界貢献度x組み合わせの出現回数を求める。

        Args:
          j: 限界貢献度を計算したい特徴量のインデックス
          s_union_j: jを含む特徴量の組み合わせ
        """

        # 特徴量がない場合の組み合わせ
        s = tuple(set(s_union_j) - set([j]))
        S = len(s)

        # 組み合わせの出現回数
        weight = factorial(S) * factorial(self.J - S - 1)

        # 限界貢献度
        marginal_contribution = (
            self.expected_values[s_union_j] - self.expected_values[s]
        )

        return weight * marginal_contribution

    def shapley_additive_explanations(self, id_to_compute: int) -> None:
        """SHAP値を求める

        Args:
          id_to_compute: SHAPを計算したいインスタンス
        """

        self.i = id_to_compute
        self.expected_values = {
            s: self._get_expected_value(s) for s in self.subsets
        }

        shap_values = np.zeros(self.J)
        for j in range(self.J):
            shap_values[j] = np.sum([
                self._calc_weighted_marginal_contribution(j, s_union_j)
                for s_union_j in self.subsets
                if j in s_union_j
            ]) / factorial(self.J)

        self.df_shap = pd.DataFrame(
            data={
                "var_name": self.var_names,
                "feature_value": self.X[id_to_compute],
                "shap_value": shap_values
            }
        )

    def plot(self) -> None:
        """SHAPを可視化"""

        # 下のデータフレームを書き換えないようコピー
        df = self.df_shap.copy()

        # グラフ用のラベルを作成
        df['label'] = [
            f"{x} = {y:.2f}" for x, y in zip(df.var_name, df.feature_value)
        ]

        # SHAP値が高い順に並べ替え
        df = df.sort_values("shap_value").reset_index(drop=True)

        # 全特徴量の値がときの予測値
        predicted_value = self.expected_values[self.subsets[-1]]

        # 棒グラフを可視化
        fig, ax = plt.subplots()
        ax.barh(df.label, df.shap_value)
        ax.set(xlabel="SHAP値", ylabel=None)
        fig.suptitle(
            f"SHAP値 \n(Baseline: {self.baseline:.2f}, Prediction: {predicted_value:.2f}, Difference: {predicted_value - self.baseline:.2f})")

        fig.show()
