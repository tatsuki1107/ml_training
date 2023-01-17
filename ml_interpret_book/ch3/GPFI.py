from PFI import PermutationFeatureImportance
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


class GroupedPermutationFeatureImportance(PermutationFeatureImportance):

    def _permutation_metrics(
        self,
        var_names_to_permute: List[str]
    ) -> float:
        """ある特徴量の値をシャッフルしたときの予測精度を求める

        Args:
          var_names_to_permutate: シャッフルする特徴量群の名前
        """

        X_permuted = self.X.copy()

        # 特徴量名をインデックスに変換
        idx_to_permute = [
            self.var_names.index(v) for v in var_names_to_permute
        ]

        # 特徴量群をまとめてシャッフルして予測
        X_permuted[:, idx_to_permute] = np.random.permutation(
            X_permuted[:, idx_to_permute]
        )
        y_pred = self.estimator.predict(X_permuted)

        return mean_squared_error(self.y, y_pred, squared=False)

    def permutation_feature_importance(
        self,
        var_groups: Optional[List[List[str]]] = None,
        n_shuffle: int = 10
    ) -> None:
        """GPFIを求める

        Args:
          var_groups: グループ化された特徴量名のリスト 
          n_shuffle: シャッフルの回数
        """

        # グループが指定されなかった場合は一つの特徴量を1グループとする。PFIと同じ
        if var_groups is None:
            var_groups = [[j] for j in self.var_names]

        # グループごとに重要度を計算
        # R回シャッフルすることで値を安定させる
        metrics_permuted = [
            np.mean(
                [self._permutation_metrics(j) for r in range(n_shuffle)]
            )
            for j in var_groups
        ]

        df_feature_importance = pd.DataFrame(
            data={
                "var_names": ["+".join(j) for j in var_groups],
                "baseline": self.baseline,
                "permutation": metrics_permuted,
                "difference": metrics_permuted - self.baseline,
                "ratio": metrics_permuted / self.baseline,
            }
        )

        self.feature_importance = df_feature_importance.sort_values(
            "permutation", ascending=False
        )
