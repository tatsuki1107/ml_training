from sklearn.metrics import mean_squared_error
from typing import Any
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import List
import pandas as pd

@dataclass
class PermutationFeatureImportance:
    """PFI
    
    Args:
      estimator: 全特徴量を用いた学習済みモデル
      X: 特徴量
      y: 目的変数
      var_names: 特徴量の名前
    """
    
    estimator: Any
    X: np.ndarray
    y: np.ndarray
    var_names: List[str]
    
    def __post_init__(self) -> None:
        # シャッフルなしの場合の予測精度
        # mean_squared_error()はsquared=TrueならMSE、squared=FalseならRMSE
        self.baseline = mean_squared_error(
            self.y, self.estimator.predict(self.X), squared=False
        )
    
    def _permutation_metrics(self, idx_to_permute:int) -> float:
        """ある特徴量をシャッフルしたときの予測精度を求める
        
        Args:
          idx_to_permute: シャッフルする特徴量のインデックス
        """
        
        X_permuted = self.X.copy()
        
        # 特徴量をシャッフルして予測
        X_permuted[:,idx_to_permute] = np.random.permutation(
            X_permuted[:,idx_to_permute]
        )
        
        y_pred = self.estimator.predict(X_permuted)
        
        return mean_squared_error(self.y, y_pred, squared=False)
    
    def permutation_feature_importance(self, n_shuffle:int = 10) -> None:
        """PFIを求める
        
        Args:
          n_shuffle: シャッフルの回数
        """
        
        J = self.X.shape[1]
        
        # J個の特徴量に対してPFIを求めたい
        # R回シャッフルを繰り返して平均をとることで値を安定させている
        metrics_permuted = [
            np.mean([self._permutation_metrics(j) for r in range(n_shuffle)]) for j in range(J)
        ]
        
        # データフレームとしてまとめる
        # シャッフルでどのくらい予測精度が落ちるかは、
        # 差(difference)と比率(ratio)の2種類を用意する
        df_feature_importance = pd.DataFrame(
            data={
                "var_names": self.var_names,
                "baseline": self.baseline,
                "permutation": metrics_permuted,
                "difference": metrics_permuted - self.baseline,
                "ratio": metrics_permuted / self.baseline
            }
        )
        
        self.feature_importance = df_feature_importance.sort_values(
            "permutation", ascending=False
        )
    
    def plot(self, importance_type:str = "difference") -> None:
        """PFIを可視化
        
        Args:
          importance_type: PFIを差(difference)と比率(ratio)のどちらで計算するか
        """
        
        fig, ax = plt.subplots()
        ax.barh(
            self.feature_importance["var_names"],
            self.feature_importance[importance_type],
            label=f"baseline: {self.baseline:.2f}"
        )
        ax.set(xlabel=importance_type, ylabel=None)
        ax.invert_yaxis() #重要度が高い順に並び替える
        ax.legend(loc="lower right")
        fig.suptitle(f'Permutationによる特徴量の重要度({importance_type})')
        
        fig.show()