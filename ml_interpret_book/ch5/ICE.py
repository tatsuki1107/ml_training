import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional
from ch4.PD import PartialDependence
import sys
sys.path.append('..')
pd.options.display.float_format = "{:.2f}".format


class IndividualConditionalExpectation(PartialDependence):

    def individual_conditional_expectation(
        self,
        var_name: str,
        ids_to_compute: List[int],
        n_grid: int = 50
    ) -> None:
        """ICEを求める

        Args:
          var_name: ICEを計算したい変数名
          ids_to_compute: ICEを計算したいインスタンスのリスト
          n_grid: なん分割するか
        """

        self.target_var_name = var_name
        var_index = self.var_names.index(var_name)
        value_range = np.linspace(
            self.X[:, var_index].min(),
            self.X[:, var_index].max(),
            num=n_grid
        )

        individual_prediction = np.array([
            self._counterfactual_prediction(var_index, x)[ids_to_compute]
            for x in value_range
        ])

        self.df_ice = (
            pd.DataFrame(
                data=individual_prediction, columns=ids_to_compute
            ).assign(
                **{var_name: value_range}
            ).melt(id_vars=var_name, var_name="instance", value_name="ice")
        )

        self.df_instance = (
            pd.DataFrame(
                data=self.X[ids_to_compute],
                columns=self.var_names
            ).assign(
                instance=ids_to_compute,
                prediction=self.estimator.predict(self.X[ids_to_compute])
            ).loc[:, ["instance", "prediction"] + self.var_names]
        )

    def plot(self, ylim: Optional[List[float]] = None) -> None:
        """ICEを可視化

        Args:
          ylim: Y軸の範囲
        """

        fig, ax = plt.subplots()
        # ICEの線
        sns.lineplot(
            self.target_var_name,
            "ice",
            units="instance",
            data=self.df_ice,
            lw=0.8,
            alpha=0.5,
            estimator=None,
            zorder=1,  # zorderを指定することで、線が背面、点が前面にくるようにする
            ax=ax,
        )
        # インスタンスからの実際の予測値を点でプロットしておく
        sns.scatterplot(
            self.target_var_name,
            "prediction",
            data=self.df_instance,
            zorder=2,
            ax=ax
        )
        ax.set(xlabel=self.target_var_name, ylabel="Prediction", ylim=ylim)
        fig.suptitle(
            f"Individual Conditional Expectation({self.target_var_name})"
        )

        fig.show()
