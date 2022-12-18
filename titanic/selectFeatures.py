from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import random
SEED = 1234

class ForwardFeatureSelection():
    def select(
        self,
        df, 
        select_columns, 
        target, 
        model_cls, 
        params, 
        eval_func
    ):
        random.shuffle(select_columns)
        choice_columns = []
        base_metric = 0
    
        for _ in range(len(select_columns)):
        
            # 全カラムを評価し、一番いい評価のカラムを特徴量として追加していく
            metrics = []
            for column in tqdm(select_columns):
                if column in choice_columns:
                    continue
            
                _c = choice_columns[:]
                _c.append(column)
        
                metric = self.validation(df, _c, target, model_cls, params, eval_func)
                metrics.append((column, metric))
                
            # 最大評価を追加する、評価を更新できなかったら終了
            metrics.sort(key=lambda x:x[1], reverse=True)
            max_metric = metrics[0]
                
            if base_metric < max_metric[1]:
                base_metric = max_metric[1]
                choice_columns.append(max_metric[0])
            else:
                break
        
        print(f'selected features: {choice_columns}')
        print(f'accuracy score: {round(base_metric,2)}')
        
        return choice_columns
    
    def validation(
        self,
        df, 
        select_columns, 
        target, 
        model_cls, 
        params, 
        eval_func, 
        df_test: pd.DataFrame = None,
        is_pred=False
    ):
        df_train = df.copy()
        val_metrics = []
        #kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        
        X = df_train[df_train.columns[df_train.columns != target]].values
        y = df_train[target].values
        
        for train_idx, true_idx in kf.split(X,y):
            df_train_sub = df_train.iloc[train_idx]
            df_true_sub = df_train.iloc[true_idx]
            train_x = df_train_sub[select_columns]
            train_y = df_train_sub[target]
            true_x = df_true_sub[select_columns]
            true_y = df_true_sub[target]
            
            model = model_cls(**params)
            model.fit(train_x, train_y)
            pred_y = model.predict(true_x)
            
            val_metrics.append(eval_func(true_y, pred_y))
        
        metric = np.mean(val_metrics)
        
        if is_pred:
            train_x = df_train[select_columns]
            train_y = df_train[target]
            test_x = df_test[select_columns]
            
            model = model_cls(**params)
            model.fit(train_x, train_y)
            pred_y = model.predict(test_x)
            
            return metric, pred_y
            
        return metric
            
        