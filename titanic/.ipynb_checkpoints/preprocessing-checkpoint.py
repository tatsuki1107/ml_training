import pandas as pd
import numpy as np

class Preprocessing():
    def preproces(self, df_train, df_test):
        df_train = self.missing_value(df_train)
        df_test = self.missing_value(df_test)
        
        df_train["Embarked"].fillna("S", inplace=True)
        df_test["Fare"].fillna(df_test["Fare"].median(), inplace=True)
        
        df_train = self.normalization(df_train)
        df_test = self.normalization(df_test)
        
        df_train = self.dummy(df_train)
        df_test = self.dummy(df_test)
        
        return df_train, df_test
    
    def missing_value(self, df):
        # 欠損値フラグ
        df["Age_na"] = df["Age"].isnull().astype(np.int64)
        # 欠損値を中央値に
        df["Age"].fillna(df["Age"].median(), inplace=True)
        
        return df

    def normalization(self, df):
        for name in ["Age", "Fare"]:
            df[name] = (df[name] - df[name].mean()) / df[name].std() # 不偏分散
        
        return df
    
    def dummy(self, df):
        df = pd.get_dummies(df, 
                        columns=[
        "Pclass", "Sex", 
        "SibSp", "Parch", 
        "Embarked"
                        ],
                        drop_first=True
                        
        )
        return df