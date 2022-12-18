import sklearn.ensemble
import sklearn.gaussian_process
import sklearn.naive_bayes
import sklearn.linear_model
import sklearn.neighbors
import sklearn.tree
import sklearn.discriminant_analysis
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

class train():
    def create_models(self, random_seed):
        models = [
           #Ensemble Methods
        sklearn.ensemble.AdaBoostClassifier(random_state=random_seed),
        sklearn.ensemble.BaggingClassifier(random_state=random_seed),
        sklearn.ensemble.ExtraTreesClassifier(random_state=random_seed),
        sklearn.ensemble.GradientBoostingClassifier(random_state=random_seed),
        sklearn.ensemble.RandomForestClassifier(random_state=random_seed),

        #Gaussian Processes
        sklearn.gaussian_process.GaussianProcessClassifier(random_state=random_seed),

        #GLM
        sklearn.linear_model.LogisticRegressionCV(random_state=random_seed),
        sklearn.linear_model.RidgeClassifierCV(),

        #Navies Bayes
        sklearn.naive_bayes.BernoulliNB(),
        sklearn.naive_bayes.GaussianNB(),

        #Nearest Neighbor
        sklearn.neighbors.KNeighborsClassifier(),

        #Trees
        sklearn.tree.DecisionTreeClassifier(random_state=random_seed),
        sklearn.tree.ExtraTreeClassifier(random_state=random_seed),

        #Discriminant Analysis
        sklearn.discriminant_analysis.LinearDiscriminantAnalysis(),
        sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis(),

        #xgboost
        xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=random_seed),
        
        # light bgm
        lgb.LGBMClassifier(random_state=random_seed),
    ]
        return models
    
    def fit(self, df, columns, target, random_seed):
        X = df[columns].to_numpy()
        y = df[target].to_numpy()
    
        model_scores = {}
        kf = KFold(n_splits=3, shuffle=True, random_state=random_seed)
        for train_idx, true_idx in kf.split(X,y):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_true = X[true_idx]
            y_true = y[true_idx]
        
            for model in self.create_models(random_seed):
                name = model.__class__.__name__
                if name not in model_scores:
                    model_scores[name] = []
            
                model.fit(X_train, y_train)
                pred_y = model.predict(X_true)
            
                model_scores[name].append((
                    accuracy_score(y_true, pred_y),
                    precision_score(y_true, pred_y),
                    recall_score(y_true, pred_y),
                    f1_score(y_true, pred_y),
                ))
    
        for k, scores in model_scores.items():
            scores = np.mean(scores, axis=0)
            print("正解率 {:.3f}, 適合率 {:.3f}, 再現率 {:.3f}, F値 {:.3f} : {}".format(
                scores[0],
                scores[1],
                scores[2],
                scores[3],
                k,
        ))
    


    