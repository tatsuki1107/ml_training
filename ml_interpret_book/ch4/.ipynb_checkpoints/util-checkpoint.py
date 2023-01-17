from sklearn.model_selection import train_test_split
import numpy as np

def generate_simulation_data1():
    
    N = 1000
    beta = np.array([1])
    
    X = np.random.uniform(0, 1, [N, 1]) # 一様分布から特徴量作成
    epsilon = np.random.normal(0, 0.1, N)
    y = X@beta + epsilon
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

