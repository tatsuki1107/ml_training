from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(42)


def generate_simulation_data():

    N = 1000
    J = 2
    beta = np.array([0, 1])

    X = np.random.normal(0, 1, [N, J])
    e = np.random.normal(0, 0.1, N)

    y = X@beta + e

    return train_test_split(X, y, test_size=0.2, random_state=42)
