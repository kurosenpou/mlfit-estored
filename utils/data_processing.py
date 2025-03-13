import numpy as np
from sklearn.model_selection import train_test_split

def load_data(filename):
    data = np.loadtxt(filename)
    time = data[:, 0] / 1000.0
    Wp = data[:, 6] / 1e09
    U_stored = (data[:, 11] - data[:, 5]) / 1e09
    beta_0 = data[:, 21]
    X = Wp.reshape(-1, 1)
    y = U_stored
    return X, y, time, beta_0

def split_data(X, y, args):
    if args.split:
        return train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)
    else:
        return X, X, y, y
