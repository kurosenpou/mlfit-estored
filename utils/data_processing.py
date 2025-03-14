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
        
def save_results(time, beta_0, X, y, y_pred, output_file):
    """
    Save time, beta_0, X (Wp), and predicted y (U_stored) to a text file
    
    Parameters:
    -----------
    time : array
        Time values
    beta_0 : array
        beta_0 values
    X : array
        Input features (Wp)
    y_pred : array
        Predicted values (U_stored)
    output_file : str
        Path to save the output file
    """
    # Create output array - make sure all arrays are flattened
    output_data = np.column_stack((time, beta_0, X.flatten(), y.flatten(), y_pred))
    
    # Generate header
    header = "Time(ps) beta_0 Wp(10^9) U_stored U_stored_pred(10^9)"
    
    # Save to file
    np.savetxt(output_file, output_data, header=header)