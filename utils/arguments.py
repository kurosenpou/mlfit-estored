import argparse
import sys
import textwrap

def show_nlsq_help():
    """Display help specific to nonlinear least squares model."""
    help_text = """
    Nonlinear Least Squares Model Options:
    -------------------------------------
    A physical model that directly fits the equation: U_stored = Umax * [1 - exp(- alpha*(Wp / Umax))]
    
    Model-specific arguments:
      --umax_init FLOAT   Initial guess for Umax parameter (default: 1.0)
      --alpha_init FLOAT  Initial guess for alpha parameter (default: 1.0)
    
    Example:
      mlfit data.txt results.png nlsq --umax_init 5.0 --alpha_init 0.5
    """
    print(textwrap.dedent(help_text))
    sys.exit(0)

def show_ensemble_help():
    """Display help specific to ensemble models (RF, GBDT)."""
    help_text = """
    Ensemble Model Options (Random Forest, Gradient Boosting):
    --------------------------------------------------------
    Tree-based ensemble methods for regression.
    
    Model-specific arguments:
      --n_estimators INT      Number of trees (default: 100)
      --max_depth INT         Maximum depth of trees (default: None)
      --min_samples_split INT Minimum samples required to split a node (default: 2)
    
    Example:
      mlfit data.txt results.png rf --n_estimators 200 --max_depth 10
      mlfit data.txt results.png gbdt --n_estimators 150
    """
    print(textwrap.dedent(help_text))
    sys.exit(0)

def show_nn_help():
    """Display help specific to neural network model."""
    help_text = """
    Neural Network Model Options:
    ---------------------------
    A simple neural network model for regression.
    
    Model-specific arguments:
      --hidden_layers INT    Number of hidden layers (default: 2)
      --neurons INT          Neurons per hidden layer (default: 10)
      --activation STRING    Activation function: relu, tanh, sigmoid (default: relu)
      --learning_rate FLOAT  Learning rate for optimizer (default: 0.001)
      --epochs INT           Number of training epochs (default: 1000)
    
    Example:
      mlfit data.txt results.png nn --hidden_layers 3 --neurons 20 --epochs 2000
    """
    print(textwrap.dedent(help_text))
    sys.exit(0)

def show_pinn_help():
    """Display help specific to physics-informed neural network."""
    help_text = """
    Physics-Informed Neural Network Options:
    --------------------------------------
    Neural network with physics constraints for parameter discovery.
    
    Model-specific arguments:
      --hidden_layers INT    Number of hidden layers (default: 4)
      --neurons INT          Neurons per hidden layer (default: 20)
      --physics_weight FLOAT Weight for physics loss term (default: 1.0)
      --epochs INT           Number of training epochs (default: 2000)
    
    Example:
      mlfit data.txt results.png pinn --hidden_layers 5 --neurons 30 --physics_weight 0.8
    """
    print(textwrap.dedent(help_text))
    sys.exit(0)

def show_timeseries_help():
    """Display help specific to time series models (RNN, LSTM)."""
    help_text = """
    Time Series Model Options (RNN, LSTM):
    ------------------------------------
    Recurrent neural networks for time series modeling.
    
    Model-specific arguments:
      --units INT            Number of units in recurrent layer (default: 50)
      --layers INT           Number of recurrent layers (default: 1)
      --dropout FLOAT        Dropout rate (default: 0.1)
      --look_back INT        Number of time steps to look back (default: 5)
      --epochs INT           Number of training epochs (default: 100)
    
    Example:
      mlfit data.txt results.png rnn --units 100 --layers 2
      mlfit data.txt results.png lstm --units 64 --dropout 0.2
    """
    print(textwrap.dedent(help_text))
    sys.exit(0)

def show_poly_help():
    """Display help specific to polynomial model."""
    help_text = """
    Polynomial Regression Model Options:
    ---------------------------------
    A polynomial regression model that fits data to a polynomial function.
    
    Model-specific arguments:
      --degree INT    Degree of the polynomial (default: 3)
    
    Example:
      mlfit data.txt results.png poly --degree 4
    """
    print(textwrap.dedent(help_text))
    sys.exit(0)

def show_svr_help():
    """Display help specific to SVR model."""
    help_text = """
    Support Vector Regression Model Options:
    -------------------------------------
    A support vector machine for regression.
    
    Model-specific arguments:
      --kernel STRING       Kernel type: linear, poly, rbf, sigmoid (default: rbf)
      --C FLOAT             Regularization parameter (default: 1.0)
      --epsilon FLOAT       Epsilon in the epsilon-SVR model (default: 0.1)
      --gamma STRING/FLOAT  Kernel coefficient: scale, auto, or float (default: scale)
    
    Example:
      mlfit data.txt results.png svr --kernel rbf --C 10 --epsilon 0.05
    """
    print(textwrap.dedent(help_text))
    sys.exit(0)

def show_gp_help():
    """Display help specific to Gaussian Process model."""
    help_text = """
    Gaussian Process Model Options:
    -----------------------------
    A Bayesian approach that can estimate uncertainty in predictions.
    
    Model-specific arguments:
      --kernel_type STRING   Kernel type: rbf, matern (default: rbf)
      --length_scale FLOAT   Length scale parameter for kernel (default: 1.0)
      --gp_alpha FLOAT       Noise level parameter (default: 1e-10)
    
    Example:
      mlfit data.txt results.png gp --kernel_type matern --length_scale 0.5
    """
    print(textwrap.dedent(help_text))
    sys.exit(0)

def show_advanced_gb_help():
    """Display help specific to advanced gradient boosting models."""
    help_text = """
    Advanced Gradient Boosting Options (XGBoost, LightGBM, CatBoost):
    ---------------------------------------------------------------
    High-performance gradient boosting frameworks.
    
    Model-specific arguments:
      --n_estimators INT      Number of boosting rounds (default: 100)
      --learning_rate FLOAT   Step size shrinkage (default: 0.1)
      --max_depth INT         Maximum tree depth (default: 6)
      --subsample FLOAT       Subsample ratio of training data (default: 1.0)
    
    Example:
      mlfit data.txt results.png xgb --n_estimators 200 --learning_rate 0.05
      mlfit data.txt results.png lgb --max_depth 8 --subsample 0.8
      mlfit data.txt results.png cat --n_estimators 150
    """
    print(textwrap.dedent(help_text))
    sys.exit(0)

def parse_arguments():
    # Check for model-specific help request
    if len(sys.argv) >= 4 and '-help' in sys.argv:
        model_type = sys.argv[3]
        
        if model_type == 'nlsq':
            show_nlsq_help()
        elif model_type in ['rf', 'gbdt']:
            show_ensemble_help()
        elif model_type == 'nn':
            show_nn_help()
        elif model_type == 'pinn':
            show_pinn_help()
        elif model_type in ['rnn', 'lstm']:
            show_timeseries_help()
        elif model_type == 'poly':
            show_poly_help()
        elif model_type == 'svr':
            show_svr_help()
        elif model_type == 'gp':
            show_gp_help()
        elif model_type in ['xgb', 'lgb', 'cat']:
            show_advanced_gb_help()
    
    # Regular argument parsing
    parser = argparse.ArgumentParser(description='Machine Learning Model for U_stored prediction')
    
    parser.add_argument('input_file', type=str, help='Path to input data file')
    parser.add_argument('output_file', type=str, help='Path to output file')
    parser.add_argument('model_type', type=str, 
                      choices=['nlsq', 'rf', 'gbdt', 'nn', 'pinn', 'rnn', 'lstm', 
                              'poly', 'svr', 'gp', 'xgb', 'lgb', 'cat'], 
                      help='Model type to use for prediction')
    
    # Common arguments
    parser.add_argument('--split', action='store_true', help='Split data into train/test sets')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    
    # NLSQ model arguments
    parser.add_argument('--umax_init', type=float, default=1.0, help='Initial guess for Umax parameter')
    parser.add_argument('--alpha_init', type=float, default=1.0, help='Initial guess for alpha parameter')
    
    # Ensemble model arguments
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees/estimators')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth of trees')
    parser.add_argument('--min_samples_split', type=int, default=2, help='Min samples to split a node')
    
    # Neural network arguments
    parser.add_argument('--hidden_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--neurons', type=int, default=10, help='Neurons per hidden layer')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'tanh', 'sigmoid'],
                       help='Activation function')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    
    # PINN specific arguments
    parser.add_argument('--physics_weight', type=float, default=1.0, help='Weight for physics loss term')
    
    # Time series model arguments
    parser.add_argument('--units', type=int, default=50, help='Number of units in recurrent layer')
    parser.add_argument('--layers', type=int, default=1, help='Number of recurrent layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--look_back', type=int, default=5, help='Number of time steps to look back')
    
    # Polynomial model arguments
    parser.add_argument('--degree', type=int, default=3, help='Degree of polynomial')
    
    # SVR model arguments
    parser.add_argument('--kernel', type=str, default='rbf', 
                       choices=['linear', 'poly', 'rbf', 'sigmoid'], 
                       help='Kernel type for SVR')
    parser.add_argument('--C', type=float, default=1.0, help='Regularization parameter')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon in epsilon-SVR model')
    parser.add_argument('--gamma', type=str, default='scale', help='Kernel coefficient')
    
    # Gaussian Process model arguments
    parser.add_argument('--kernel_type', type=str, default='rbf', choices=['rbf', 'matern'],
                       help='Kernel type for Gaussian Process')
    parser.add_argument('--length_scale', type=float, default=1.0, 
                       help='Length scale parameter for GP kernel')
    parser.add_argument('--gp_alpha', type=float, default=1e-10,
                       help='Noise level parameter for GP (renamed to avoid conflict)')
    
    # Advanced GBM arguments
    parser.add_argument('--subsample', type=float, default=1.0, 
                       help='Subsample ratio of training data')
    
    args = parser.parse_args()
    return args