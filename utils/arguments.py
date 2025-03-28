import argparse
import sys
import textwrap

def show_nlls_help():
    """Display help specific to nonlinear least squares model."""
    help_text = """
    Nonlinear Least Squares Model Options:
    -------------------------------------
    A physical model that directly fits the equation: U_stored = Umax * [1 - exp(- alpha*(Wp / Umax))]
    
    Model-specific arguments:
      --umax_init FLOAT   Initial guess for Umax parameter (default: 1.0)
      --alpha_init FLOAT  Initial guess for alpha parameter (default: 1.0)
    
    Example:
      mlfit data.txt results.png nlls --umax_init 5.0 --alpha_init 0.5
    """
    print(textwrap.dedent(help_text))
    sys.exit(0)

def show_nn_help():
    """Display help specific to neural network model."""
    help_text = """
    Neural Network Model Options:
    ---------------------------
    A neural network model for fitting non-linear relationships.
    
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

def show_polynomial_help():
    """Display help specific to polynomial model."""
    help_text = """
    Polynomial Regression Model Options:
    ---------------------------------
    A polynomial regression model that fits data to a polynomial function.
    
    Model-specific arguments:
      --degree INT    Degree of the polynomial (default: 3)
    
    Example:
      mlfit data.txt results.png polynomial --degree 4
    """
    print(textwrap.dedent(help_text))
    sys.exit(0)

def parse_arguments():
    # Check for model-specific help request
    if len(sys.argv) >= 4 and '-help' in sys.argv:
        model_idx = sys.argv.index(sys.argv[3])
        model_type = sys.argv[model_idx]
        
        if model_type == 'nlls':
            show_nlls_help()
        elif model_type == 'nn':
            show_nn_help()
        elif model_type == 'polynomial':
            show_polynomial_help()
    
    # Regular argument parsing
    parser = argparse.ArgumentParser(description='Machine Learning Model for U_stored prediction')
    
    parser.add_argument('input_file', type=str, help='Path to input data file')
    parser.add_argument('output_file', type=str, help='Path to output file')
    parser.add_argument('model', type=str, choices=['nlls', 'nn', 'polynomial'], 
                        help='Model type: nonlinear least squares (nlls), neural network (nn), or polynomial')
    
    # Common arguments
    parser.add_argument('--split', action='store_true', help='Split data into train/test sets')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    
    # NLLS model arguments
    parser.add_argument('--umax_init', type=float, default=1.0, help='Initial guess for Umax parameter')
    parser.add_argument('--alpha_init', type=float, default=1.0, help='Initial guess for alpha parameter')
    
    # Neural network arguments
    parser.add_argument('--hidden_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--neurons', type=int, default=10, help='Neurons per hidden layer')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'tanh', 'sigmoid'],
                        help='Activation function')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    
    # Polynomial model arguments
    parser.add_argument('--degree', type=int, default=3, help='Degree of polynomial')
    
    args = parser.parse_args()
    return args