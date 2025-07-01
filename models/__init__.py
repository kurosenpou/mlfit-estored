from .nonlinear_least_squares import NonlinearLeastSquaresModel
from .ensemble import EnsembleModel
from .neural_network import SimpleNeuralNetwork
from .pinn import UmaxPINN
from .timeseries import TimeSeriesModel
from .polynomial import PolynomialRegressionModel
from .svr_model import SVRModel
from .gaussian_process import GaussianProcessModel
from .advanced_gb import AdvancedGBModel

def select_model(args):
    if args.model_type == "nlsq":
        return NonlinearLeastSquaresModel(
            initial_guess=(args.umax_init, args.alpha_init)
        )
    elif args.model_type in ["rf", "gbdt"]:
        return EnsembleModel(
            model_type=args.model_type,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=args.random_state,
        )
    elif args.model_type == "nn":
        hidden_layers = [args.neurons] * args.hidden_layers
        return SimpleNeuralNetwork(
            hidden_layers=hidden_layers,
            activation=args.activation,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
        )
    elif args.model_type == "pinn":
        hidden_layers = [args.neurons] * args.hidden_layers
        return UmaxPINN(
            hidden_layers=hidden_layers,
            activation=args.activation,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
        )
    elif args.model_type in ["rnn", "lstm"]:
        return TimeSeriesModel(
            cell_type=args.model_type,
            units=args.units,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
        )
    elif args.model_type == "poly":
        return PolynomialRegressionModel(degree=args.degree)
    elif args.model_type == "svr":
        return SVRModel(
            kernel=args.kernel,
            C=args.C,
            epsilon=args.epsilon,
            gamma=args.gamma,
        )
    elif args.model_type == "gp":
        return GaussianProcessModel(
            kernel_type=args.kernel_type,
            length_scale=args.length_scale,
            alpha=args.gp_alpha,
        )
    elif args.model_type in ["xgb", "lgb", "cat"]:
        return AdvancedGBModel(
            boosting_type=args.model_type,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            random_state=args.random_state,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
