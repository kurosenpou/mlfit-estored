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
        return NonlinearLeastSquaresModel(initial_guess=(args.umax_init, args.alpha_init))
    elif args.model_type in ["rf", "gbdt"]:
        return EnsembleModel(model_type=args.model_type)
    elif args.model_type == "nn":
        return SimpleNeuralNetwork()
    elif args.model_type == "pinn":
        return UmaxPINN()
    elif args.model_type in ["rnn", "lstm"]:
        return TimeSeriesModel(cell_type=args.model_type)
    elif args.model_type == "poly":
        return PolynomialRegressionModel()
    elif args.model_type == "svr":
        return SVRModel()
    elif args.model_type == "gp":
        return GaussianProcessModel()
    elif args.model_type in ["xgb", "lgb", "cat"]:
        return AdvancedGBModel(boosting_type=args.model_type)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
