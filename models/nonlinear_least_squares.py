import numpy as np
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, RegressorMixin

def two_param_model(Wp, Umax, alpha):
    return Umax * (1.0 - np.exp(-(alpha * Wp / Umax)))


#----------------------------------------
# 1. 非線形最小二乗法（物理モデル直接フィット）
#----------------------------------------
class NonlinearLeastSquaresModel(BaseEstimator, RegressorMixin):
    """
    2パラメータ (Umax, alpha) の非線形回帰モデル:
    U_stored = Umax * [1 - exp(- alpha*(Wp / Umax))].
    
    Parameters
    ----------
    initial_guess : tuple, default=(1.0, 1.0)
        Initial guess for (Umax, alpha) parameters
    """
    def __init__(self, initial_guess=(1.0, 1.0)):
        self.initial_guess = initial_guess
        self.params_ = None  # (Umax, alpha)
        self.history = {'loss': []}  # Keep interface consistent with other models
        print(f"Initial guess: Umax={initial_guess[0]}, alpha={initial_guess[1]}")
        
    def fit(self, X, y):
        Wp = X.reshape(-1)
        popt, pcov = curve_fit(two_param_model, Wp, y, p0=self.initial_guess)
        self.params_ = popt  # (Umax, alpha)
        
        # Calculate final loss for consistency with other models
        y_pred = self.predict(X)
        loss = np.mean((y - y_pred) ** 2)
        self.history['loss'].append(loss)
        return self

    def predict(self, X):
        Umax, alpha = self.params_
        Wp = X.reshape(-1)
        return two_param_model(Wp, Umax, alpha)

    def print_results(self):
        """Print the final fitted parameter values."""
        print("\n" + "="*50)
        print("NONLINEAR LEAST SQUARES MODEL RESULTS")
        print("="*50)
        print(f"Umax = {self.Umax_:.6f}")
        print(f"alpha = {self.alpha_:.6f}")
        print("="*50 + "\n")

    @property
    def Umax_(self):
        return self.params_[0] if self.params_ is not None else None

    @property
    def alpha_(self):
        return self.params_[1] if self.params_ is not None else None