import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

def two_param_model(Wp, Umax, alpha):
    return Umax * (1.0 - np.exp(-(Wp / (alpha * Umax))))

#----------------------------------------
# 6. 多項式回帰
#----------------------------------------
class PolynomialRegressionModel(BaseEstimator, RegressorMixin):
    """
    多項式回帰モデル：
    シンプルな低次多項式による回帰
    """
    def __init__(self, degree=3, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias
        self.model = None
        self.history = {'loss': []}
        self._Umax = None
        self._alpha = None
        
    def fit(self, X, y):
        # 多項式特徴量を生成してから線形回帰
        self.model = Pipeline([
            ('poly', PolynomialFeatures(degree=self.degree, include_bias=self.include_bias)),
            ('linear', LinearRegression(fit_intercept=True))
        ])
        
        self.model.fit(X, y)
        
        # 物理モデルにフィットして解釈可能なパラメータを取得
        Wp_values = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = self.model.predict(Wp_values)
        
        try:
            popt, _ = curve_fit(two_param_model, Wp_values.ravel(), y_pred, p0=(1.0, 1.0))
            self._Umax, self._alpha = popt
        except:
            self._Umax = max(y)
            self._alpha = 1.0
        
        # 損失を計算
        y_pred = self.predict(X)
        loss = np.mean((y - y_pred) ** 2)
        self.history['loss'].append(loss)
        
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def print_results(self):
        """Print polynomial coefficients and equivalent physical parameters."""
        print("\n" + "="*50)
        print("POLYNOMIAL MODEL RESULTS")
        print("="*50)
        print(f"Polynomial coefficients: {self.coef_}")
    
        # Calculate approximate Umax and alpha from polynomial coefficients
        # This is theoretical and depends on your approach
        approx_umax = self._approximate_umax()
        approx_alpha = self._approximate_alpha()
    
        print(f"Approximated Umax ≈ {approx_umax:.6f}")
        print(f"Approximated alpha ≈ {approx_alpha:.6f}")
        print("Note: Values are approximations based on polynomial fit")
        print("="*50 + "\n")
    
    def _approximate_umax(self):
        """Approximate Umax from polynomial coefficients."""
        # Implementation depends on your polynomial approach
        # Placeholder implementation
        return 0.0
    
    def _approximate_alpha(self):
        """Approximate alpha from polynomial coefficients."""
        # Implementation depends on your polynomial approach
        # Placeholder implementation
        return 0.0

    @property
    def Umax_(self):
        return self._Umax
    
    @property
    def alpha_(self):
        return self._alpha
