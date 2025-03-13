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
    
    @property
    def Umax_(self):
        return self._Umax
    
    @property
    def alpha_(self):
        return self._alpha
