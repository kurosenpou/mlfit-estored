import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.optimize import curve_fit

def two_param_model(Wp, Umax, alpha):
    return Umax * (1.0 - np.exp(-(Wp / (alpha * Umax))))

#----------------------------------------
# 8. ガウス過程回帰
#----------------------------------------
class GaussianProcessModel(BaseEstimator, RegressorMixin):
    """
    ガウス過程回帰：
    不確実性推定も可能なベイズ的アプローチ
    """
    def __init__(self, kernel_type='rbf', length_scale=1.0, alpha=1e-10):
        self.kernel_type = kernel_type
        self.length_scale = length_scale
        self.alpha = alpha
        self.model = None
        self.history = {'loss': []}
        self._Umax = None
        self._alpha = None
        
    def fit(self, X, y):
        # カーネル選択
        if self.kernel_type == 'rbf':
            kernel = ConstantKernel() * RBF(length_scale=self.length_scale)
        elif self.kernel_type == 'matern':
            kernel = ConstantKernel() * Matern(length_scale=self.length_scale, nu=1.5)
        else:
            kernel = None  # デフォルトカーネルを使用
        
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            n_restarts_optimizer=10,
            random_state=42
        )
        
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
