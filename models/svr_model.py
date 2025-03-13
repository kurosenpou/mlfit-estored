import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.svm import SVR
from scipy.optimize import curve_fit

def two_param_model(Wp, Umax, alpha):
    return Umax * (1.0 - np.exp(-(Wp / (alpha * Umax))))

#----------------------------------------
# 7. SVMによる回帰
#----------------------------------------
class SVRModel(BaseEstimator, RegressorMixin):
    """
    サポートベクターマシンによる回帰：
    カーネルを用いた非線形回帰
    """
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.model = None
        self.history = {'loss': []}
        self._Umax = None
        self._alpha = None
        
    def fit(self, X, y):
        self.model = SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
            gamma=self.gamma
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
