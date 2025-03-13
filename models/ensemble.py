import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.optimize import curve_fit

def two_param_model(Wp, Umax, alpha):
    return Umax * (1.0 - np.exp(-(Wp / (alpha * Umax))))


#----------------------------------------
# 2. ランダムフォレスト / GBDT
#----------------------------------------
class EnsembleModel(BaseEstimator, RegressorMixin):
    """
    ランダムフォレストまたはGBDT (Gradient Boosting Decision Tree)
    を用いたU_stored予測モデル
    """
    def __init__(self, model_type='rf', n_estimators=100, max_depth=None, 
                 learning_rate=0.1, random_state=42):
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.history = {'loss': []}  # Keep interface consistent with other models
        self._Umax = None
        self._alpha = None
        
    def fit(self, X, y):
        # Define the model based on model_type
        if self.model_type.lower() == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
        else:  # 'gbdt'
            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state
            )
        
        # Fit the model
        self.model.fit(X, y)
        
        # Estimate Umax and alpha by fitting the physical model to our predictions
        # This is a heuristic approach to get interpretable parameters from black-box models
        Wp_values = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = self.model.predict(Wp_values)
        
        try:
            # Fit physical model to our predictions
            popt, _ = curve_fit(two_param_model, Wp_values.ravel(), y_pred, p0=(1.0, 1.0))
            self._Umax, self._alpha = popt
        except:
            # If curve fitting fails, use some reasonable defaults
            self._Umax = max(y)
            self._alpha = 1.0
        
        # Calculate loss
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