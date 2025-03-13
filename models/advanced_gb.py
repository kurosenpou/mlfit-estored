import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import curve_fit

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from sklearn.ensemble import GradientBoostingRegressor

def two_param_model(Wp, Umax, alpha):
    return Umax * (1.0 - np.exp(-(Wp / (alpha * Umax))))

#----------------------------------------
# 9. 高度勾配ブースティングモデル
#----------------------------------------
class AdvancedGBModel(BaseEstimator, RegressorMixin):
    """
    高度な勾配ブースティングモデル：
    XGBoost、LightGBM、CatBoostから選択可能
    """
    def __init__(self, 
                 boosting_type='xgb', 
                 n_estimators=100, 
                 learning_rate=0.1, 
                 max_depth=6,
                 random_state=42):
        self.boosting_type = boosting_type
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.history = {'loss': []}
        self._Umax = None
        self._alpha = None
        
    def fit(self, X, y):
        # ブースターの選択
        if self.boosting_type == 'xgb' and XGBOOST_AVAILABLE:
            self.model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
        elif self.boosting_type == 'lgb' and LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
        elif self.boosting_type == 'cat' and CATBOOST_AVAILABLE:
            self.model = cb.CatBoostRegressor(
                iterations=self.n_estimators,
                learning_rate=self.learning_rate,
                depth=self.max_depth,
                random_seed=self.random_state,
                verbose=0
            )
        else:
            # フォールバックとして標準のGBDTを使用
            print(f"{self.boosting_type} not available, using scikit-learn GradientBoostingRegressor instead.")
            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
        
        # モデル学習
        self.model.fit(X, y)
        
        # 物理モデルにフィットして解釈可能なパラメータを取得
        Wp_values = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = self.model.predict(Wp_values)
        
        try:
            popt, _ = curve_fit(two_param_model, Wp_values.ravel(), y_pred, p0=(1.0, 1.0))
            self._Umax, self._alpha = popt
        except:
            # If curve fitting fails, use some reasonable defaults
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
