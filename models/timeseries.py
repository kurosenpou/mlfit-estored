import numpy as np
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import curve_fit

def two_param_model(Wp, Umax, alpha):
    return Umax * (1.0 - np.exp(-(alpha * Wp / Umax)))


#----------------------------------------
# 5. RNN/LSTM等の時系列モデル
#----------------------------------------
class TimeSeriesModel(BaseEstimator, RegressorMixin):
    """
    RNN/LSTMを使った時系列モデル
    """
    def __init__(self, cell_type='lstm', units=50, 
                 learning_rate=0.001, batch_size=32, epochs=1000, patience=50):
        self.cell_type = cell_type
        self.units = units
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.model = None
        self.history = None
        self._Umax = None
        self._alpha = None
        self.sequence_length = 10  # Default sequence length for time series
    
    def build_model(self, input_shape):
        """RNN/LSTMモデルの構築"""
        inputs = layers.Input(shape=input_shape)
        
        # Choose cell type
        if self.cell_type.lower() == 'lstm':
            x = layers.LSTM(self.units)(inputs)
        else:  # 'rnn' or other
            x = layers.SimpleRNN(self.units)(inputs)
        
        outputs = layers.Dense(1)(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        
        self.model = model
        return model
    
    def create_sequences(self, X, y):
        """Create sequence data for RNN/LSTM"""
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        return np.array(X_seq), np.array(y_seq)
    
    def fit(self, X, y):
        # For time series models, we need to create sequences
        # Assuming X is already time-ordered
        X_seq, y_seq = self.create_sequences(X, y)
        
        if X_seq.shape[0] == 0:  # Not enough data for sequences
            raise ValueError(f"Not enough data for sequence length {self.sequence_length}")
        
        if self.model is None:
            self.build_model((self.sequence_length, X.shape[1]))
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True
        )
        
        # Train the model
        history = self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        self.history = history.history
        
        # Estimate Umax and alpha by fitting the physical model to our predictions
        X_flat = X.reshape(-1, 1) if len(X.shape) == 1 else X
        y_pred = self.predict(X_flat)
        
        try:
            # Fit physical model to our predictions
            popt, _ = curve_fit(two_param_model, X_flat.ravel(), y_pred[self.sequence_length:], p0=(1.0, 1.0))
            self._Umax, self._alpha = popt
        except:
            # If curve fitting fails, use some reasonable defaults
            self._Umax = max(y)
            self._alpha = 1.0
        
        return self
    
    def predict(self, X):
        """
        Predict U_stored values
        Note: First 'sequence_length' predictions will be NaN
        """
        X_flat = X.reshape(-1, 1) if len(X.shape) == 1 else X
        result = np.full(X_flat.shape[0], np.nan)
        
        # Create sequences for prediction
        X_seq = []
        for i in range(X_flat.shape[0] - self.sequence_length):
            X_seq.append(X_flat[i:i + self.sequence_length])
        
        if X_seq:
            X_seq = np.array(X_seq)
            preds = self.model.predict(X_seq).flatten()
            result[self.sequence_length:] = preds
        
        return result
    
    @property
    def Umax_(self):
        return self._Umax
    
    @property
    def alpha_(self):
        return self._alpha
