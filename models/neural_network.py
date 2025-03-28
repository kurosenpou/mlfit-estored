import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import curve_fit

def two_param_model(Wp, Umax, alpha):
    return Umax * (1.0 - np.exp(-(alpha * Wp / Umax)))


#----------------------------------------
# 3. ニューラルネットワーク
#----------------------------------------
class SimpleNeuralNetwork(BaseEstimator, RegressorMixin):
    """
    シンプルなニューラルネットワークモデル
    """
    def __init__(self, hidden_layers=[20, 20], activation='tanh', 
                 learning_rate=0.001, batch_size=32, epochs=1000, patience=50):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.model = None
        self.history = None
        self._Umax = None
        self._alpha = None
    
    def build_model(self):
        """ニューラルネットワークの構築"""
        inputs = layers.Input(shape=(1,))
        
        x = inputs
        for units in self.hidden_layers:
            x = layers.Dense(units, activation=self.activation)(x)
        
        outputs = layers.Dense(1)(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        
        self.model = model
        return model
    
    def fit(self, X, y):
        if self.model is None:
            self.build_model()
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True
        )
        
        # Train the model
        history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        self.history = history.history
        
        # Estimate Umax and alpha by fitting the physical model to our predictions
        Wp_values = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = self.model.predict(Wp_values)
        
        try:
            # Fit physical model to our predictions
            popt, _ = curve_fit(two_param_model, Wp_values.ravel(), y_pred.ravel(), p0=(1.0, 1.0))
            self._Umax, self._alpha = popt
        except:
            # If curve fitting fails, use some reasonable defaults
            self._Umax = max(y)
            self._alpha = 1.0
        
        return self
    
    def predict(self, X):
        return self.model.predict(X).flatten()
    
    def print_results(self):
        """Print the equivalent Umax and alpha parameters."""
        print("\n" + "="*50)
        print("NEURAL NETWORK MODEL RESULTS")
        print("="*50)
        # Calculate equivalent Umax and alpha if possible
        # This is an example - implementation depends on your NN architecture
        if hasattr(self, 'model_'):
            # Extract weights from trained model to estimate Umax and alpha
            # This is theoretical and depends on your model structure
            weights = self.model_.get_weights()
            estimated_umax = self._estimate_umax(weights)
            estimated_alpha = self._estimate_alpha(weights)
            print(f"Estimated Umax ≈ {estimated_umax:.6f}")
            print(f"Estimated alpha ≈ {estimated_alpha:.6f}")
        print("Note: Neural network parameters are indirect approximations")
        print("="*50 + "\n")
    
    def _estimate_umax(self, weights):
        """Estimate Umax parameter from NN weights."""
        # Implementation depends on your NN architecture
        # Placeholder implementation
        return 0.0
    
    def _estimate_alpha(self, weights):
        """Estimate alpha parameter from NN weights."""
        # Implementation depends on your NN architecture
        # Placeholder implementation
        return 0.0

    @property
    def Umax_(self):
        return self._Umax
    
    @property
    def alpha_(self):
        return self._alpha