import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.base import BaseEstimator, RegressorMixin


#----------------------------------------
# 4. 物理インフォームドNN
#----------------------------------------
class UmaxPINN(BaseEstimator, RegressorMixin):
    """
    Physics-Informed Neural Network (PINN) for the Umax model:
    U_stored = Umax * [1 - exp(- (Wp / (alpha * Umax)))]
    """
    def __init__(self, 
                 hidden_layers=[20, 20], 
                 activation='tanh', 
                 learning_rate=0.001, 
                 batch_size=32, 
                 epochs=1000, 
                 patience=50):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.model = None
        self.history = None
        self.Umax_var = tf.Variable(1.0, dtype=tf.float32, name="Umax")
        self.alpha_var = tf.Variable(1.0, dtype=tf.float32, name="alpha")

    def build_model(self):
        """Build the neural network model"""
        inputs = layers.Input(shape=(1,))  # Wp input
        
        # Initialize hidden layers
        x = inputs
        for units in self.hidden_layers:
            x = layers.Dense(units, activation=self.activation)(x)
        
        # Output layer - no activation for raw output
        outputs = layers.Dense(1)(x)
        
        # Create the model
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        # Define optimizer
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        
        # Compile model - custom loss will be used in train_step
        self.model.compile(optimizer=optimizer)
        
        return self.model

    def physics_loss(self, Wp, U_stored):
        """
        Physics-informed loss function enforcing the model:
        U_stored = Umax * [1 - exp(- (Wp / (alpha * Umax)))]
        """
        # Get neural network prediction
        nn_output = self.model(Wp, training=True)
        
        # Calculate the physical model's prediction
        Umax = tf.abs(self.Umax_var)  # Ensure positivity
        alpha = tf.abs(self.alpha_var)  # Ensure positivity
        physical_term = Umax * (1.0 - tf.exp(-(Wp / (alpha * Umax))))
        
        # MSE between NN output and physical model
        physics_loss = tf.reduce_mean(tf.square(nn_output - physical_term))
        
        # MSE between NN output and actual data
        data_loss = tf.reduce_mean(tf.square(nn_output - U_stored))
        
        # Combine losses (could use weighting factors if desired)
        total_loss = data_loss + physics_loss
        
        return total_loss

    @tf.function  # Use tf.function to optimize for GPU execution
    def train_step(self, Wp, U_stored):
        """Custom training step with physics constraints"""
        with tf.GradientTape() as tape:
            loss = self.physics_loss(Wp, U_stored)
            
        # Get gradients of loss with respect to trainable variables
        trainable_vars = self.model.trainable_variables + [self.Umax_var, self.alpha_var]
        gradients = tape.gradient(loss, trainable_vars)
        
        # Apply gradients
        self.model.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return loss

    def fit(self, X, y):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        # Convert inputs to TensorFlow tensors and ensure they're on GPU if available
        Wp = tf.convert_to_tensor(X, dtype=tf.float32)
        U_stored = tf.convert_to_tensor(y, dtype=tf.float32)
        
        # Use tf.data.Dataset for more efficient data handling on GPU
        dataset = tf.data.Dataset.from_tensor_slices((Wp, U_stored))
        dataset = dataset.shuffle(buffer_size=len(X)).batch(self.batch_size)
        
        # Training loop
        history = {'loss': []}
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            total_loss = 0
            num_batches = 0
            
            # Process in batches using the dataset
            for batch_Wp, batch_U_stored in dataset:
                # Perform one training step
                batch_loss = self.train_step(batch_Wp, batch_U_stored)
                total_loss += batch_loss
                num_batches += 1
            
            # Average loss for the epoch
            avg_loss = total_loss / num_batches
            history['loss'].append(avg_loss.numpy())
            
            # Print progress every 100 epochs
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}, "
                      f"Umax: {abs(self.Umax_var.numpy()):.6f}, alpha: {abs(self.alpha_var.numpy()):.6f}")
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        self.history = history
        return self
    
    @tf.function  # Use tf.function for faster prediction on GPU
    def _predict_gpu(self, Wp):
        """Optimized prediction function for GPU"""
        return self.model(Wp)
    
    def predict(self, X):
        """Predict using the trained model"""
        Wp = tf.convert_to_tensor(X, dtype=tf.float32)
        return self._predict_gpu(Wp).numpy().flatten()
    
    @property
    def Umax_(self):
        """Get the optimized Umax parameter"""
        return abs(self.Umax_var.numpy()) if self.Umax_var is not None else None

    @property
    def alpha_(self):
        """Get the optimized alpha parameter"""
        return abs(self.alpha_var.numpy()) if self.alpha_var is not None else None