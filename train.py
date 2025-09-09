# train.py
import tensorflow as tf
import numpy as np

# Generate dummy dataset 
X = np.random.rand(500, 10)   # 500 samples, 10 features
y = np.random.randint(0, 2, size=(500,))  # Binary labels (0 or 1)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(10,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# Save trained model
model.save("my_model.keras")
print("✅ Model saved as my_model.keras")
