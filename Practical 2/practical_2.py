import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt


# ============================================================
# 1. Load the MNIST Training and Testing Data
# ============================================================

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)


# ============================================================
# 2. Define the Feed-Forward Neural Network
# ============================================================

model = Sequential([
    # Convert 28x28 image into 784-dimensional vector
    Flatten(input_shape=(28, 28)),

    # Hidden layer
    Dense(128, activation="relu"),

    # Output layer - 10 classes (digits 0-9)
    Dense(10, activation="softmax")
])

print("\nModel Summary:")
model.summary()


# ============================================================
# 3. Compile the Model using SGD
# ============================================================

model.compile(
    optimizer=SGD(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 4. Train the Model
# ============================================================

print("\nTraining the model...")

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_data=(x_test, y_test)
)


# ============================================================
# 5. Evaluate the Network
# ============================================================

print("\nEvaluating the model...")

test_loss, test_acc = model.evaluate(
    x_test,
    y_test,
    verbose=2
)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")


# ============================================================
# 6. Plot Training and Validation Accuracy
# ============================================================

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)


# ============================================================
# 7. Plot Training and Validation Loss
# ============================================================

plt.subplot(1, 2, 2)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()