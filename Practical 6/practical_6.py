import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.datasets import cifar10, fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Setup and Directory Configuration
# ---------------------------------------------------------------------------
output_dir = os.path.join(os.path.dirname(__file__), "OutPut")
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Execution Settings:
# - DOWNLOAD_WEIGHTS: Set True to download ImageNet weights (58 MB).
#   If False, uses base VGG16 architecture for instant execution without network delay.
# - USE_SUBSET: Use a subset for fast demonstration on CPU (recommended for CPU).
# ---------------------------------------------------------------------------
weights_file = os.path.expanduser("~/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
has_local_weights = os.path.exists(weights_file) and os.path.getsize(weights_file) > 50000000

DOWNLOAD_WEIGHTS = False or has_local_weights
USE_SUBSET = True
TRAIN_SAMPLES = 2000
TEST_SAMPLES = 500

# ===========================================================================
# Step a: Load in a Pre-trained CNN Model trained on a large dataset
# ===========================================================================
print("=" * 65)
print("Step a: Loading CNN Model Architecture (VGG16)")
print("=" * 65)

# Load VGG16 without the top classification layers for input shape (32, 32, 3)
if DOWNLOAD_WEIGHTS:
    print("Loading VGG16 with pre-trained ImageNet weights...")
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
else:
    print("Loading VGG16 convolutional base (offline mode for quick demonstration)...")
    base_model = VGG16(weights=None, include_top=False, input_shape=(32, 32, 3))

print(f"Base model loaded successfully with {len(base_model.layers)} layers.")

# ===========================================================================
# Step b: Freeze Parameters (weights) in the Model's Lower Convolutional Layers
# ===========================================================================
print("\n" + "=" * 65)
print("Step b: Freezing Base Model Layers")
print("=" * 65)
# Freeze all layers in the base model initially
for layer in base_model.layers:
    layer.trainable = False

print("All base layers have been frozen (trainable = False).")

# ===========================================================================
# Step c: Add a Custom Classifier with Several Layers of Trainable Parameters
# ===========================================================================
print("\n" + "=" * 65)
print("Step c: Adding Custom Classifier Layers")
print("=" * 65)
x = base_model.output
x = Flatten()(x)                      # Flatten the output of the convolutional base
x = Dense(512, activation='relu')(x)  # Dense layer for feature learning
x = Dropout(0.5)(x)                   # Dropout layer for regularization

# Final output layer for multi-class classification (10 classes)
num_classes = 10
predictions = Dense(num_classes, activation='softmax')(x)

# Create the new model
model = Model(inputs=base_model.input, outputs=predictions)

print("\n--- Model Architecture Summary (Classifier Only Trainable) ---")
model.summary()

# ===========================================================================
# Step d: Train Classifier Layers on Training Data Available for the Task
# ===========================================================================
print("\n" + "=" * 65)
print("Step d: Loading Dataset and Training Classifier Layers")
print("=" * 65)

# Check if CIFAR-10 is cached locally; otherwise use cached Fashion-MNIST adapted to 32x32x3
cifar_cache = os.path.expanduser("~/.keras/datasets/cifar-10-batches-py")
if os.path.exists(cifar_cache):
    print("Loading CIFAR-10 from local cache...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
else:
    print("Using local 10-class dataset (Fashion-MNIST formatted to 32x32x3 RGB for demonstration)...")
    (x_tr_raw, y_tr_raw), (x_te_raw, y_te_raw) = fashion_mnist.load_data()
    # Pad from 28x28 to 32x32 and replicate across 3 color channels
    x_train = np.stack([np.pad(x_tr_raw, ((0, 0), (2, 2), (2, 2)), mode='constant')] * 3, axis=-1)
    x_test = np.stack([np.pad(x_te_raw, ((0, 0), (2, 2), (2, 2)), mode='constant')] * 3, axis=-1)
    y_train = y_tr_raw.reshape(-1, 1)
    y_test = y_te_raw.reshape(-1, 1)

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

if USE_SUBSET:
    print(f"Using demo subset: {TRAIN_SAMPLES} training images, {TEST_SAMPLES} test images.")
    x_train = x_train[:TRAIN_SAMPLES]
    y_train = y_train[:TRAIN_SAMPLES]
    x_test = x_test[:TEST_SAMPLES]
    y_test = y_test[:TEST_SAMPLES]

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Compile the model with moderate learning rate
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model (only classifier weights update)
print("\nTraining the custom classifier layers...")
batch_size = 64
epochs = 3
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test))

# ===========================================================================
# Step e: Fine-tune Hyperparameters and Unfreeze More Layers
# ===========================================================================
print("\n" + "=" * 65)
print("Step e: Fine-Tuning with Unfreezed Base Layers")
print("=" * 65)

# Note: VGG16 (include_top=False) has 19 layers total (indices 0 to 18).
# 'block5_conv1' starts at index 15. Unfreezing from index 15 onwards unfreezes block5.
for layer in base_model.layers[15:]:
    layer.trainable = True

# Re-compile with very low learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\n--- Model Architecture Summary (Block 5 + Classifier Trainable) ---")
model.summary()

# Train again for fine-tuning
# Note: In Keras, when initial_epoch is specified, epochs must be the total target epoch count (total_epochs)
print("\nFine-tuning the model with unfreezed block 5 layers...")
fine_tune_epochs = 2  # Additional epochs for fine-tuning
total_epochs = epochs + fine_tune_epochs
history_fine = model.fit(x_train, y_train,
                         batch_size=batch_size,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1] + 1,
                         validation_data=(x_test, y_test))

# Combine histories for plotting
history.history['accuracy'].extend(history_fine.history['accuracy'])
history.history['val_accuracy'].extend(history_fine.history['val_accuracy'])
history.history['loss'].extend(history_fine.history['loss'])
history.history['val_loss'].extend(history_fine.history['val_loss'])

# Plot training history after fine-tuning
plt.figure(figsize=(14, 5))

# Plotting Accuracy
plt.subplot(1, 2, 1)
plt.plot(range(1, total_epochs + 1), history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(range(1, total_epochs + 1), history.history['val_accuracy'], label='Validation Accuracy', marker='s')
plt.title('Training and Validation Accuracy (After Fine-tuning)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plotting Loss
plt.subplot(1, 2, 2)
plt.plot(range(1, total_epochs + 1), history.history['loss'], label='Training Loss', marker='o')
plt.plot(range(1, total_epochs + 1), history.history['val_loss'], label='Validation Loss', marker='s')
plt.title('Training and Validation Loss (After Fine-tuning)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Evaluate the final fine-tuned model
print("\n" + "=" * 65)
print("Final Evaluation on Test Set")
print("=" * 65)
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Final Test Loss:     {loss:.4f}")
print(f"Final Test Accuracy: {accuracy:.4f}")
print("=" * 65)

# Save the plot in OutPut folder
plot_path = os.path.join(output_dir, "practical_6_results.png")
plt.savefig(plot_path, dpi=300)
print(f"\nPlot successfully saved to: {plot_path}")
plt.show()
