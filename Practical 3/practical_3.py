import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 
from tensorflow.keras.datasets import cifar10 
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import matplotlib.pyplot as plt 
import numpy as np 
import os

# --- a. Loading and Preprocessing the Image Data --- 
# Load CIFAR-10 dataset 
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 

# Normalize pixel values to be between 0 and 1 
x_train = x_train.astype('float32') / 255.0 
x_test = x_test.astype('float32') / 255.0 

# Convert labels to one-hot encoding 
num_classes = 10 
y_train = to_categorical(y_train, num_classes) 
y_test = to_categorical(y_test, num_classes) 

print(f"Original training data shape: {x_train.shape}") # (50000, 32, 32, 3) 
print(f"Original training labels shape: {y_train.shape}") # (50000, 10) 

# Data Augmentation (optional but recommended) 
datagen = ImageDataGenerator( 
    rotation_range=15, 
    width_shift_range=0.1, 
    height_shift_range=0.1, 
    horizontal_flip=True, 
    zoom_range=0.1 
) 

# Fit the generator on the training data 
datagen.fit(x_train) 

# --- b. Defining the Model’s Architecture (CNN) --- 
model = Sequential([ 
    # Convolutional Layer 1 
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)), 
    MaxPooling2D((2, 2)), 
    
    # Convolutional Layer 2 
    Conv2D(64, (3, 3), activation='relu', padding='same'), 
    MaxPooling2D((2, 2)), 
    
    # Convolutional Layer 3 
    Conv2D(128, (3, 3), activation='relu', padding='same'), 
    MaxPooling2D((2, 2)), 
    
    # Flatten the output for Dense layers 
    Flatten(), 
    
    # Dense layer for classification 
    Dense(512, activation='relu'), 
    Dropout(0.5), # Dropout for regularization 
    Dense(num_classes, activation='softmax') # Output layer 
]) 

model.summary() 

# --- c. Training the Model --- 
# Compile the model 
model.compile(optimizer='adam', # Adam is a popular choice 
              loss='categorical_crossentropy', 
              metrics=['accuracy']) 

# Training parameters 
batch_size = 64 
epochs = 25 # Reduced epochs for quicker demonstration, increase for better results 

print("\nTraining the model with data augmentation...") 
# Use the augmented data for training 
history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), 
                    steps_per_epoch=len(x_train) // batch_size, 
                    epochs=epochs, 
                    validation_data=(x_test, y_test)) # Validation on original test data 

# --- d. Estimating the Model’s Performance --- 
print("\nEvaluating the model on the test set...") 
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0) 
print(f'Test Loss: {test_loss:.4f}') 
print(f'Test Accuracy: {test_acc:.4f}') 

# Plot training history 
plt.figure(figsize=(14, 5)) 

# Plotting Accuracy 
plt.subplot(1, 2, 1) 
plt.plot(history.history['accuracy'], label='Training Accuracy') 
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') 
plt.title('Training and Validation Accuracy') 
plt.xlabel('Epoch') 
plt.ylabel('Accuracy') 
plt.legend() 
plt.grid(True) 

# Plotting Loss 
plt.subplot(1, 2, 2) 
plt.plot(history.history['loss'], label='Training Loss') 
plt.plot(history.history['val_loss'], label='Validation Loss') 
plt.title('Training and Validation Loss') 
plt.xlabel('Epoch') 
plt.ylabel('Loss') 
plt.legend() 
plt.grid(True) 

plt.tight_layout() 

# I've saved the plot to a file instead of using plt.show() which halts execution
plot_path = os.path.join(os.path.dirname(__file__), 'training_history.png')
plt.savefig(plot_path)
print(f"\nTraining history plot saved to: {plot_path}")

# Displaying class names for context 
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck'] 
print("\nClass names:", class_names)
