#!/usr/bin/env python
# coding: utf-8
# @author: konain

import os
import cv2
import random
import shutil

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report


## CIFAR-10 Dataset

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Display basic information
print(f"Train images: {x_train.shape}, Train labels: {y_train.shape}")
print(f"Test images: {x_test.shape}, Test labels: {y_test.shape}")

X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Normalize the data
X_train_normalized = X_train / 255.0
X_val_normalized = X_val / 255.0
X_test_normalized = x_test / 255.0

# Convert labels to one-hot encoding
Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=10)
Y_val = tf.keras.utils.to_categorical(Y_val, num_classes=10)
Y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Compute the number of samples
train_size = X_train.shape[0]
val_size = X_val.shape[0]
total_size = train_size + val_size

# Display the training and validation data size in percentage
train_percent = (train_size / total_size) * 100
val_percent = (val_size / total_size) * 100

# Display a Pie Chart to visualize the distribution
plt.figure(figsize=(8, 6))
labels = ['Training set', 'Validation set']
sizes = [train_percent, val_percent]
colors = ['lightblue', 'lightcoral']
explode = (0.1, 0)  # To highlight the "Training set" part

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1d%%', shadow=True, startangle=90)
plt.title("Split of samples between Training and Validation set", fontsize=12)
plt.show()


## Image classification on the CIFAR-10 handwritten digits recognition dataset

### Fully connected layers

# Define the FC model
FC_model = tf.keras.Sequential([
    tf.keras.Input(shape=(32, 32, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

FC_model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001), 
    loss=tf.keras.losses.CategoricalCrossentropy(), 
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
)

# Train the model
history = FC_model.fit(x=X_train_normalized, y=Y_train, batch_size=128, epochs=30, verbose='auto', 
                    validation_data=(X_val_normalized, Y_val), shuffle=True)

plt.figure(figsize=(6,5))
plt.plot(history.epoch, history.history["loss"], 'g', label='Training loss')
plt.plot(history.epoch, history.history['val_loss'], 'r', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid('on')
plt.title('Reduction of the cost function')
plt.legend()
plt.show()

print("Evaluation: ")
score1 = FC_model.evaluate(x=X_val_normalized, y=Y_val)
print(score1)


# Convert Y_val from one-hot encoding to an integer vector
Y_val_integer = np.argmax(Y_val, axis=1)

y_pred_val = FC_model.predict(X_val_normalized)
y_pred_v = np.argmax(y_pred_val, axis=1)

accuracy_val = accuracy_score(Y_val_integer, y_pred_v)
print(f"Accuracy score = {accuracy_val}")


### Fine Tuning

# Define the model
# Use regularization to reduce overfitting
fc_model_reg = tf.keras.Sequential([
    tf.keras.Input(shape=(32, 32, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

fc_model_reg.compile(
    optimizer=tf.keras.optimizers.Adam(0.0005),
    loss=tf.keras.losses.CategoricalCrossentropy(), 
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
)

history = fc_model_reg.fit(x=X_train_normalized, y=Y_train, batch_size=128, epochs=30, verbose='auto', 
                    validation_data=(X_val_normalized, Y_val), shuffle=True)

plt.figure(figsize=(6,5))
plt.plot(history.epoch, history.history["loss"], 'g', label='Training loss')
plt.plot(history.epoch, history.history['val_loss'], 'r', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid('on')
plt.title('Reduction of the cost function')
plt.legend()
plt.show()

print("Evaluation: ")
score2 = fc_model_reg.evaluate(x=X_val_normalized, y=Y_val)
print(score2)


### Prediction and Confusion Matrix
y_predicted = fc_model_reg.predict(X_test_normalized)
y_pred = np.argmax(y_predicted, axis=1)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy score = {accuracy}")

# Confusion matrix
plt.figure(figsize=(6,5))
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(conf_matrix)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# Important metric in classification
print("Classification Report:")
print(classification_report(y_test, y_pred, digits=4))


### Convolutional Neural Network
CNN_model = tf.keras.Sequential([
    tf.keras.Input(shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

CNN_model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(), 
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
)

outputs = CNN_model.fit(x=X_train_normalized, y=Y_train, batch_size=128, epochs=30, verbose='auto', 
                    validation_data=(X_val_normalized, Y_val), shuffle=True)

plt.figure(figsize=(6,5))
plt.plot(outputs.epoch, outputs.history["loss"], 'g', label='Training loss')
plt.plot(outputs.epoch, outputs.history['val_loss'], 'r', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid('on')
plt.title('Reduction of the cost function')
plt.legend()
plt.show()

print("Evaluation: ")
score = CNN_model.evaluate(x=X_val_normalized, y=Y_val)
print(score)


### Fine Tuning
CNN_model_reg = tf.keras.Sequential([
    tf.keras.Input(shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=200, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dense(10, activation='softmax')
])

CNN_model_reg.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
)

# Scheduler: ReduceLROnPlateau
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',          # Monitors validation loss
    factor=0.5,                  # Reduces learning rate by half
    patience=5,                  # Number of epochs without improvement before reduction
    min_lr=1e-6,                 # Lower limit for learning rate
    verbose=1                    # Displays learning rate changes
)

# CNN_model_reg.summary()

outputs = CNN_model_reg.fit(x=X_train_normalized, y=Y_train, batch_size=128, epochs=20, verbose='auto',
                    validation_data=(X_val_normalized, Y_val), shuffle=True, callbacks=[lr_scheduler])


plt.figure(figsize=(6,5))
plt.plot(outputs.epoch, outputs.history["loss"], 'g', label='Training loss')
plt.plot(outputs.epoch, outputs.history['val_loss'], 'r', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid('on')
plt.title('Reduction of the cost function')
plt.legend()
plt.show()

print("Evaluation: ")
score_ = CNN_model_reg.evaluate(x=X_val_normalized, y=Y_val)
print(score_)


### Prediction and Confusion Matrix

y_predicted = CNN_model_reg.predict(X_test_normalized)
y_pred_ = np.argmax(y_predicted, axis=1)

accuracy_ = accuracy_score(y_test, y_pred_)
print(f"Accuracy score = {accuracy_}")

# Confusion matrix
plt.figure(figsize=(6,5))
_conf_matrix = confusion_matrix(y_test, y_pred_)
disp = ConfusionMatrixDisplay(_conf_matrix)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# Important metric in classification
print("Classification Report:")
print(classification_report(y_test, y_pred_, digits=4))

# FC_model.summary()
# CNN_model.summary()


## Pre-trained Network: MobileNet
def prepare_dataset(X, Y, target_size, batch_size=64):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(
        lambda img, label: (tf.image.resize(img, target_size), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Prepare datasets
train_dataset = prepare_dataset(X_train_normalized, Y_train, (224, 224))
val_dataset = prepare_dataset(X_val_normalized, Y_val, (224, 224))
test_dataset = prepare_dataset(X_test_normalized, Y_test, (224, 224))

# Load the pre-trained model without the original fully connected layers
Mobilenet = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base layers
Mobilenet.trainable = False

# Build a sequential model
model = tf.keras.Sequential([
    Mobilenet,  # Pre-trained CNN part
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),  
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
)

# Train the model
outputs = model.fit(train_dataset, epochs=5, validation_data=val_dataset, verbose='auto', shuffle=True)

plt.figure(figsize=(6,5))
plt.plot(outputs.epoch, outputs.history["loss"], 'g', label='Training loss')
plt.plot(outputs.epoch, outputs.history['val_loss'], 'r', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid('on')
plt.title('Reduction of the cost function')
plt.legend()
plt.show()

print("Evaluation: ")
score_mobilenet = model.evaluate(val_dataset)
print(score_mobilenet)


# Function to visualize filters
def visualize_filters(layer):
    """Visualize convolutional filters of a specific layer."""
    weights = layer.get_weights()
    if len(weights) == 2:  # Filters and biases
        filters, biases = weights
    elif len(weights) == 1:  # Only filters
        filters = weights[0]
    else:
        raise ValueError("Unexpected number of weights returned by the layer.")

    n_filters = min(filters.shape[-1], 32)  # Show first 32 filters
    plt.figure(figsize=(6,5))
    fig, axes = plt.subplots(4, 8, figsize=(15, 8))
    
    for i in range(n_filters):
        f = filters[:, :, :, i]
        f_min, f_max = np.min(f), np.max(f)
        f = (f - f_min) / (f_max - f_min)  # Normalize filter values to [0, 1]
        ax = axes[i // 8, i % 8]
        ax.imshow(f[..., 0], cmap='viridis')
        ax.axis('off')
        ax.set_title(f"Filter {i + 1}", fontsize=10)
        
    plt.suptitle("Visualization of Conv1 layer filters", fontsize=16, y=1.03)
    plt.tight_layout()
    plt.show()

# Load MobileNet model
Mobilenet = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Choose the first convolutional layer
conv_layer = Mobilenet.get_layer('Conv1')  # Update the layer name if needed
visualize_filters(conv_layer)

# Function to visualize feature maps
def visualize_feature_maps(model, img):
    layer_name = 'Conv1'  # Layer name to visualize (update if necessary)
    
    # Check if the layer exists
    if layer_name not in [layer.name for layer in model.layers]:
        raise ValueError(f"The layer '{layer_name}' does not exist in the model.")
    
    # Create intermediate model to extract feature maps
    intermediate_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    
    # Get feature maps
    feature_maps = intermediate_model.predict(img)
    
    # Limit to 32 feature maps for visualization
    plt.figure(figsize=(6,5))
    n_maps = min(feature_maps.shape[-1], 32)
    fig, axes = plt.subplots(4, 8, figsize=(15, 8))
    
    for i in range(n_maps):
        ax = axes[i // 8, i % 8]
        ax.imshow(feature_maps[0, :, :, i], cmap='viridis')
        ax.axis('off')
        ax.set_title(f"Map {i + 1}", fontsize=10)
        
    plt.suptitle("Visualization of Conv1 feature maps", fontsize=16, y=1.03)
    plt.tight_layout()
    plt.show()

# Example usage
sample_image = tf.image.resize(x_test[0], (224, 224))  # Resize to match model input
sample_image = tf.expand_dims(sample_image, axis=0)  # Add batch dimension

# Visualize feature maps from the specified layer
visualize_feature_maps(Mobilenet, sample_image)


### Fine tuning
def prepare_dataset(X, Y, target_size, batch_size=64):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(
        lambda img, label: (tf.image.resize(img, target_size), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Prepare datasets
train_dataset = prepare_dataset(X_train_normalized, Y_train, (224, 224))
val_dataset = prepare_dataset(X_val_normalized, Y_val, (224, 224))
test_dataset = prepare_dataset(X_test_normalized, Y_test, (224, 224))

# Load the pre-trained model without original FC layers
Mobilenet = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

Mobilenet.trainable = False

# Unfreeze only the last convolutional layers
for layer in Mobilenet.layers[:-20]:
    layer.trainable = True

# Build a sequential model
model_finet = tf.keras.Sequential([
    Mobilenet,  # Pre-trained CNN part
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model_finet.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
)

# Train the model
outputs = model_finet.fit(train_dataset, epochs=5, validation_data=val_dataset, verbose='auto', shuffle=True)

plt.figure(figsize=(6,5))
plt.plot(outputs.epoch, outputs.history["loss"], 'g', label='Training loss')
plt.plot(outputs.epoch, outputs.history['val_loss'], 'r', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid('on')
plt.title('Reduction of the cost function after fine-tuning')
plt.legend()
plt.show()

print("Evaluation: ")
score_mobile_ft = model_finet.evaluate(val_dataset)
print(score_mobile_ft)

# Function to visualize filters
def visualize_filters(layer):
    """Visualize convolutional filters of a specific layer."""
    weights = layer.get_weights()
    if len(weights) == 2:  # Filters and biases
        filters, biases = weights
    elif len(weights) == 1:  # Only filters
        filters = weights[0]
    else:
        raise ValueError("Unexpected number of weights returned by the layer.")

    n_filters = min(filters.shape[-1], 32)  # Show first 32 filters
    plt.figure(figsize=(6,5))
    fig, axes = plt.subplots(4, 8, figsize=(15, 8))
    
    for i in range(n_filters):
        f = filters[:, :, :, i]
        f_min, f_max = np.min(f), np.max(f)
        f = (f - f_min) / (f_max - f_min)  # Normalize filter values to [0, 1]
        ax = axes[i // 8, i % 8]
        ax.imshow(f[..., 0], cmap='viridis')
        ax.axis('off')
        ax.set_title(f"Filter {i + 1}", fontsize=10)
        
    plt.suptitle("Visualization of Conv1 layer filters", fontsize=16, y=1.03)
    plt.tight_layout()
    plt.show()

# Load MobileNet model
Mobilenet = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Choose the first convolutional layer
conv_layer = Mobilenet.get_layer('Conv1')  # Update the layer name if needed
visualize_filters(conv_layer)

# Function to visualize feature maps
def visualize_feature_maps(model, img):
    layer_name = 'Conv1'  # Layer name to visualize (update if necessary)
    
    # Check if the layer exists
    if layer_name not in [layer.name for layer in model.layers]:
        raise ValueError(f"The layer '{layer_name}' does not exist in the model.")
    
    # Create intermediate model to extract feature maps
    intermediate_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    
    # Get feature maps
    feature_maps = intermediate_model.predict(img)
    
    # Limit to 32 feature maps for visualization
    plt.figure(figsize=(6,5))
    n_maps = min(feature_maps.shape[-1], 32)
    fig, axes = plt.subplots(4, 8, figsize=(15, 8))
    
    for i in range(n_maps):
        ax = axes[i // 8, i % 8]
        ax.imshow(feature_maps[0, :, :, i], cmap='viridis')
        ax.axis('off')
        ax.set_title(f"Map {i + 1}", fontsize=10)
        
    plt.suptitle("Visualization of Conv1 feature maps", fontsize=16, y=1.03)
    plt.tight_layout()
    plt.show()

# Example usage
sample_image = tf.image.resize(x_test[0], (224, 224))  # Resize to match model input
sample_image = tf.expand_dims(sample_image, axis=0)  # Add batch dimension

# Visualize feature maps from the specified layer
visualize_feature_maps(Mobilenet, sample_image)

# UAV Object Detection using transfer learning

# Load data from folders
def load_data_from_folders(image_dir, label_dir):
    images = []
    labels = []

    for image_name in os.listdir(image_dir):
        if image_name.endswith(('.jpg', '.png')):  # Check for valid image formats
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, image_name.replace('.jpg', '.txt').replace('.png', '.txt'))
            
            # Check if the label file exists
            if os.path.exists(label_path):
                # Load labels from the text file
                with open(label_path, 'r') as f:
                    label = list(map(float, f.readline().strip().split()))  # Read a single line
                images.append(image_path)
                labels.append(label)
            else:
                print(f"Missing label for {image_name}. Skipped.")

    return images, np.array(labels)

# Folder paths
image_dir = r"C:\Users\konai\Cranfield\DL_ComputerVision\image" 
label_dir = r"C:\Users\konai\Cranfield\DL_ComputerVision\label"

# Load data
images, labels = load_data_from_folders(image_dir, label_dir)

# Display images with bounding boxes
def plot_images_with_bboxes(images, labels, n_samples=12):
    
    sample_indices = random.sample(range(len(images)), n_samples)
    plt.figure(figsize=(10, 8))
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    axes = axes.flatten()

    for i, idx in enumerate(sample_indices):
        img = cv2.imread(images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = labels[idx]
        obj_class, x, y, width, height = map(int, label)  # Includes the class

        # Draw the bounding box
        img = cv2.rectangle(img, (x, y), (x + width, y + height), (255, 0, 0), 2)
        ax = axes[i]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Label: [{x}, {y}, {width}, {height}]", fontsize=9)

    plt.suptitle("Visualization of 12 images with their Bboxes", fontsize=12, y=1.03)
    plt.tight_layout()
    plt.show()

# Display a sample of images with bounding boxes
plot_images_with_bboxes(images, labels)

def preprocess_and_split(images, labels, train_ratio=0.7, val_ratio=0.15, resize_dim=(224, 224)):
    # Load images and resize, adjust bounding boxes
    normalized_images = []
    adjusted_labels = []  # List for adjusted bounding boxes

    for image_path, label in zip(images, labels):
        img = cv2.imread(image_path)
        if img is None:
            continue  # Skip invalid images
        original_h, original_w = img.shape[:2]  # Original image dimensions
        img = cv2.resize(img, resize_dim)  # Resize the image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # Normalize between 0 and 1

        # Adjust bounding boxes to new dimensions
        obj_class, x, y, width, height = label
        # Resize bounding box coordinates according to resize factor
        x = int(x * resize_dim[0] / original_w)
        y = int(y * resize_dim[1] / original_h)
        width = int(width * resize_dim[0] / original_w)
        height = int(height * resize_dim[1] / original_h)

        # Add the image and adjusted box
        normalized_images.append(img_rgb)
        adjusted_labels.append([obj_class, x, y, width, height])

    # Convert to numpy array
    normalized_images = np.array(normalized_images)
    adjusted_labels = np.array(adjusted_labels)

    # Shuffle data
    indices = np.arange(len(normalized_images))
    np.random.shuffle(indices)
    normalized_images = normalized_images[indices]
    adjusted_labels = adjusted_labels[indices]

    # Compute dataset sizes
    n_total = len(normalized_images)
    n_train = int(np.round(n_total * train_ratio))
    n_val = int(np.round(n_total * val_ratio))

    # Split data
    train_images, train_labels = normalized_images[:n_train], adjusted_labels[:n_train]
    val_images, val_labels = normalized_images[n_train:n_train + n_val], adjusted_labels[n_train:n_train + n_val]
    test_images, test_labels = normalized_images[n_train + n_val:], adjusted_labels[n_train + n_val:]

    return {
        "train": (train_images, train_labels),
        "val": (val_images, val_labels),
        "test": (test_images, test_labels)
    }

# Split the data
splits = preprocess_and_split(images, labels)
train_data, val_data, test_data = splits['train'], splits['val'], splits['test']

# Correct input data for training
train_images, train_labels = train_data
val_images, val_labels = val_data
test_images, test_labels = test_data

# Ensure the data is in the correct format (NumPy arrays)
train_images = np.array(train_images)
train_labels = np.array(train_labels)
val_images = np.array(val_images)
val_labels = np.array(val_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Summary of split data
print(f"Training: {len(train_data[0])} images")
print(f"Validation: {len(val_data[0])} images")
print(f"Test: {len(test_data[0])} images")

# Compute the number of samples
train_size = train_images.shape[0]
val_size = val_images.shape[0]
test_size = test_images.shape[0]
total_size = train_size + val_size + test_size

# Display the size of training and validation data as a percentage
train_percent = (train_size / total_size) * 100
val_percent = (val_size / total_size) * 100
test_percent = (test_size / total_size) * 100

# Display with a Pie Chart to visualize data split
plt.figure(figsize=(8, 6))
labels = ['Training set', 'Validation set', 'Test set']
sizes = [train_percent, val_percent, test_percent]
colors = ['lightblue', 'lightcoral', 'lightgreen']
explode = (0.08, 0.05, 0.05)  # Highlight "Training set"

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.f%%',
        shadow=True, startangle=90)
plt.title("Split of samples between Training, Validation and Test set", fontsize=12)
plt.show()


### YOLO v8

def convert_to_yolo_format(data, output_dir):
    images, labels = data
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for idx, (img, label) in enumerate(zip(images, labels)):
        img_name = f"image_{idx}.jpg"
        img_path = os.path.join(images_dir, img_name)
        cv2.imwrite(img_path, (img * 255).astype(np.uint8))  # Save the image

        label_path = os.path.join(labels_dir, img_name.replace('.jpg', '.txt'))
        obj_class, x, y, width, height = label
        img_h, img_w = img.shape[:2]

        # Normalize coordinates for YOLO
        x_center = (x + width / 2) / img_w
        y_center = (y + height / 2) / img_h
        norm_width = width / img_w
        norm_height = height / img_h

        with open(label_path, 'w') as f:
            f.write(f"{int(obj_class)} {x_center} {y_center} {norm_width} {norm_height}\n")

def organize_dataset_for_yolo(output_base_dir, splits):
    for subset, (images, labels) in splits.items():
        subset_dir = os.path.join(output_base_dir, subset)
        convert_to_yolo_format((images, labels), subset_dir)

def train_yolov8(data_yaml_path, pre_trained_model="yolov8n.pt", epochs=50):
    model = YOLO(pre_trained_model)
    model.train(data=data_yaml_path, epochs=epochs, imgsz=640)

def evaluate_yolov8(model_path, test_images_dir):
    model = YOLO(model_path)
    results = model.predict(source=test_images_dir, save=True)
    print("Evaluation completed. Results saved.")

# Prepare and organize data
splits = {
    "train": (train_images, train_labels),
    "val": (val_images, val_labels),
    "test": (test_images, test_labels)
}

base_output_dir = r"C:/Users/konai/Cranfield/DL_ComputerVision/UAV_YOLO_Dataset"
splits_dir = os.path.join(base_output_dir, "splits")

organize_dataset_for_yolo(splits_dir, splits)

# Create the YAML file for YOLOv8
data_yaml_path = os.path.join(base_output_dir, "data.yaml")
with open(data_yaml_path, 'w') as f:
    f.write(
        f"path: {base_output_dir}\n"
        f"train: {os.path.join(splits_dir, 'train/images')}\n"
        f"val: {os.path.join(splits_dir, 'val/images')}\n"
        f"test: {os.path.join(splits_dir, 'test/images')}\n"
        "names:\n"
        "  0: UAV\n"
    )

# Train YOLOv8
train_yolov8(data_yaml_path, epochs=10)

# Evaluate YOLOv8
evaluate_yolov8("./runs/detect/train5/weights/best.pt", os.path.join(splits_dir, "test/images"))

def plot_predictions(result_dir, num_images=6, images_per_row=3):
    predicted_images = [
        os.path.join(result_dir, img) for img in os.listdir(result_dir) if img.endswith('.jpg')
    ]
    
    if not predicted_images:
        print("No images found with predictions.")
        return

    # Limit the number of displayed images
    num_images = min(num_images, len(predicted_images))
    images_per_row = min(images_per_row, num_images)

    # Calculate the number of required rows
    num_rows = (num_images + images_per_row - 1) // images_per_row

    # Create a figure with subplots
    plt.figure(figsize=(10, 8))
    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(10, 8))
    axes = axes.flatten()  # Flatten for easy access

    for i, ax in enumerate(axes):
        if i < num_images:
            img_path = predicted_images[i]
            img = cv2.imread(img_path)
            if img is None:
                ax.axis('off')
                continue
            # Convert to RGB for display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.axis('off')
            ax.set_title(f"Image {i + 1}")
        else:
            ax.axis('off')  # Disable axes for unused subplots

    plt.tight_layout()
    plt.show()

# Example usage
result_dir = "./runs/detect/predict2/"  # Folder where YOLOv8 saves prediction images
plot_predictions(result_dir, num_images=9, images_per_row=3)


## SSD (Single Short MultiBox Detection)

# SSD Model with MobileNetV2 Backbone
def create_ssd_model(input_shape=(224, 224, 3)):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_output = base_model.output

    # Feature extraction for bounding box regression
    bbox_branch = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(base_output)
    
    bbox_branch = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(bbox_branch)
    bbox_branch = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(bbox_branch)
    bbox_branch = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(bbox_branch)
    
    bbox_branch = tf.keras.layers.Flatten()(bbox_branch)
    bbox_branch = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(bbox_branch)
    bbox_branch = tf.keras.layers.Dropout(0.2)(bbox_branch)
    bbox_output = tf.keras.layers.Dense(4, activation='linear', name='bbox')(bbox_branch)

    return tf.keras.models.Model(inputs=base_model.input, outputs=bbox_output)

# Compile the SSD model
ssd_model = create_ssd_model(input_shape=(224, 224, 3))
ssd_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss={
        'bbox': tf.keras.losses.Huber(delta=1.0),  # Huber loss for bounding box regression
    },
    metrics={
        'bbox': 'mae',  # Mean Absolute Error
    }
)

# ssd_model.summary()

# IoU Calculation Function
def compute_iou(true_boxes, pred_boxes):

    ious = []
    for pred_box in pred_boxes:  # Loop on each predicted box
        x1 = np.maximum(true_boxes[:, 0], pred_box[0])
        y1 = np.maximum(true_boxes[:, 1], pred_box[1])
        x2 = np.minimum(true_boxes[:, 0] + true_boxes[:, 2], pred_box[0] + pred_box[2])
        y2 = np.minimum(true_boxes[:, 1] + true_boxes[:, 3], pred_box[1] + pred_box[3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        true_area = true_boxes[:, 2] * true_boxes[:, 3]
        pred_area = pred_box[2] * pred_box[3]
        union = true_area + pred_area - intersection

        ious.append(np.mean(intersection / (union + 1e-7)))

    return np.mean(ious)


def apply_nms(boxes, scores, iou_threshold=0.5, max_boxes=80, min_score_threshold=0.5):

    # Convert to float32 to ensure compatibility with TensorFlow
    boxes = tf.cast(boxes, dtype=tf.float32)
    scores = tf.cast(scores, dtype=tf.float32)

    # Filter out boxes with low scores
    mask = scores > min_score_threshold
    boxes, scores = tf.boolean_mask(boxes, mask), tf.boolean_mask(scores, mask)

    # Apply TensorFlow's NMS
    selected_indices = tf.image.non_max_suppression(
        boxes=boxes,
        scores=scores,
        max_output_size=max_boxes,
        iou_threshold=iou_threshold
    )
    return tf.gather(boxes, selected_indices), tf.gather(scores, selected_indices)


# Training the SSD model
history_ssd = ssd_model.fit(
    x=train_images,
    y={
        'bbox': train_labels[:, 1:]  # Bounding boxes
    },
    validation_data=(
        val_images,
        {
            'bbox': val_labels[:, 1:]  # Validation bounding boxes
        }
    ),
    epochs=50,
    batch_size=128
)

# Plot training and validation loss
plt.figure(figsize=(6, 5))
plt.plot(history_ssd.epoch, history_ssd.history["loss"], 'g', label='Training loss')
plt.plot(history_ssd.epoch, history_ssd.history['val_loss'], 'r', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid('on')
plt.title('Reduction of the cost function')
plt.legend()
plt.show()

# Metrics Calculation
def calculate_metrics(model, test_images, test_labels):

    
    pred_bboxes = model.predict(test_images)
    pred_scores = np.random.random(pred_bboxes.shape[0])  # Placeholder for confidence scores
    true_bboxes = test_labels[:, 1:]

    # Apply NMS
    pred_bboxes, pred_scores = apply_nms(pred_bboxes, pred_scores)

    # Calculate IoU
    iou = compute_iou(true_bboxes, pred_bboxes)

    print(f"Metrics:\n - IoU: {iou}\n")
    return iou


# Evaluation
iou = calculate_metrics(ssd_model, test_images, test_labels)


def plot_ssd_predictions_with_cv2(model, images, labels, num_examples=5):
    """
    Visualize SSD model predictions with OpenCV.
    - Green box: ground truth.
    - Red box: prediction with confidence score displayed.
    """

    sample_indices = random.sample(range(len(images)), num_examples)
    plt.figure(figsize=(10, 8))
    fig, axes = plt.subplots((num_examples + 1) // 2, 2, figsize=(10, 8))  # Adjust grid dimensions
    axes = axes.flatten()

    for i, idx in enumerate(sample_indices):
        img = (images[idx] * 255).astype(np.uint8)  # Convert to 8-bit integer
        true_bbox = labels[idx, 1:]  # Ground truth box [x_min, y_min, width, height]
        
        # Get predictions
        pred_bbox = model.predict(np.expand_dims(images[idx], axis=0))
        pred_bbox = pred_bbox[0]  # Extract predicted box [x_min, y_min, width, height]

        # Simulated confidence score (replace with real probability if available)
        pred_score = random.uniform(0.5, 1.0)  # Example confidence between 50% and 100%

        # Convert bounding box coordinates to integers for OpenCV
        true_x, true_y, true_w, true_h = map(int, true_bbox)
        pred_x, pred_y, pred_w, pred_h = map(int, pred_bbox)

        # Draw ground truth box in green
        img = cv2.rectangle(img, (true_x, true_y), (true_x + true_w, true_y + true_h), (0, 255, 0), 2) # (B, G, R)
        cv2.putText(img, "GT", (true_x, true_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw predicted box in red
        img = cv2.rectangle(img, (pred_x, pred_y), (pred_x + pred_w, pred_y + pred_h), (0, 0, 255), 2)
        cv2.putText(img, f"Pred: {pred_score:.2f}", (pred_x, pred_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Add image to subplot
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].axis('off')

    # Remove unused axes
    for j in range(num_examples, len(axes)):
        axes[j].axis('off')

    # Add a legend
    handles = [
        plt.Line2D([0], [0], color='green', lw=2, label='Ground Truth (GT)'),
        plt.Line2D([0], [0], color='red', lw=2, label='Prediction'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
    plt.tight_layout()
    plt.show()

# Visualize a few predictions
plot_ssd_predictions_with_cv2(ssd_model, test_images, test_labels, num_examples=6)
