import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dropout, Flatten, Dense, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Load data and labels from .npy files
data_path = 'D:\Ghanaian Sign Language\Videos\Video_tensors'
data_file = 'preprocessed_frames.npy'
labels_file = 'labels.npy'

# Load the data and labels
data = np.load(os.path.join(data_path, data_file))
labels = np.load(os.path.join(data_path, labels_file))

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# Normalize the data
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0

print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)
print("Training labels shape:", y_train.shape)
print("Validation labels shape:", y_val.shape)

# Combine training and validation labels for consistent encoding
all_labels = np.concatenate([y_train, y_val])

# Initialize and fit label encoder on all labels
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# Transform training and validation labels
y_train_encoded = label_encoder.transform(y_train)
y_val_encoded = label_encoder.transform(y_val)

# Convert integer labels to one-hot encoded labels
num_classes = len(label_encoder.classes_)
y_train_one_hot = to_categorical(y_train_encoded, num_classes=num_classes)
y_val_one_hot = to_categorical(y_val_encoded, num_classes=num_classes)

# Check shapes and data types
print("y_train_one_hot shape:", y_train_one_hot.shape)
print("y_val_one_hot shape:", y_val_one_hot.shape)
print("y_train_one_hot dtype:", y_train_one_hot.dtype)
print("y_val_one_hot dtype:", y_val_one_hot.dtype)

# Define the enhanced 3D CNN model
def build_enhanced_cnn_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling3D((2, 2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((2, 2, 2))(x)
    x = BatchNormalization()(x)
    
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((2, 2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((2, 2, 2))(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# Build the model
input_shape = X_train.shape[1:]  # Shape of the input videos (e.g., 30x64x64x3)
model = build_enhanced_cnn_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train the model
batch_size = 8
history = model.fit(
    X_train, y_train_one_hot,
    validation_data=(X_val, y_val_one_hot),
    epochs=50,
    batch_size=batch_size,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val_one_hot)
print(f"Validation loss: {val_loss:.4f}")
print(f"Validation accuracy: {val_accuracy:.4f}")

# Save the model in the new format
model.save('sign_language_model.keras')
