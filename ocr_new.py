import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import cv2
import os

# TensorFlow bellek ayarları
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Bellek sınırını ayarla
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(f"GPU memory config error: {e}")

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Define character mapping
def get_character_mapping():
    """Create a mapping from class indices to actual characters"""
    # The mapping for EMNIST 'byclass' is: 
    # 0-9: digits 0-9
    # 10-35: uppercase letters A-Z
    # 36-61: lowercase letters a-z
    chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    return {i: c for i, c in enumerate(chars)}

# Define model architecture
def build_model(input_shape=(28, 28, 1), num_classes=62):
    """Build the CNN model"""
    model = Sequential()
    
    # First convolutional block
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    # Second convolutional block
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    # Third convolutional block
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    # Flatten and dense layers
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

# Image preprocessing for prediction
def preprocess_image(img):
    """Preprocess an image for prediction"""
    # Resize to 28x28 if needed
    if img.shape != (28, 28):
        img = cv2.resize(img, (28, 28))
    
    # Normalize
    img = img.astype('float32') / 255.0
    
    # Add batch and channel dimensions
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    
    return img

def predict_character(img, model, char_mapping):
    """Predict a character from an image"""
    # Preprocess the image
    processed_img = preprocess_image(img)
    
    # Make prediction
    prediction = model.predict(processed_img)[0]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]
    
    # Get character from mapping
    predicted_char = char_mapping[predicted_class]
    
    return predicted_char, confidence, prediction

def main():
    # Get character mapping
    char_mapping = get_character_mapping()
    print("Character mapping created.")
    
    # Method 1: Try to load data using emnist package
    data_loaded = False
    try:
        from emnist import extract_training_samples, extract_test_samples
        
        # Load training and test data
        x_train, y_train = extract_training_samples('byclass')
        x_test, y_test = extract_test_samples('byclass')
        
        print("EMNIST dataset loaded successfully using emnist package.")
        print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
        print(f"Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}")
        
        data_loaded = True
    except Exception as e:
        print(f"Error loading EMNIST package: {e}")
        print("Trying alternative method...")
    
    # Method 2: If Method 1 fails, try loading from CSV files
    if not data_loaded:
        try:
            print("Attempting to load CSV files...")
            # Load CSV files (assuming they're in your working directory)
            train_data = pd.read_csv('emnist-byclass-train.csv', dtype= 'uint8')
            print(f"Loaded training CSV with shape: {train_data.shape}")
            
            test_data = pd.read_csv('emnist-byclass-test.csv', dtype= 'uint8')
            print(f"Loaded test CSV with shape: {test_data.shape}")
            
            # Separate labels and images
            y_train = train_data.iloc[:, 0].values
            x_train = train_data.iloc[:, 1:].values
            y_test = test_data.iloc[:, 0].values
            x_test = test_data.iloc[:, 1:].values
            
            print(f"Extracted feature shapes - X_train: {x_train.shape}, y_train: {y_train.shape}")
            
            # Reshape images to 28x28
            x_train = x_train.reshape(-1, 28, 28)
            x_test = x_test.reshape(-1, 28, 28)
            
            print("EMNIST dataset loaded from CSV files.")
            print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
            print(f"Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}")
            
            data_loaded = True
        except Exception as e:
            print(f"Error loading EMNIST from CSV: {e}")
            import traceback
            traceback.print_exc()
            print("Please download the EMNIST dataset from https://www.nist.gov/itl/products-and-services/emnist-dataset")
    
    if not data_loaded:
        print("Failed to load data. Exiting.")
        return
    
    # Preprocess the data
    print("Preprocessing data...")
    
    # Normalize the data (scale to [0, 1])
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Add channel dimension for CNN
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    # Check the number of unique classes
    num_classes = len(np.unique(y_train))
    print(f"Number of classes: {num_classes}")
    
    # Convert labels to one-hot encoding
    y_train_one_hot = to_categorical(y_train, num_classes)
    y_test_one_hot = to_categorical(y_test, num_classes)
    
    # Create data augmentation generator
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False
    )
    
    # Compute class weights for imbalanced dataset
    print("Computing class weights...")
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}
    
    # Build the model
    print("Building the model...")
    model = build_model(input_shape=(28, 28, 1), num_classes=num_classes)
    model.summary()
    
    # Set up model checkpoint to save the best model
    checkpoint_filepath = 'emnist_ocr_model_best.h5'
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    
    # Training parameters
    epochs = 5  # 20'den 5'e düşürüldü
    batch_size = 64  # 256'dan 64'e düşürün
    
    # Adım takibi için callback ekle
    class EpochLogger(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            print(f"\nStarting Epoch {epoch+1}/{epochs}")
        
        def on_epoch_end(self, epoch, logs=None):
            print(f"\nCompleted Epoch {epoch+1}/{epochs}")
    
    epoch_logger = EpochLogger()
    
    # Train the model
    print("Training the model...")
    try:
        # Örneğin, veri setinin %20'sini kullanın:
        x_train = x_train[:int(len(x_train)*0.2)]
        y_train = y_train[:int(len(y_train)*0.2)]
        y_train_one_hot = y_train_one_hot[:int(len(y_train_one_hot)*0.2)]
        
        history = model.fit(
            datagen.flow(x_train, y_train_one_hot, batch_size=batch_size),
            steps_per_epoch=len(x_train) // batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test_one_hot),
            class_weight=class_weights_dict,
            callbacks=[model_checkpoint, epoch_logger],
            verbose=1
        )
        print("Training completed successfully for all epochs.")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    # Save the final model
    model.save('emnist_ocr_model_final.h5')
    print("Final model saved as 'emnist_ocr_model_final.h5'")
    print("Best model saved as 'emnist_ocr_model_best.h5'")
    
    # Load the best model
    try:
        best_model = tf.keras.models.load_model(checkpoint_filepath)
        print("Loaded the best model from checkpoint.")
    except:
        best_model = model
        print("Using the current model (no checkpoint found).")
    
    # Evaluate on test data
    test_loss, test_accuracy = best_model.evaluate(x_test, y_test_one_hot, verbose=1)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Generate predictions
    y_pred = best_model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Generate classification report
    target_names = [char_mapping[i] for i in range(num_classes)]
    print(classification_report(y_test, y_pred_classes, target_names=target_names))
    
    print("Training and evaluation completed successfully.")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Bu satırı kodun başına ekleyin - GPU'yu devre dışı bırakır
    main() 