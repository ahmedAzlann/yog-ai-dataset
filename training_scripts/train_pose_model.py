import pandas as pd
import json
import os
import numpy as np

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Constants ---
LANDMARK_CSV_FILE = 'dataset/landmarks_for_training.csv'
MASTER_LIST_CSV = 'dataset/poses_master_list.csv'

# We save in the .keras format, not .pkl
MODEL_OUTPUT_PATH = 'models/yoga_pose_classifier.keras' 
LABELS_OUTPUT_PATH = 'models/pose_labels.json'


def train_model():
    """
    Loads landmark data, trains a Keras neural network, evaluates it,
    and saves the model and label mapping for TFLite conversion.
    """
    print("Starting model training process (TensorFlow/Keras)...")
    
    # Ensure the models directory exists
    os.makedirs('models', exist_ok=True)

    # --- 1. Load the Landmark Data ---
    try:
        df = pd.read_csv(LANDMARK_CSV_FILE)
    except FileNotFoundError:
        print(f"ERROR: Could not find {LANDMARK_CSV_FILE}")
        print("Please run the extract_landmarks.py script first.")
        return
        
    df = df.dropna()
    if df.empty:
        print("ERROR: The landmark CSV is empty or all rows had errors.")
        return

    print(f"Loaded {len(df)} total frames from {len(df['pose_id'].unique())} poses.")

    # --- 2. Create and Save Label Mapping ---
    try:
        labels_df = pd.read_csv(MASTER_LIST_CSV)
    except FileNotFoundError:
        print(f"ERROR: {MASTER_LIST_CSV} not found. Cannot create label map.")
        return
        
    pose_map = pd.Series(labels_df.pose_name.values, index=labels_df.pose_id).to_dict()
    pose_map_json_keys = {str(k): v for k, v in pose_map.items()}
    num_classes = len(pose_map) # Get the number of poses (e.g., 91)

    with open(LABELS_OUTPUT_PATH, 'w') as f:
        json.dump(pose_map_json_keys, f, indent=4)
    print(f"Saved label mapping for {num_classes} classes to {LABELS_OUTPUT_PATH}")

    # --- 3. Separate Features (X) and Labels (y) ---
    y_int = df['pose_id'] # Our labels are integers (0, 1, 2...)
    X = df.drop(columns=['pose_id', 'pose_name'])
    
    # --- 4. PREPARE DATA FOR NEURAL NETWORK ---
    # Neural networks need labels to be "one-hot encoded"
    # e.g., '2' becomes [0, 0, 1, 0, ...]
    y_hot = to_categorical(y_int, num_classes=num_classes)
    
    print(f"Data shape: X features={X.shape}, y labels={y_hot.shape}")

    # --- 5. Split Data into Training and Testing Sets ---
    # We stratify using the *integer* labels (y_int) to ensure
    # a balanced split, but we split the one-hot labels (y_hot).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_hot, test_size=0.2, random_state=42, stratify=y_int
    )
    print(f"Training set size: {len(X_train)} frames")
    print(f"Test set size: {len(X_test)} frames")

    # --- 6. Define the Keras Model ---
    model = Sequential([
        # Input layer: Must match the number of features (e.g., 132)
        Input(shape=(X_train.shape[1],)),
        
        # Hidden layer 1
        Dense(128, activation='relu'),
        Dropout(0.3), # Dropout prevents overfitting
        
        # Hidden layer 2
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        # Output layer: Must have 'num_classes' neurons
        # 'softmax' gives a probability for each pose
        Dense(num_classes, activation='softmax')
    ])
    
    # --- 7. Compile the Model ---
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy', # Use this for one-hot labels
        metrics=['accuracy']
    )
    
    model.summary() # Print a summary of the model architecture

    # --- 8. Train the Model ---
    print("\nTraining neural network...")
    
    # Stop training early if validation loss doesn't improve
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, # Wait 10 epochs
        restore_best_weights=True # Keep the best version of the model
    )
    
    history = model.fit(
        X_train,
        y_train,
        epochs=100, # Max epochs (will likely stop early)
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    print("Model training complete.")

    # --- 9. Evaluate the Model ---
    print("\nEvaluating model on test data...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"--- Model Accuracy: {accuracy * 100:.2f}% ---")
    
    # Create a full classification report
    # 1. Get probability predictions from the model
    y_pred_probs = model.predict(X_test)
    # 2. Convert probabilities to a single class ID (e.g., [0.1, 0.9] -> 1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    # 3. Convert the one-hot test labels back to class IDs
    y_test_int = np.argmax(y_test, axis=1)
    
    target_ids = sorted(y_int.unique())
    target_names = [pose_map[id] for id in target_ids]
    
    try:
        print("\nClassification Report:")
        report = classification_report(y_test_int, y_pred, labels=target_ids, target_names=target_names)
        print(report)
    except Exception as e:
        print(f"Could not generate full classification report: {e}")

    # --- 10. Save the Trained Model ---
    # Save in the native .keras format
    model.save(MODEL_OUTPUT_PATH)
    print(f"\n--- Successfully saved trained model to {MODEL_OUTPUT_PATH} ---")
    print("This file is ready for conversion to TFLite.")


if __name__ == "__main__":
    train_model()