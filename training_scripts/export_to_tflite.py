import tensorflow as tf
import os

# --- Constants ---
KERAS_MODEL_PATH = 'models/yoga_pose_classifier.keras'
TFLITE_MODEL_PATH = 'models/yoga_pose_classifier.tflite'

def export_model():
    """
    Loads the trained .keras model and converts it into a 
    TensorFlow Lite (.tflite) model.
    """
    print("Starting TFLite export process...")
    
    # --- 1. Load the Keras Model ---
    if not os.path.exists(KERAS_MODEL_PATH):
        print(f"ERROR: Model file not found at {KERAS_MODEL_PATH}")
        print("Please run the train_pose_model.py script first.")
        return
        
    try:
        print(f"Loading model from {KERAS_MODEL_PATH}...")
        model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    print("Model loaded successfully.")

    # --- THIS IS THE MODIFIED SECTION ---
    # We use 'from_concrete_functions' for more stability.
    # This avoids the Keras-specific conversion bugs.
    print("Initializing converter using 'concrete_function' method...")
    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
    )

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    # --- END OF MODIFIED SECTION ---

    # --- 3. Set Converter Options ---
    # This enables a set of optimizations that TFLite can perform.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # --- 4. Convert the Model ---
    print("Converting model to TensorFlow Lite...")
    try:
        tflite_model = converter.convert()
        print("Model conversion successful.")
    except Exception as e:
        print(f"An error occurred during conversion: {e}")
        return

    # --- 5. Save the TFLite Model ---
    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
        
    print(f"\n--- Successfully saved TFLite model to {TFLITE_MODEL_PATH} ---")
    print(f"File size: {os.path.getsize(TFLITE_MODEL_PATH) / 1024:.2f} KB")

if __name__ == "__main__":
    export_model()