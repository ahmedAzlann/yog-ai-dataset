import cv2
import mediapipe as mp
import numpy as np
import json
import os

# Try to import the lightweight TFLite runtime
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    # If that fails, import the full TensorFlow as a fallback
    print("tflite_runtime not found, falling back to tensorflow.lite")
    import tensorflow.lite as tflite

# --- Constants ---
MODEL_PATH = 'models/yoga_pose_classifier.tflite'
LABELS_PATH = 'models/pose_labels.json'

# --- MediaPipe Initialization ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# --- Load TFLite Model ---
print(f"Loading TFLite model from {MODEL_PATH}...")
try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    exit()

# --- Load Labels ---
print(f"Loading labels from {LABELS_PATH}...")
with open(LABELS_PATH, 'r') as f:
    labels = json.load(f)
print(f"Loaded {len(labels)} labels.")


# --- Normalization Function (CRITICAL: Must be identical to training) ---
def normalize_pose(landmarks):
    """
    Normalizes landmarks by centering at the hips and scaling by torso size.
    """
    if landmarks is None:
        return None

    # Convert LandmarkList to a NumPy array
    landmarks_array = np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks.landmark]
    )

    # 1. Get coordinates for hips and shoulders
    left_hip = landmarks_array[23]
    right_hip = landmarks_array[24]
    left_shoulder = landmarks_array[11]
    right_shoulder = landmarks_array[12]

    # 2. Calculate hip center
    hip_center = (left_hip[:3] + right_hip[:3]) / 2.0

    # 3. Calculate torso size
    shoulder_center = (left_shoulder[:3] + right_shoulder[:3]) / 2.0
    torso_size = np.linalg.norm(shoulder_center - hip_center)

    # 4. Normalize
    if torso_size < 1e-6:
        return None

    # Translate and scale
    translated_landmarks = landmarks_array[:, :3] - hip_center
    scaled_landmarks = translated_landmarks / torso_size

    # Re-attach visibility
    visibility_data = landmarks_array[:, 3].reshape(-1, 1)
    final_data = np.hstack((scaled_landmarks, visibility_data))
    
    # Flatten the (33, 4) array into a (132,) 1D array
    return final_data.flatten()

# --- Main Detection Loop ---
print("Starting webcam feed...")
cap = cv2.VideoCapture(0) # 0 is the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a "mirror" view
    frame = cv2.flip(frame, 1)

    # Convert BGR (OpenCV) to RGB (MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe Pose
    results = pose.process(frame_rgb)
    
    current_pose = "No Pose Detected"

    # If landmarks are detected
    if results.pose_landmarks:
        # Draw the landmarks on the frame
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        # --- TFLite Inference ---
        # 1. Normalize landmarks (using the exact same function)
        normalized_data = normalize_pose(results.pose_landmarks)
        
        if normalized_data is not None:
            # 2. Prepare data for the model
            # Ensure data is float32 and in the correct "batch" shape (1, 132)
            input_data = np.array(normalized_data, dtype=np.float32).reshape(1, -1)
            
            # 3. Set the tensor
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # 4. Run inference
            interpreter.invoke()
            
            # 5. Get the output
            # This will be an array of probabilities, e.g., [0.1, 0.05, 0.8, ...]
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # 6. Find the winning class
            predicted_id = np.argmax(output_data)
            confidence = output_data[0][predicted_id]
            
            # Get the pose name from the ID
            if confidence > 0.5: # Only show if confidence is > 50%
                current_pose = labels[str(predicted_id)]
            else:
                current_pose = "Uncertain"
    
    # Display the predicted pose name on the frame
    cv2.putText(
        frame,
        current_pose,
        (10, 50), # Position
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5, # Font scale
        (0, 255, 0), # Color (green)
        2, # Thickness
        cv2.LINE_AA
    )

    # Display the final frame
    cv2.imshow('Yoga Pose Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
pose.close()

print("Webcam feed stopped.")