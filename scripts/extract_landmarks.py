import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# --- Constants ---
MASTER_LIST_CSV = 'dataset/poses_master_list.csv'
OUTPUT_CSV = 'dataset/landmarks_for_training.csv'
VIDEO_DIR = 'videos/'

# --- MediaPipe Initialization ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, # We are processing video
    model_complexity=1,      # 0, 1, or 2. Higher = more accurate but slower
    enable_segmentation=False,
    min_detection_confidence=0.5
)

def normalize_pose(landmarks):
    """
    Normalizes landmarks by centering at the hips and scaling by torso size.
    This makes the pose data invariant to position and scale.
    """
    if landmarks is None:
        return None

    # Convert LandmarkList to a NumPy array (33 landmarks, 4 values: x, y, z, visibility)
    landmarks_array = np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks.landmark]
    )

    # 1. Get coordinates for hips and shoulders (indices from MediaPipe)
    left_hip = landmarks_array[23]
    right_hip = landmarks_array[24]
    left_shoulder = landmarks_array[11]
    right_shoulder = landmarks_array[12]

    # 2. Calculate hip center (our new origin: 0,0,0)
    # We use x, y, z for this (first 3 columns)
    hip_center = (left_hip[:3] + right_hip[:3]) / 2.0

    # 3. Calculate torso size (for scaling)
    # This is the distance between hip center and shoulder center
    shoulder_center = (left_shoulder[:3] + right_shoulder[:3]) / 2.0
    
    # Euclidean distance
    torso_size = np.linalg.norm(shoulder_center - hip_center)

    # 4. Normalize
    if torso_size < 1e-6:
        # Avoid division by zero if torso size is extremely small
        return None

    # Translate: Move all landmarks so hip center is at (0,0,0)
    translated_landmarks = landmarks_array[:, :3] - hip_center
    
    # Scale: Divide by torso size
    scaled_landmarks = translated_landmarks / torso_size

    # Re-attach visibility to the (x, y, z) data
    # visibility is the 4th column (index 3)
    visibility_data = landmarks_array[:, 3].reshape(-1, 1)
    
    # final_data shape will be (33, 4)
    final_data = np.hstack((scaled_landmarks, visibility_data))
    
    # Flatten the (33, 4) array into a (132,) 1D array
    return final_data.flatten()


def extract_landmarks():
    """
    Main function to loop through videos, process frames, and save data.
    """
    print("Starting landmark extraction process...")

    # 1. Read the master list
    try:
        master_df = pd.read_csv(MASTER_LIST_CSV)
    except FileNotFoundError:
        print(f"ERROR: {MASTER_LIST_CSV} not found.")
        print("Please make sure you have created this file.")
        return

    all_pose_data = []

    # 2. Loop through each video in the master list
    print(f"Found {len(master_df)} videos to process.")
    for index, row in tqdm(master_df.iterrows(), total=master_df.shape[0], desc="Processing Videos"):
        
        pose_id = row['pose_id']
        pose_name = row['pose_name']
        video_filename = row['video_filename']
        video_path = os.path.join(VIDEO_DIR, video_filename)

        if not os.path.exists(video_path):
            print(f"Warning: Video file not found, skipping: {video_path}")
            continue

        # 3. Open video file with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video file, skipping: {video_path}")
            continue

        # 4. Read video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # 5. Pass frame to MediaPipe
            # Convert BGR (OpenCV) to RGB (MediaPipe)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # 6. If landmarks are detected:
            if results.pose_landmarks:
                # 7. Normalize the landmarks
                normalized_data = normalize_pose(results.pose_landmarks)
                
                if normalized_data is not None:
                    # 8. Create a data row and append
                    # [pose_id, pose_name, lm_0_x, lm_0_y, lm_0_z, lm_0_vis, lm_1_x, ...]
                    data_row = [pose_id, pose_name] + list(normalized_data)
                    all_pose_data.append(data_row)

        cap.release()

    pose.close()
    print("\nVideo processing complete.")

    # 9. Convert big list to a DataFrame
    if not all_pose_data:
        print("ERROR: No pose data was extracted. Check your videos and MediaPipe settings.")
        return

    # Create dynamic column names
    # ['pose_id', 'pose_name', 'lm_0_x', 'lm_0_y', 'lm_0_z', 'lm_0_vis', ...]
    columns = ['pose_id', 'pose_name']
    for i in range(33): # 33 landmarks
        columns.extend([
            f'lm_{i}_x', f'lm_{i}_y', f'lm_{i}_z', f'lm_{i}_vis'
        ])

    final_df = pd.DataFrame(all_pose_data, columns=columns)
    
    # 10. Save the final CSV
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    print("---")
    print(f"Successfully extracted {len(final_df)} frames.")
    print(f"Landmark data saved to: {OUTPUT_CSV}")
    print("---")


if __name__ == "__main__":
    extract_landmarks()