import os 
import cv2 
import numpy as np 
import mediapipe as mp 

DATA_DIR = 'chunked_dataset/'
folders = os.listdir(DATA_DIR)
folders.sort()

# initialize mediapipe
mp_face_mesh = mp.solutions.face_mesh

# Based on standard MediaPipe Face Mesh landmark map:
# Left Eye:
LEFT_EYE_TOP_LID_INDEX = 159
LEFT_EYE_BOTTOM_LID_INDEX = 145
LEFT_EYE_HORZ_INDEX0 = 33 
LEFT_EYE_HORZ_INDEX1 = 133 
# Right Eye:
RIGHT_EYE_TOP_LID_INDEX = 386
RIGHT_EYE_BOTTOM_LID_INDEX = 374
RIGHT_EYE_HORZ_INDEX0 = 362 
RIGHT_EYE_HORZ_INDEX1 = 263 
EYELID_MARKER_INDICES = [LEFT_EYE_TOP_LID_INDEX, LEFT_EYE_BOTTOM_LID_INDEX,
                        RIGHT_EYE_TOP_LID_INDEX, RIGHT_EYE_BOTTOM_LID_INDEX,
                        LEFT_EYE_HORZ_INDEX0, LEFT_EYE_HORZ_INDEX1,
                        RIGHT_EYE_HORZ_INDEX0, RIGHT_EYE_HORZ_INDEX1]

eos_dict = {}
for folder in folders:
    print("Processing folder:", folder)
    folder_path = os.path.join(DATA_DIR, folder)
    files = os.listdir(folder_path)
    files = [f for f in files if f.startswith('rgb_left')]
    N = len(files)
    left_eye_distances = []
    right_eye_distances = []
    with mp_face_mesh.FaceMesh(max_num_faces=1,
                                    refine_landmarks=True, # Essential for iris landmarks
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5) as face_mesh:
        for i in range(N):
            file_name = f'rgb_left_{i}.npy'
            file_path = os.path.join(folder_path, file_name)
            x_video_frames = np.load(file_path) 
            for frame_idx, frame_rgb in enumerate(x_video_frames): 
                # Ensure the frame is uint8 and C-contiguous
                if frame_rgb.dtype != np.uint8:
                    if frame_rgb.max() <= 1.0: frame_rgb = (frame_rgb * 255).astype(np.uint8)
                    else: frame_rgb = frame_rgb.astype(np.uint8)
                frame_rgb = np.ascontiguousarray(frame_rgb)
                # Process the frame
                results = face_mesh.process(frame_rgb)
                current_left_distance = np.nan
                current_right_distance = np.nan
                if results.multi_face_landmarks: 
                    for face_landmarks in results.multi_face_landmarks: 
                        # --- Left Eye Distance ---
                        if len(face_landmarks.landmark) > LEFT_EYE_TOP_LID_INDEX and \
                        len(face_landmarks.landmark) > LEFT_EYE_BOTTOM_LID_INDEX:
                            top_lid_left = face_landmarks.landmark[LEFT_EYE_TOP_LID_INDEX]
                            bottom_lid_left = face_landmarks.landmark[LEFT_EYE_BOTTOM_LID_INDEX]
                            eyelid_horz_left0 = face_landmarks.landmark[LEFT_EYE_HORZ_INDEX0]
                            eyelid_horz_left1 = face_landmarks.landmark[LEFT_EYE_HORZ_INDEX1]
                            # Calculate normalized vertical distance
                            # Y coordinates increase downwards
                            current_left_distance = abs(top_lid_left.y - bottom_lid_left.y)
                            # Normalize by the distance between the horizontal landmarks
                            if abs(eyelid_horz_left0.x - eyelid_horz_left1.x) > 0:
                                current_left_distance /= abs(eyelid_horz_left0.x - eyelid_horz_left1.x)
                            else:
                                current_left_distance = 0
                        
                        # --- Right Eye Distance ---
                        if len(face_landmarks.landmark) > RIGHT_EYE_TOP_LID_INDEX and \
                        len(face_landmarks.landmark) > RIGHT_EYE_BOTTOM_LID_INDEX:
                            top_lid_right = face_landmarks.landmark[RIGHT_EYE_TOP_LID_INDEX]
                            bottom_lid_right = face_landmarks.landmark[RIGHT_EYE_BOTTOM_LID_INDEX]
                            eyelid_horz_right0 = face_landmarks.landmark[RIGHT_EYE_HORZ_INDEX0]
                            eyelid_horz_right1 = face_landmarks.landmark[RIGHT_EYE_HORZ_INDEX1]
                            # Calculate normalized vertical distance
                            current_right_distance = abs(top_lid_right.y - bottom_lid_right.y)
                            # Normalize by the distance between the horizontal landmarks
                            if abs(eyelid_horz_right0.x - eyelid_horz_right1.x) > 0:
                                current_right_distance /= abs(eyelid_horz_right0.x - eyelid_horz_right1.x)
                            else:
                                current_right_distance = 0
                left_eye_distances.append(current_left_distance)
                right_eye_distances.append(current_right_distance)
    left_eye_distances = np.array(left_eye_distances)
    right_eye_distances = np.array(right_eye_distances)
    avg_eye_distances = (left_eye_distances + right_eye_distances) / 2
    eos_dict[folder] = {
        'eo_signal_left': left_eye_distances,
        'eo_signal_right': right_eye_distances,
        'eo_signal_avg': avg_eye_distances
    }
    # print(f"Processed {N} frames for folder {folder}, signal length: {len(avg_eye_distances)}")

# save dictionary as a .pkl file
import pickle
with open('eos_norm_dict.pkl', 'wb') as f:
    pickle.dump(eos_dict, f)
print("Eye openness signals extracted and saved to eos_norm_dict.pkl")