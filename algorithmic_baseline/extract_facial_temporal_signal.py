import os 
import cv2 
import numpy as np 
import mediapipe as mp 

DATA_DIR = 'path/to/chunked_dataset/'
folders = os.listdir(DATA_DIR)
folders.sort()

# initialize mediapipe 
mp_face_mesh = mp.solutions.face_mesh 

facial_temporal_signal_dict = {}
for folder in folders: 
    print("Processing folder:", folder)
    folder_path = os.path.join(DATA_DIR, folder)
    files = os.listdir(folder_path)
    files = [f for f in files if f.startswith('rgb_left')]
    N = len(files)
    facial_temporal_signal = []
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
                if results.multi_face_landmarks: 
                    for face_landmarks in results.multi_face_landmarks: 
                        lc = face_landmarks.landmark
                        lc_x = [lmk.x for lmk in lc]
                        lc_y = [lmk.y for lmk in lc]
                        Xmin = int(np.floor(min(lc_x) * frame_rgb.shape[1]))
                        Xmax = int(np.ceil(max(lc_x) * frame_rgb.shape[1]))
                        Ymin = int(np.floor(min(lc_y) * frame_rgb.shape[0]))
                        Ymax = int(np.ceil(max(lc_y) * frame_rgb.shape[0]))
                        if Xmin < 0: Xmin = 0
                        if Xmax > frame_rgb.shape[1]: Xmax = frame_rgb.shape[1]
                        if Ymin < 0: Ymin = 0
                        if Ymax > frame_rgb.shape[0]: Ymax = frame_rgb.shape[0]
                        # Crop the frame
                        cropped_frame = frame_rgb[Ymin:Ymax, Xmin:Xmax, :]
                        if cropped_frame is None:
                            print(f"Warning: Cropped frame is None for folder {folder}, file {file_name}, frame {frame_idx}.")
                            continue
                        cropped_frame = cropped_frame.astype(np.float32)/255.
                        # Average the cropped frame and store
                        facial_temporal_signal.append(np.mean(cropped_frame, axis=(0, 1)))
                else:
                    facial_temporal_signal.append(np.array([np.nan, np.nan, np.nan]))  # Append NaN if no landmarks detected
    # Store the signal for the folder
    facial_temporal_signal_dict[folder] = np.array(facial_temporal_signal) 

import pickle
# Save the dictionary to a file
with open('facial_temporal_signal_dict.pkl', 'wb') as f:
    pickle.dump(facial_temporal_signal_dict, f)
print("Facial temporal signal extraction completed and saved to 'facial_temporal_signal_dict.pkl'.")
