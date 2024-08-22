import os
import cv2
import numpy as np

def extract_frames(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (64, 64))  # Resize frame to 64x64
        frame = frame / 255.0  # Normalize pixel values
        frames.append(frame)
        count += 1
    cap.release()
    return np.array(frames)

def preprocess_videos(directory_path, output_path, max_videos=450):
    video_files = [f for f in os.listdir(directory_path) if f.endswith('.mp4')]
    video_files = video_files[:max_videos]  # Limit to max_videos if more files are present

    all_frames = []
    all_labels = []

    for video_file in video_files:
        video_path = os.path.join(directory_path, video_file)
        frames = extract_frames(video_path)
        if len(frames) > 0:
            all_frames.append(frames)
            # Extract label from filename or use a placeholder if necessary
            label = video_file.split('_')[2]  # Adjust based on your filename format
            all_labels.append(label)

    # Convert lists to NumPy arrays
    all_frames = np.array(all_frames)
    all_labels = np.array(all_labels)

    # Save preprocessed data
    np.save(os.path.join(output_path, 'preprocessed_frames.npy'), all_frames)
    np.save(os.path.join(output_path, 'labels.npy'), all_labels)

# Specify paths
input_directory = 'D:\Ghanaian Sign Language\Videos\GSL_videos'
output_directory = 'D:\Ghanaian Sign Language\Videos\Video_tensors'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

preprocess_videos(input_directory, output_directory)
