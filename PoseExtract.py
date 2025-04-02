import cv2
import mediapipe as mp
import numpy as np
import os
import random
from tqdm import tqdm

class PoseExtract:
    def __init__(self, train_split=0.8, batch_size=32):
        """
        Initialize PoseExtract for sequential processing
        
        Args:
            train_split (float): Proportion of data to use for training
            batch_size (int): Number of frames to process at once
        """
        
        self.train_split = train_split
        self.batch_size = batch_size
        
        self.keypoints = {
            'nose': 0,
            'left_eye_inner': 1,
            'left_eye': 2,
            'left_eye_outer': 3,
            'right_eye_inner': 4,
            'right_eye': 5,
            'right_eye_outer': 6,
            'left_ear': 7,
            'right_ear': 8,
            'mouth_left': 9,
            'mouth_right': 10,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_pinky': 17,
            'right_pinky': 18,
            'left_index': 19,
            'right_index': 20,
            'left_thumb': 21,
            'right_thumb': 22,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
            'left_heel': 29,
            'right_heel': 30,
            'left_foot_index': 31,
            'right_foot_index': 32
        }
        self.pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
        )

    def preprocess_frame(self, frame):
        """Optimize frame preprocessing"""
        # Resize frame for faster processing while maintaining accuracy
        height, width = frame.shape[:2]
        target_height = 480 
        scale = target_height / height
        new_width = int(width * scale)
        frame = cv2.resize(frame, (new_width, target_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        landmarks = []
        while cap.isOpened():
            frames = []
            for _ in range(self.batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(self.preprocess_frame(frame))
                
            if not frames:
                break
                
            for frame in frames:
                res_landmarks = self.pose.process(frame)
                frame_landmarks = []
                    
                if res_landmarks.pose_landmarks:
                    for _, idx in self.keypoints.items():
                        landmark = res_landmarks.pose_landmarks.landmark[idx]
                        frame_landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                else:
                    frame_landmarks = [0, 0, 0, 0] * len(self.keypoints)
                    
                landmarks.append(frame_landmarks)
        
        cap.release()
        return np.array(landmarks)

    def process_class_videos(self, video_path, speed_category, complexity_category, action_class, output_filename):
        """Process a single video"""
        try:
            landmarks = self.process_video(video_path)
            
            metadata = {
                'speed': speed_category,
                'complexity': complexity_category,
                'action_class': action_class
            }
            
            np.savez_compressed(output_filename, landmarks=landmarks, metadata=metadata)
            return True
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            return False

    def extract_from_folder(self, directory_path, output_dir):
        """Extract features sequentially"""
        train_output_dir = os.path.join(output_dir, 'train')
        test_output_dir = os.path.join(output_dir, 'test')
        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(test_output_dir, exist_ok=True)

        # Collect all processing tasks
        processing_tasks = []
        
        for speed_category in os.listdir(directory_path):
            speed_path = os.path.join(directory_path, speed_category)
            if not os.path.isdir(speed_path):
                continue

            for complexity_category in os.listdir(speed_path):
                complexity_path = os.path.join(speed_path, complexity_category)
                if not os.path.isdir(complexity_path):
                    continue

                for action_class in os.listdir(complexity_path):
                    class_path = os.path.join(complexity_path, action_class)
                    if not os.path.isdir(class_path):
                        continue

                    train_class_dir = os.path.join(train_output_dir, action_class)
                    test_class_dir = os.path.join(test_output_dir, action_class)
                    os.makedirs(train_class_dir, exist_ok=True)
                    os.makedirs(test_class_dir, exist_ok=True)

                    video_files = [f for f in os.listdir(class_path) 
                                 if f.endswith(('.mp4', '.avi', '.mov'))]
                    
                    random.shuffle(video_files)
                    split_index = int(len(video_files) * self.train_split)
                    
                    # Prepare tasks for sequential processing
                    for is_train, videos in [(True, video_files[:split_index]), 
                                           (False, video_files[split_index:])]:
                        output_dir = train_class_dir if is_train else test_class_dir
                        
                        for video_file in videos:
                            video_path = os.path.join(class_path, video_file)
                            output_filename = os.path.join(
                                output_dir,
                                f"{os.path.splitext(video_file)[0]}_features.npz"
                            )
                            processing_tasks.append((
                                video_path,
                                speed_category,
                                complexity_category,
                                action_class,
                                output_filename
                            ))
        video = ["MoppingFloor"]
        # Process videos sequentially with progress bar
        for task in tqdm(processing_tasks, desc="Processing videos"):
            video_path, speed_category, complexity_category, action_class, output_filename = task
            if action_class in video:
                self.process_class_videos(video_path, speed_category, complexity_category, action_class, output_filename)

        print("\nDataset preprocessing complete!")

if __name__ == "__main__":
    pose_extract = PoseExtract(
        train_split=0.8,
        batch_size=32
    )
    pose_extract.extract_from_folder(
        '/users/k22001386/action_speed_categories',
        '/users/k22001386/HAR/HAR/processed_features2'
    )