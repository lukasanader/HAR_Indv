import cv2
import mediapipe as mp
import numpy as np
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import time

class PoseMotionAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        # Define the limb groups
        self.limb_groups = { 
            'left_arm': [(self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW),
                        (self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST)],
            'right_arm': [(self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW),
                         (self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST)],
            'left_leg': [(self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
                        (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE)],
            'right_leg': [(self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE),
                         (self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE)]
        }
        

    def calculate_group_motion(self, landmarks1, landmarks2, group):
        speeds = []
        for start_limb, end_limb in self.limb_groups[group]:
            # Check if the landmarks are present and the visibility is above 0.5
            if (landmarks1.pose_landmarks and landmarks2.pose_landmarks and
                landmarks1.pose_landmarks.landmark[start_limb].visibility > 0.5 and
                landmarks1.pose_landmarks.landmark[end_limb].visibility > 0.5 and
                landmarks2.pose_landmarks.landmark[start_limb].visibility > 0.5 and
                landmarks2.pose_landmarks.landmark[end_limb].visibility > 0.5):
                # Calculate the displacement of the start and end limb
                start_displacement = np.sqrt(
                    (landmarks2.pose_landmarks.landmark[start_limb].x - landmarks1.pose_landmarks.landmark[start_limb].x)**2 +
                    (landmarks2.pose_landmarks.landmark[start_limb].y - landmarks1.pose_landmarks.landmark[start_limb].y)**2
                )
                
                end_displacement = np.sqrt(
                    (landmarks2.pose_landmarks.landmark[end_limb].x - landmarks1.pose_landmarks.landmark[end_limb].x)**2 +
                    (landmarks2.pose_landmarks.landmark[end_limb].y - landmarks1.pose_landmarks.landmark[end_limb].y)**2
                )
                if start_displacement == 0:
                    start_displacement = 0.0001
                if end_displacement == 0:
                    end_displacement = 0.0001
                # Calculate the average displacement
                speeds.append((start_displacement + end_displacement) / 2)
        # Return the average speed of the group
        return np.mean(speeds) if speeds else 0

    def process_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        # Dictionary to store the motion of each group
        group_motions = {
            'left_arm': [],
            'right_arm': [],
            'left_leg': [],
            'right_leg': []
        }
        
        ret, prev_frame = cap.read()
        prev_landmarks = self.pose.process(prev_frame)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = self.pose.process(frame)
            # Calculate the motion of each group
            if landmarks.pose_landmarks and prev_landmarks.pose_landmarks:
                for group in self.limb_groups.keys():
                    motion = self.calculate_group_motion(prev_landmarks, landmarks, group)
                    group_motions[group].append(motion)
            
            prev_landmarks = landmarks
        
        cap.release()
        
        # Calculate the avergae motion in each limb group
        avg_motions = []
        for group, motions in group_motions.items():
            if motions:
                avg_motion = np.mean(motions)
                if np.isnan(avg_motion):
                    avg_motion = 0
                avg_motions.append(avg_motion)

        
        # Count the number of active groups
        active_groups = 0
        for motion in avg_motions:
            if motion > 0.1:
                active_groups += 1
        
        total_correlation = 0
        correlation_count = 0
        # Calculate the correlation between the limbs of each group
        if len(group_motions['left_arm']) > 1 and len(group_motions['right_arm']) > 1:
            if np.std(group_motions['left_arm']) > 0 and np.std(group_motions['right_arm']) > 0:
                corr = np.corrcoef(group_motions['left_arm'], group_motions['right_arm'])[0,1]
                if not np.isnan(corr):
                    total_correlation += abs(corr)
                    correlation_count += 1
        
        if len(group_motions['left_leg']) > 1 and len(group_motions['right_leg']) > 1:
            if np.std(group_motions['left_leg']) > 0 and np.std(group_motions['right_leg']) > 0:
                corr = np.corrcoef(group_motions['left_leg'], group_motions['right_leg'])[0,1]
                if not np.isnan(corr):
                    total_correlation += abs(corr)
                    correlation_count += 1

        # Calculate the average correlation
        avg_correlation = 0
        if correlation_count > 0:
            avg_correlation = total_correlation / correlation_count
        
        # Calculate the complexity score
        return (active_groups * 0.6 + (1 - avg_correlation) * 0.4)

    def analyze_action_folder(self, action_dir):
        all_complexity_scores = []
        

        video_files = list(action_dir.glob("*.avi"))
        if not video_files:
            return None
        # Process each video in the action directory
        for video_file in tqdm(video_files):
            complexity_score = self.process_video(video_file)
            if complexity_score is not None:
                all_complexity_scores.append(complexity_score)
        # Return the average complexity score
        return np.mean(all_complexity_scores)

def organize_actions(source_dir="~/Documents/action_speed_categories2"):
    analyzer = PoseMotionAnalyzer()
    source_dir = Path(source_dir).expanduser()
    
    for speed_dir in source_dir.iterdir():
        if not speed_dir.is_dir():
            continue
            
        action_complexities = []
        
        for action_dir in tqdm(speed_dir.iterdir(),desc="Processing actions"):
            if not action_dir.is_dir():
                continue
                
            complexity_score = analyzer.analyze_action_folder(action_dir)
            
            if complexity_score is not None:
                action_complexities.append((action_dir, complexity_score))
        # Sort the actions by complexity score and split them into simple and complex categories
        action_complexities.sort(key=lambda x: x[1])
        median = len(action_complexities) // 2
        
        for index, (action_dir, score) in enumerate(action_complexities):
            category=""
            if index < median:
                category = "simple"
            else:
                category = "complex"
            category_dir = speed_dir / category
            category_dir.mkdir(exist_ok=True)
            # move action_dir to category_dir
            shutil.move(action_dir, category_dir)
            print(f"Categorized {action_dir.name} as {category}")

if __name__ == "__main__":
    organize_actions()