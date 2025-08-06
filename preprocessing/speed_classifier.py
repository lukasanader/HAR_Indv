import cv2
import mediapipe as mp
import numpy as np
import os
import shutil
from pathlib import Path

class ActionSpeedAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        # Define the limb groups
        self.limbs = [
            (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_ELBOW),
            (mp.solutions.pose.PoseLandmark.LEFT_ELBOW, mp.solutions.pose.PoseLandmark.LEFT_WRIST),
            (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW),
            (mp.solutions.pose.PoseLandmark.RIGHT_ELBOW, mp.solutions.pose.PoseLandmark.RIGHT_WRIST),
            (mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.LEFT_KNEE),
            (mp.solutions.pose.PoseLandmark.LEFT_KNEE, mp.solutions.pose.PoseLandmark.LEFT_ANKLE),
            (mp.solutions.pose.PoseLandmark.RIGHT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_KNEE),
            (mp.solutions.pose.PoseLandmark.RIGHT_KNEE, mp.solutions.pose.PoseLandmark.RIGHT_ANKLE),
        ]
        self.action_speeds = {}

    def calculate_limb_size(self, landmarks1, landmarks2):
        size = []
        for start_limb, end_limb in self.limbs:
            # Check if the landmarks are present and the visibility is above 0.5
            if (landmarks1 and landmarks2 and 
                landmarks1.pose_landmarks.landmark[start_limb].visibility > 0.5 and
                landmarks1.pose_landmarks.landmark[end_limb].visibility > 0.5 and
                landmarks2.pose_landmarks.landmark[start_limb].visibility > 0.5 and
                landmarks2.pose_landmarks.landmark[end_limb].visibility > 0.5):
                # Calculate the size of the limb
                size_of_limb = np.sqrt((landmarks1.pose_landmarks.landmark[start_limb].x - landmarks1.pose_landmarks.landmark[end_limb].x)**2 + 
                      (landmarks1.pose_landmarks.landmark[start_limb].y - landmarks1.pose_landmarks.landmark[end_limb].y)**2)
                size.append(size_of_limb)
        # Return the average size of the limbs
        return np.mean(size) if size else 0

    def calculate_limb_speeds(self, landmarks1, landmarks2):

        speed = []
        for start_limb, end_limb in self.limbs:
            if (landmarks1 and landmarks2 and # Check if the landmarks are present and the visibility is above 0.5
                landmarks1.pose_landmarks.landmark[start_limb].visibility > 0.5 and
                landmarks1.pose_landmarks.landmark[end_limb].visibility > 0.5 and
                landmarks2.pose_landmarks.landmark[start_limb].visibility > 0.5 and
                landmarks2.pose_landmarks.landmark[end_limb].visibility > 0.5):
                
                start_displacement = np.sqrt(
                    (landmarks2.pose_landmarks.landmark[start_limb].x - landmarks1.pose_landmarks.landmark[start_limb].x)**2 +
                    (landmarks2.pose_landmarks.landmark[start_limb].y - landmarks1.pose_landmarks.landmark[start_limb].y)**2
                ) 
                
                end_displacement = np.sqrt(
                    (landmarks2.pose_landmarks.landmark[end_limb].x - landmarks1.pose_landmarks.landmark[end_limb].x)**2 +
                    (landmarks2.pose_landmarks.landmark[end_limb].y - landmarks1.pose_landmarks.landmark[end_limb].y)**2
                )
                # Calculate the average displacement
                avg_displacement = (start_displacement + end_displacement) / 2     
                speed.append(avg_displacement)    
        # Return the average speed of the group
        return np.mean(speed) if speed else 0

    def calculate_motion(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, prev_frame = cap.read()
        if not ret:
            return 0
        
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
        prev_frame.flags.writeable = False
        # Extract the landmarks from the first frame
        prev_landmarks = self.pose.process(prev_frame)
        
        motion_scores = []
        sizes = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            # Extract the landmarks from the current frame
            landmarks = self.pose.process(frame)
            
            if prev_landmarks.pose_landmarks and landmarks.pose_landmarks:
                size = self.calculate_limb_size(prev_landmarks, landmarks)
                motion = self.calculate_limb_speeds(prev_landmarks, landmarks)
                if size > 0:
                    # Append the motion and size to the lists
                    motion_scores.append(motion)
                    sizes.append(size)
            # Update the previous landmarks
            prev_landmarks = landmarks
        
        cap.release()
        
        if not motion_scores:
            return 0
            
        scale_factor = 10 / np.median(sizes) # Scale the motion scores based on the median size
        motion_score = np.percentile(motion_scores, 75) * scale_factor * fps # Calculate the motion score
        return motion_score

    def analyze_action_directory(self, action_dir):
        action_speeds = []
        for video_file in action_dir.glob("*.avi"): # Iterate over all the videos in the action directory
                motion = self.calculate_motion(str(video_file))
                if motion > 0:
                    action_speeds.append(motion)      
        if action_speeds:
            return np.mean(action_speeds) # Return the average speed of the action
        return 0

    def classify_actions(self, source_dir):
        source_dir = Path(source_dir).expanduser()
        
        # First pass: calculate average speed for each action

        for action_dir in source_dir.iterdir():
            if not action_dir.is_dir():
                continue
            
            avg_speed = self.analyze_action_directory(action_dir)
            self.action_speeds[action_dir.name] = avg_speed

        # Calculate speed thresholds using percentiles of action speeds
        speeds = list(self.action_speeds.values())
        slow_threshold = np.percentile(speeds, 33)
        medium_threshold = np.percentile(speeds, 66)

        action_categories = {
            'slow': [],
            'medium': [],
            'fast': []
        }
        # Second pass: categorize actions based on speed thresholds
        for action, speed in self.action_speeds.items():
            if speed < slow_threshold:
                action_categories['slow'].append(action)
            elif speed < medium_threshold:
                action_categories['medium'].append(action)
            else:
                action_categories['fast'].append(action)

        return action_categories

def organize_actions(source_dir, dest_base):
    analyzer = ActionSpeedAnalyzer()
    source_dir = Path(source_dir).expanduser()
    dest_base = Path(dest_base).expanduser()

    action_categories = analyzer.classify_actions(source_dir)

    for speed_category, actions in action_categories.items():
        speed_dir = dest_base / speed_category
        speed_dir.mkdir(parents=True, exist_ok=True)
        
        for action in actions:
            source_action_dir = source_dir / action
            dest_action_dir = speed_dir / action
            
            if dest_action_dir.exists():
                shutil.rmtree(dest_action_dir)

            # Copy the action directory to the destination
            print(f"Copying {source_action_dir} to {dest_action_dir}")
            shutil.copytree(source_action_dir, dest_action_dir)


if __name__ == "__main__":
    source_dir = "~/Documents/videosTest"
    dest_base = "~/Documents/action_speed_categories"
    organize_actions(source_dir, dest_base)