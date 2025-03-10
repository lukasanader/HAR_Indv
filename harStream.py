import cv2
import mediapipe as mp
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from resnetModel import ResNet3DClassifier
from model import LSTMModel
import torchvision.transforms as transforms
class Stream:
    def __init__(self,resnet_model_path,lstm_model_path,weight=0.6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet =  torch.load(resnet_model_path, map_location=self.device,weights_only=False)
        self.lstm =  torch.load(lstm_model_path, map_location=self.device,weights_only=False)
        self.resnet.eval()
        self.lstm.eval()
        self.resnet_weight=weight
        self.lstm.weight=1.0-weight
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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
    
    def optical_flow_to_hsv(self, flow_x, flow_y):
        """
        Convert optical flow components to HSV image format
        - Hue: direction (angle) of flow
        - Saturation: always max (255)
        - Value: magnitude of flow
        """
        magnitude, angle = cv2.cartToPolar(flow_x, flow_y)
        
        magnitude = np.clip(magnitude * 20, 0, 255).astype(np.uint8)
        
        # Create HSV image
        hsv = np.zeros((flow_x.shape[0], flow_x.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = magnitude
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return rgb
    

    def extract_landmarks(self,frame):
        landmarks = []
        res_landmarks = self.pose.process(self.preprocess_frame(frame))
        frame_landmarks = []
        if res_landmarks.pose_landmarks:
            for _, idx in self.keypoints.items():
                landmark = res_landmarks.pose_landmarks.landmark[idx]
                frame_landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        else:
            frame_landmarks = [0, 0, 0, 0] * len(self.keypoints)
                    
        landmarks.append(frame_landmarks)
        return landmarks

    def compute_optical_flow(self,prev_frame,next_frame):
        prev_frame = self.preprocess_frame(prev_frame)
        next_frame = self.preprocess_frame(next_frame)
        flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_image = self.optical_flow_to_hsv(flow[..., 0], flow[..., 1])
        flow_image = cv2.resize(flow_image, (224, 224))
        return flow_image

        
    def har(sequence_length=50):
        actions = ["BabyCrawling","BalanceBeam","BaseballPitch","Basketball","BasketballDunk","BenchPress",
        "Biking","BodyWeightSquats","Bowling","clap","climb","Diving","FrisbeeCatch","handstand",
        "HandstandPushups","HandstandWalking","HorseRiding","jump","JumpingJack","JumpRope","kick",
        "LongJump","Lunges","MoppingFloor","PullUps","Punch","push","Rowing","run","Shotput","sit",
        "situp","SkateBoarding","somersault","stand","Swing","throw","turn","walk","WallPushups","wave"]

        cap = cv2.VideoCapture(0)
        prev_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract landmarks
            landmarks = self.extract_landmarks(frame)
            landmark_sequence.append(landmarks)
            if len(landmark_sequence) > sequence_length:
                landmark_sequence.pop(0)

            # Compute optical flow
            if prev_frame is not None:
                flow = self.compute_optical_flow(prev_frame, frame)
                optical_flow_sequence.append(flow)
                if len(optical_flow_sequence) > sequence_length:
                    optical_flow_sequence.pop(0)
            
            prev_frame = frame.copy()

            if len(landmark_sequence) == sequence_length and len(optical_flow_sequence) == sequence_length:
                # Convert to tensors
                lstm_input = torch.tensor([landmark_sequence], dtype=torch.float32).to(device)
                resnet_input = torch.tensor([optical_flow_sequence], dtype=torch.float32).permute(0, 4, 1, 2, 3).to(device)

                # Get predictions
                lstm_output = lstm_model(lstm_input)
                resnet_output = resnet3d_model(resnet_input)

                final_output = lstm_weight * lstm_output + res_weight * resnet_output
                action_idx = torch.argmax(final_output, dim=1).item()
                action_name = actions[action_idx]

                # Display result
                cv2.putText(frame, f"Action: {action_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Action Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    resnet_model_path = 'resnet_model.pt'
    lstm_model_path = 'lstm_model.pt'
    stream = Stream(resnet_model_path,lstm_model_path)
    stream.har()


