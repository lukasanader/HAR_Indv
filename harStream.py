import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from HAR_Indv.models.resnetModel import ResNet3DClassifier
from HAR_Indv.models.model import LSTMModel
import torchvision.transforms as transforms
import threading
import queue
import time

class Stream:
    def __init__(self,resnet_model_path,lstm_model_path,weight=0.6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet =  torch.load(resnet_model_path, map_location=self.device,weights_only=False)
        self.lstm =  torch.load(lstm_model_path, map_location=self.device,weights_only=False)
        self.resnet.eval()
        self.lstm.eval()
        self.resnet_weight=weight
        self.lstm_weight=1.0-weight
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
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
        )
        self.actions = ["BabyCrawling","BalanceBeam","BaseballPitch","Basketball","BasketballDunk","BenchPress",
        "Biking","BodyWeightSquats","Bowling","Diving","FrisbeeCatch",
        "HandstandPushups","HandstandWalking","HorseRiding","JumpRope","JumpingJack",
        "LongJump","Lunges","MoppingFloor","PullUps","Punch","Rowing","Shotput","SkateBoarding","Swing","WallPushups","clap","climb","handstand","jump","kick","push","run","sit","situp","somersault","stand","throw","turn","walk","wave"]
    

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
        # Extract pose landmarks
        landmarks = []
        res_landmarks = self.pose.process(self.preprocess_frame(frame))
        frame_landmarks = []
        if res_landmarks.pose_landmarks:
            for _, idx in self.keypoints.items():
                landmark = res_landmarks.pose_landmarks.landmark[idx]
                frame_landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        else:
            frame_landmarks = [0, 0, 0, 0] * len(self.keypoints)
                    
        # Display landmarks on the frame using mediapipe
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(frame, res_landmarks.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        # Draw keypoints
        
        landmarks.append(frame_landmarks)
        return landmarks

    def compute_optical_flow(self,prev_frame,next_frame):
        # Compute optical flow
        prev_frame = self.preprocess_frame(prev_frame)
        next_frame = self.preprocess_frame(next_frame)
        if prev_frame.shape != next_frame.shape:
            next_frame = cv2.resize(next_frame, (prev_frame.shape[1], prev_frame.shape[0]))
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_image = self.optical_flow_to_hsv(flow[..., 0], flow[..., 1])
        flow_image = cv2.resize(flow_image, (224, 224))
        # Return optical flow image
        return flow_image

    def get_prediction(self,final_output):
        action_idx = torch.argmax(final_output, dim=1).item()
        action_name = self.actions[action_idx]
        probability = torch.nn.functional.softmax(final_output, dim=1)[0][action_idx].item() * 100

        return action_name, probability

    
    def har(self, sequence_length=30, confidence_threshold=50.0):
        """
        Human Activity Recognition with non-blocking prediction
        
        Args:
            sequence_length: Number of frames to collect for prediction
            confidence_threshold: Minimum confidence to display a prediction
            resnet_weight: Weight given to ResNet model (0.0-1.0)
        """
        
        # Create a prediction queue and result variable
        pred_queue = queue.Queue()
        prediction_result = {"action": "Waiting...", "confidence": 0.0}
        
        # Flag to control the prediction thread
        running = True
        
        def prediction_worker():
            """Worker thread function to process predictions"""
            while running:
                try:
                    # Get data from queue with a timeout to allow thread to exit
                    data = pred_queue.get(timeout=1.0)
                    
                    # Unpack the data
                    landmarks, flows = data
                    flows = np.array(flows, dtype=np.float32)
                    # Prepare inputs
                    lstm_input = torch.from_numpy(np.array(landmarks, dtype=np.float32)).unsqueeze(0).to(self.device)
                    resnet_input = torch.from_numpy(np.transpose(flows, (3, 0, 1, 2))).unsqueeze(0).to(self.device) / 255.0
                    
                    # Fix dimensions
                    if lstm_input.dim() == 4:
                        lstm_input = lstm_input.squeeze(2)
                    
                    
                    with torch.no_grad(): 
                        lstm_output = self.lstm(lstm_input)
                        resnet_output = self.resnet(resnet_input)
                        
                        # Combine outputs with revised weights
                        final_output = self.lstm_weight * lstm_output + self.resnet_weight * resnet_output
                        
                        # Get top prediction and its confidence
                        pred_class, confidence  = self.get_prediction(final_output)
                        
                        if confidence >= confidence_threshold:
                            prediction_result["action"] = pred_class
                            prediction_result["confidence"] = confidence
                    
                    pred_queue.task_done()
                except queue.Empty:
                    pass
                except Exception as e:
                    print(e)
                    
        # Start prediction thread
        pred_thread = threading.Thread(target=prediction_worker, daemon=True)
        pred_thread.start()
        cap = cv2.VideoCapture(0)
        prev_frame = None
        landmark_sequence = []
        optical_flow_sequence = []
        frame_count = 0
        last_prediction_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    landmarks = self.extract_landmarks(frame)
                    landmark_sequence.append(landmarks[0])
                    if len(landmark_sequence) > sequence_length:
                        landmark_sequence.pop(0)
                except Exception as e:
                    print(e)
                    
                frame_count += 1
                
                # Compute optical flow
                if prev_frame is not None:
                    try:
                        flow = self.compute_optical_flow(prev_frame, frame)
                        optical_flow_sequence.append(flow)
                        if len(optical_flow_sequence) > 16:
                            optical_flow_sequence.pop(0)
                    except Exception as e:
                        print(e)
                
                prev_frame = frame.copy()
                
                # Display current action with confidence
                current_action = f"{prediction_result['action']} ({prediction_result['confidence']:.1f}%)"
                cv2.putText(frame, f"Action: {current_action}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2) 
                
                
                current_time = time.time()
                if (current_time - last_prediction_time >= 3.0 and 
                    len(landmark_sequence) >= sequence_length and 
                    len(optical_flow_sequence) >= 16 and
                    pred_queue.empty()):
                    
                    # Make a copy of the data for the prediction thread
                    landmarks_copy = landmark_sequence.copy()
                    flows_copy = optical_flow_sequence.copy()
                    
                    # Put data in queue for prediction thread
                    pred_queue.put((landmarks_copy, flows_copy))
                    last_prediction_time = current_time
                
                cv2.imshow("Action Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            # Clean up
            running = False
            if pred_thread.is_alive():
                pred_thread.join(timeout=2.0)
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    resnet_model_path = 'resnet_model.pt'
    lstm_model_path = 'lstm_model.pt'
    stream = Stream(resnet_model_path,lstm_model_path)
    stream.har()