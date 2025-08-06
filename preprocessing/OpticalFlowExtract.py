import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

class OpticalFlowExtract:
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
    
    def compute_optical_flow(self, video_path):
        cap = cv2.VideoCapture(video_path)
        flow_images = []
        
        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return np.array([])
            
        prev_frame = self.preprocess_frame(prev_frame)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            next_frame = self.preprocess_frame(frame)
            flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_image = self.optical_flow_to_hsv(flow[..., 0], flow[..., 1])
            flow_image = cv2.resize(flow_image, (224, 224))
            flow_images.append(flow_image)
            prev_frame = next_frame
            
        cap.release()
        return np.array(flow_images)

    def process_video(self, video_path, speed_category, complexity_category, action_class, output_filename):
        """Process a single video"""
        try:
            optical_flows = self.compute_optical_flow(video_path)
            
            metadata = {
                'speed': speed_category,
                'complexity': complexity_category,
                'action_class': action_class
            }
            
            np.savez_compressed(output_filename, optical_flows=optical_flows, metadata=metadata)
            return True
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            return False

    def extract_from_folder(self, directory_path, output_dir, pose_features_dir):
        """Extract features from all videos in the directory"""
        train_output_dir = os.path.join(output_dir, 'train')
        test_output_dir = os.path.join(output_dir, 'test')
        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(test_output_dir, exist_ok=True)

        # Collect all processing tasks
        processing_tasks = []
        train_videos = self.get_videos(pose_features_dir, 'train')
        test_videos = self.get_videos(pose_features_dir, 'test')
        
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

                    for video_file in video_files:
                        video_base_name = os.path.splitext(video_file)[0]
                        
                        is_train = video_base_name in train_videos.get(action_class, set())
                        is_test = video_base_name in test_videos.get(action_class, set())
                            
                        output_dir_path = train_class_dir if is_train else test_class_dir
                        video_path = os.path.join(class_path, video_file)
                        output_filename = os.path.join(
                            output_dir_path,
                            f"{video_base_name}_optical_flow.npz"
                        )
                        
                        processing_tasks.append((
                            video_path,
                            speed_category,
                            complexity_category,
                            action_class,
                            output_filename
                        ))
        videos=["HorseRiding","BalanceBeam","BodyWeightSquats","Bowling","clap","climb","FrisbeeCatch","handstand","HandstandPushups","JumpRope","kick","MoppingFloor","Punch","situp","Swing","turn","walk","WallPushups"]
        for task in tqdm(processing_tasks, desc="Processing optical flow to HSV"):
            video_path, speed_category, complexity_category, action_class, output_filename = task
            if action_class in videos:
                self.process_video(video_path, speed_category, complexity_category, action_class, output_filename)

        print("\nOptical flow extraction complete!")

    def get_videos(self, direc, split):
        split_dir = os.path.join(direc, split)
        videos = {}
        for action_class in os.listdir(split_dir):
            class_path = os.path.join(split_dir, action_class)
            video_names = []
            feature_files = glob.glob(os.path.join(class_path, '*_features.npz'))
            for file in feature_files:
                video_names.append(os.path.basename(file).replace('_features.npz', ''))
                
            videos[action_class] = video_names

        return videos


def main():
    extractor = OpticalFlowExtract()
    
    video_dir = '/users/k22001386/action_speed_categories'
    output_dir = '/scratch/prj/inf_media_pipe_dynamics/optical_flow_features'
    pose_features_dir = '/users/k22001386/HAR/HAR/processed_features2'
    
    extractor.extract_from_folder(video_dir, output_dir, pose_features_dir)
    
    print("Extraction complete!")


if __name__ == "__main__":
    main()