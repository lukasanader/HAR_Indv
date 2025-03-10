import numpy as np


class Augmentation:
    def __init__(self, landmarks):
        self.landmarks = landmarks

    def rotate(self,landmarks, max_angle=10):
        # landmarks is a list of x,y,z,visibility
        # Rotate the landmarks
        new_landmarks = []
        rotation_angle = np.radians(np.random.uniform(-max_angle, max_angle))
        rotation_matrix_3d= np.array([[np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                                      [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                                      [0, 0, 1]])
        landmarks = np.array(landmarks).reshape(-1, 4)
        for landmark in landmarks:
            x,y,z,visibility = landmark
            new_x, new_y, new_z = np.dot(rotation_matrix_3d, [x,y,z])
            new_landmarks.append([new_x, new_y, new_z, visibility])

        return new_landmarks;
    
    def flip(self, landmarks):
        # Flip the landmarks
        new_landmarks = []
        for landmark in landmarks:
            x,y,z,visibility = landmark
            new_landmarks.append([-x, y, z, visibility])
        return new_landmarks
    
    def translate(self, landmarks, max_shift=0.05):
        # Translate the landmarks
        new_landmarks = []
        shift = np.random.uniform(-max_shift, max_shift)
        for landmark in landmarks:
            x,y,z,visibility = landmark
            new_landmarks.append([x + shift, y + shift, z, visibility])
        return new_landmarks
    
    def augment_frame(self, frame):
        new_landmarks = frame
        new_landmarks = self.rotate(new_landmarks)
        new_landmarks = self.flip(new_landmarks)
        new_landmarks = self.translate(new_landmarks)
        return np.array(new_landmarks).flatten()
    
    def augment(self, landmarks):
        # Augment the landmarks
        augmented_landmarks = []
        for frame in landmarks:
            new_landmarks = self.augment_frame(frame)
            augmented_landmarks.append(new_landmarks)
        return augmented_landmarks