import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from model import Trainer
from model import LSTMModel
from resnetModel import ResTrainer
from resnetModel import ResNet3DClassifier
import re

class Ensemble:
    def __init__(self,resnet_model_path,lstm_model_path,weight=0.6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet =  torch.load(resnet_model_path, map_location=self.device,weights_only=False)
        self.lstm =  torch.load(lstm_model_path, map_location=self.device,weights_only=False)
        self.resnet.eval()
        self.lstm.eval()
        self.resnet_weight=weight
        self.lstm_weight=1.0-weight
        self.label_encoder = LabelEncoder()
        self.batch_size=16


    def get_data(self, landmark_dir, optical_flow_dir):
        sequence_length = 40
        X_val_land = []
        X_val_flow = []
        y_val = []  
        test_meta = []
        landmark_files = {}
        optical_flow_files = {}

        # Extract and map landmark files
        for action in os.listdir(landmark_dir):
            action_path = os.path.join(landmark_dir, action)
            for f in os.listdir(action_path):
                if f.endswith('_features.npz'):
                    base_name = f.replace('_features.npz', ' ')  # Remove suffix
                    landmark_files[base_name] = (os.path.join(action_path, f), action)

        # Extract and map optical flow files
        for action in os.listdir(optical_flow_dir):
            action_path = os.path.join(optical_flow_dir, action)
            for f in os.listdir(action_path):
                if f.endswith('_optical_flow.npz'):
                    base_name = f.replace('_optical_flow.npz', ' ')  # Remove suffix
                    optical_flow_files[base_name] = os.path.join(action_path, f)

        common_files = sorted(set(landmark_files.keys()) & set(optical_flow_files.keys()))

        for file_key in common_files:
            land_path, action = landmark_files[file_key]
            flow_path = optical_flow_files[file_key]

            # Load landmark data
            land_data = np.load(land_path, allow_pickle=True)
            landmarks = land_data['landmarks']
            metadata = land_data.get('metadata', {})
            meta_dict = metadata.item()

            # Load optical flow data
            flow_data = np.load(flow_path, allow_pickle=True)
            optical_flows = flow_data['optical_flows']

            # Process optical flow data (ensure exactly 16 frames)
            if len(optical_flows) < 16:
                indices = np.array(list(range(len(optical_flows))) * (16 // len(optical_flows) + 1))[:16]
                optical_flows = optical_flows[indices]
            else:
                indices = np.linspace(0, len(optical_flows) - 1, 16, dtype=int)
                optical_flows = optical_flows[indices]

            reshaped_flow = np.transpose(optical_flows, (3, 0, 1, 2))
            
            # Process landmark sequences
            for i in range(0, len(landmarks) - sequence_length + 1, sequence_length // 2):
                sequence = landmarks[i:i+sequence_length]
                if len(sequence) == sequence_length:
                    X_val_land.append(sequence)
                    X_val_flow.append(reshaped_flow)  # Use the same optical flow data for each landmark sequence
                    y_val.append(action)
                    test_meta.append(meta_dict)

        X_val_land = np.array(X_val_land, dtype=np.float32)
        X_val_flow = np.array(X_val_flow, dtype=np.float32)
        
        self.label_encoder = LabelEncoder()
        y_val_encoded = self.label_encoder.fit_transform(y_val)
        y_val = np.array(y_val_encoded, dtype=np.int64)

        # Create a custom dataset
        dataset = TensorDataset(torch.from_numpy(X_val_land), torch.from_numpy(X_val_flow)/255.0, torch.from_numpy(y_val))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        return dataloader, test_meta


    def analyse_speed(self, all_labels, all_preds, test_meta):
        speeds = ['slow', 'medium', 'fast']
        accuracies = []
        for speed in speeds:
            indices = []
            for i in range(len(test_meta)):
                if test_meta[i].get('speed') == speed:
                    indices.append(i)

            speed_true = [all_labels[i] for i in indices]
            speed_pred = [all_preds[i] for i in indices]
            accuracy = np.mean(np.array(speed_true) == np.array(speed_pred))
            accuracies.append(accuracy)

        plt.figure(figsize=(10, 6))
        plt.bar(speeds, accuracies)
        plt.title('Speed analysis')
        plt.xlabel('Speed')
        plt.ylabel('Accuracy')
        plt.savefig('ens_speed.png')
        plt.close()

    def analyse_complexity(self, all_labels, all_preds, test_meta):
        complexities = ['simple', 'complex']
        accuracies = []
        for complexity in complexities:
            indices = []
            for i in range(len(test_meta)):
                if test_meta[i].get('complexity') == complexity:
                    indices.append(i)
            comp_true = [all_labels[i] for i in indices]
            comp_pred = [all_preds[i] for i in indices]
            accuracy = np.mean(np.array(comp_true) == np.array(comp_pred))
            accuracies.append(accuracy)

        plt.figure(figsize=(10, 6))
        plt.bar(complexities, accuracies)
        plt.title('Complexity analysis')
        plt.xlabel('Complexity')
        plt.ylabel('Accuracy')
        plt.savefig('ens_complexity.png')
        plt.close()

    def per_class_recall_graph(self, all_labels, all_preds, num_classes=41):
            class_recall = []
            for i in range(num_classes):
                true_positive=0
                false_negative=0
                for j in range(len(all_labels)):
                    if all_labels[j] == i:
                        if all_preds[j] == i:
                            true_positive+=1
                        else:
                            false_negative+=1
                recall = true_positive/(true_positive+false_negative)
                if (true_positive+false_negative)!=0:
                    recall = true_positive/(true_positive+false_negative)
                else:
                    recall=0
                class_recall.append(recall)
            plt.figure(figsize=(12,6))
            plt.bar(self.label_encoder.classes_, class_recall)
            plt.xlabel('Class')
            plt.ylabel('Recall')
            plt.xticks(rotation=90)
            plt.title('Per Class Recall')
            plt.tight_layout()
            plt.savefig('per_class_recall_ens.png')
            plt.close()

    def per_class_precision_graph(self, all_labels, all_preds, num_classes=41):
        class_precision = []
        for i in range(num_classes):
            true_positive=0
            false_positive=0
            for j in range(len(all_labels)):
                if all_preds[j] == i:
                    if all_labels[j] == i:
                        true_positive+=1
                    else:
                        false_positive+=1
            if (true_positive+false_positive)!=0:
                precision = true_positive/(true_positive+false_positive)
            else:
                precision=0
            class_precision.append(precision)
        plt.figure(figsize=(12,6))
        plt.bar(self.label_encoder.classes_, class_precision)
        plt.xlabel('Class')
        plt.ylabel('Precision')
        plt.xticks(rotation=90)
        plt.title('Per Class Precision')
        plt.tight_layout()
        plt.savefig('per_class_precision_ens.png')
        plt.close()

    def per_class_f1_graph(self, all_labels, all_preds, num_classes=41):
        class_f1 = []
        for i in range(num_classes):
            true_positive=0
            false_positive=0
            false_negative=0
            for j in range(len(all_labels)):
                if all_preds[j] == i:
                    if all_labels[j] == i:
                        true_positive+=1
                    else:
                        false_positive+=1
                elif all_labels[j] == i:
                    false_negative+=1
            precision = true_positive/(true_positive+false_positive)
            recall = true_positive/(true_positive+false_negative)
            if (precision+recall)!=0:
                f1 = 2*precision*recall/(precision+recall)
            else:
                f1=0
            class_f1.append(f1)
        plt.figure(figsize=(12,6))
        plt.bar(self.label_encoder.classes_, class_f1)
        plt.xlabel('Class')
        plt.ylabel('F1 Score')
        plt.xticks(rotation=90)
        plt.title('Per Class F1 Score')
        plt.tight_layout()
        plt.savefig('per_class_f1_ens.png')
        plt.close()
    
    def evaluate(self,dataloader,test_meta):
        print("Evaluating...")
        all_preds = []
        all_labels = []
        for landmarks, optical_flows, labels in tqdm(dataloader):
            landmarks = landmarks.to(self.device)
            optical_flows = optical_flows.to(self.device)
            labels = labels.to(self.device)

            lstm_outputs = self.lstm(landmarks)
            resnet_outputs = self.resnet(optical_flows)

            resnet_preds.extend(resnet_predicted.cpu().numpy())

            final_output = self.lstm_weight * lstm_outputs + self.resnet_weight * resnet_outputs

            _, predicted = torch.max(final_output, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Classification report
        print("\nEnsemble Classification Report:")            
        print(classification_report(all_labels, all_preds, target_names=self.label_encoder.classes_))
        
        # Analyze by speed and complexity
        self.analyse_speed(all_labels, all_preds, test_meta)
        self.analyse_complexity(all_labels, all_preds, test_meta)
        self.per_class_recall_graph(all_labels,all_preds)
        self.per_class_precision_graph(all_labels,all_preds)
        self.per_class_f1_graph(all_labels,all_preds)

if __name__ == "__main__":
    resnet_model_path = 'resnet_model.pt'
    lstm_model_path = 'lstm_model.pt'
    landmark_dir = '/users/k22001386/HAR/HAR/processed_features2/test'
    optical_flow_dir = '/scratch/prj/inf_media_pipe_dynamics/optical_flow_features/test'


    ensemble = Ensemble(resnet_model_path,lstm_model_path)
    dataloader,test_meta = ensemble.get_data(landmark_dir,optical_flow_dir)
    ensemble.evaluate(dataloader,test_meta)


