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

class ResNet3DClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNet3DClassifier, self).__init__()
        self.resnet3d = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)

        in_features = self.resnet3d.fc.in_features
        self.resnet3d.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.resnet3d(x)

class ResTrainer:
    def __init__(self, num_classes, learning_rate, batch_size=32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        torch.cuda.empty_cache()
        # self.model = ResNet3DClassifier(num_classes).to(self.device)
        self.model = torch.load('resnet_model.pt', map_location=self.device,weights_only=False)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.label_encoder = LabelEncoder()
    
    def get_data(self, direc):
        X_train=[]
        y_train=[]
        X_val = []
        y_val = []
        test_meta=[]
        for split in ['test']:
            split_path=os.path.join(direc, split)
            for action in os.listdir(split_path):
                action_path=os.path.join(split_path,action)
                for file in os.listdir(action_path):
                    if file.endswith('.npz'):
                        file_path=os.path.join(action_path,file)
                        data = np.load(file_path, allow_pickle=True)
                        optical_flows = data['optical_flows']
                        metadata = data['metadata']
                        meta_dict = metadata.item()

                        if len(optical_flows) < 16:
                            indices = np.array(list(range(len(optical_flows))) * (16 // len(optical_flows) + 1))
                            indices = indices[:16]
                            optical_flows = optical_flows[indices]
                        elif len(optical_flows) > 16:
                            # Uniformly sample 16 frames
                            indices = np.linspace(0, len(optical_flows)-1, 16, dtype=int)
                            optical_flows = optical_flows[indices]
                        reshaped_flow = np.transpose(optical_flows, (3, 0, 1, 2))
                        if split == "train":
                            X_train.append(reshaped_flow)
                            y_train.append(action)
                        else:
                            X_val.append(reshaped_flow)
                            y_val.append(action)
                            test_meta.append(meta_dict)
                        
        X_train_np = np.array(X_train, dtype=np.float32)
        X_val_np = np.array(X_val, dtype=np.float32)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.fit_transform(y_val)
        y_train_np = np.array(y_train_encoded, dtype=np.int64)
        y_val_np = np.array(y_val_encoded, dtype=np.int64)
        
        # Create TensorDatasets
        train_dataset = TensorDataset(torch.from_numpy(X_train_np) /255.0, torch.from_numpy(y_train_np))
        val_dataset = TensorDataset(torch.from_numpy(X_val_np)/255.0, torch.from_numpy(y_val_np))
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_meta

    def train(self, train_loader):
        num_epochs = 15
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            batch_count = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            avg_loss = total_loss / batch_count
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}")
    
    def analyse_speed(self, all_labels, all_preds, test_meta):
        speeds = ['slow', 'medium', 'fast']
        accuracies = []
        for speed in speeds:
            indices = []
            for i in range(len(test_meta)):
                if test_meta[i].get('speed') == speed:
                    indices.append(i)
            
            if indices:  # Check if there are any samples for this speed
                speed_true = [all_labels[i] for i in indices]
                speed_pred = [all_preds[i] for i in indices]
                accuracy = np.mean(np.array(speed_true) == np.array(speed_pred))
                accuracies.append(accuracy)
                print(f"Speed '{speed}': Accuracy = {accuracy:.4f}")
            else:
                print(f"No samples for speed '{speed}'")

        plt.figure(figsize=(10, 6))
        plt.bar(speeds, accuracies)
        plt.title('Speed analysis')
        plt.xlabel('Speed')
        plt.ylabel('Accuracy')
        plt.savefig('res_speed.png')
        plt.close()

    def analyse_complexity(self, all_labels, all_preds, test_meta):
        complexities = ['simple', 'complex']
        accuracies = []
        for complexity in complexities:
            indices = []
            for i in range(len(test_meta)):
                if test_meta[i].get('complexity') == complexity:
                    indices.append(i)
            
            if indices:  # Check if there are any samples for this complexity
                comp_true = [all_labels[i] for i in indices]
                comp_pred = [all_preds[i] for i in indices]
                accuracy = np.mean(np.array(comp_true) == np.array(comp_pred))
                accuracies.append(accuracy)
                print(f"Complexity '{complexity}': Accuracy = {accuracy:.4f}")
            else:
                print(f"No samples for complexity '{complexity}'")
            plt.figure(figsize=(10, 6))
            plt.bar(complexities, accuracies)
            plt.title('Complexity analysis')
            plt.xlabel('Complexity')
            plt.ylabel('Accuracy')
            plt.savefig('res_complexity.png')
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
            plt.savefig('per_class_recall_res.png')
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
        plt.savefig('per_class_precision_res.png')
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
        plt.savefig('per_class_f1_res.png')
        plt.close()

    def evaluate(self, val_loader, test_meta):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            all_labels, 
            all_preds, 
            target_names=self.label_encoder.classes_
        ))

        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrixRes.png')
        plt.close()
        
        self.analyse_speed(all_labels, all_preds, test_meta)
        self.analyse_complexity(all_labels, all_preds, test_meta)
        self.per_class_recall_graph(all_labels,all_preds)
        self.per_class_precision_graph(all_labels,all_preds)
        self.per_class_f1_graph(all_labels,all_preds)

if __name__ == "__main__":
    optical_flow_dir = '/scratch/prj/inf_media_pipe_dynamics/optical_flow_features'

    batch_size = 16
    
    trainer = ResTrainer(
        num_classes=41,
        learning_rate=1e-4,
        batch_size=batch_size
    )
    
    train_loader, val_loader, test_meta = trainer.get_data(optical_flow_dir)
    trainer.train(train_loader)
    trainer.evaluate(val_loader, test_meta)
    torch.save(trainer.model,'resnet_model.pt')
    