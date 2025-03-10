import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from data_aug import Augmentation
import gc



class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=4):
        super().__init__()
        self.hidden_size = hidden_size
        
        # First LSTM layer for capturing basic motion
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Attention mechanism for focusing on important frames
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Second LSTM for capturing higher-level motion patterns
        self.lstm2 = nn.LSTM(hidden_size, hidden_size//2, batch_first=True)
        
        # Output classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size//2, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # First LSTM layer
        lstm1_out, _ = self.lstm1(x)
        
        # Attention mechanism
        attn_weights = F.softmax(self.attention(lstm1_out).squeeze(-1), dim=1)
        attn_weights = attn_weights.unsqueeze(2).expand(-1, -1, self.hidden_size)
        weighted_output = torch.sum(lstm1_out * attn_weights, dim=1)
        
        # Reshape for second LSTM
        weighted_output = weighted_output.unsqueeze(1).expand(-1, x.size(1), -1)
        
        # Second LSTM layer
        lstm2_out, _ = self.lstm2(weighted_output)
        
        # Get the final time step output
        final_out = lstm2_out[:, -1, :]
        
        # Classification
        output = self.classifier(final_out)
        return output

class Trainer:
    def __init__(self, input_size, hidden_size, num_classes, num_layers, learning_rate, batch_size=32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        gc.collect()
        torch.cuda.empty_cache()
        self.model = LSTMModel(input_size, hidden_size, num_classes, num_layers).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.label_encoder = LabelEncoder()
    
    def get_data(self, direc):
        sequence_length=60
        X_train=[]
        y_train=[]
        X_val = []
        y_val = []
        test_meta=[]
        for split in ['train', 'test']:
            split_path=os.path.join(direc, split)
            for action in os.listdir(split_path):
                action_path=os.path.join(split_path,action)
                for file in os.listdir(action_path):
                    if file.endswith('.npz'):
                        file_path=os.path.join(action_path,file)
                        data = np.load(file_path, allow_pickle=True)
                        landmarks = data['landmarks']
                        metadata = data['metadata']
                        if isinstance(metadata, np.ndarray):
                            try:
                                meta_dict = metadata.item()
                                if not isinstance(meta_dict, dict):
                                    meta_dict = {'unknown': 'unknown'}
                            except (ValueError, TypeError):
                                meta_dict = {'unknown': 'unknown'}
                        # extract sequences
                        for i in range(0, len(landmarks) - sequence_length + 1, sequence_length // 2):
                            sequence = landmarks[i:i+sequence_length]
                            if len(sequence)==sequence_length:
                                if split == "train":
                                    X_train.append(sequence)
                                    y_train.append(action)
                                else:
                                    X_val.append(sequence)
                                    y_val.append(action)
                                    test_meta.append(meta_dict)
        class_count={}
        for i in y_train:
            if i in class_count:
                class_count[i]+=1
            else:
                class_count[i]=1
    
        max_count=max(class_count.values())
        augment = Augmentation(None)
        X_train_balanced = []
        y_train_balanced = []
        for action, count in class_count.items():
            class_indices = [i for i, label in enumerate(y_train) if label == action]
            class_samples = [X_train[i] for i in class_indices]
            # Add original samples
            X_train_balanced.extend(class_samples)
            y_train_balanced.extend([action] * len(class_samples))
            if count<max_count:
                num_aug=(max_count-count)//4
                for i in range(num_aug):
                    idx = np.random.randint(0, len(class_samples))
                    sample = class_samples[idx]
                    # Apply augmentation
                    augmented_sample = augment.augment(sample)
                    
                    # Add to balanced dataset
                    X_train_balanced.append(augmented_sample)
                    y_train_balanced.append(action)
        
        X_train_np = np.array(X_train_balanced, dtype=np.float32)
        X_val_np = np.array(X_val, dtype=np.float32)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train_balanced)
        y_val_encoded = self.label_encoder.transform(y_val)
        y_train_np = np.array(y_train_encoded, dtype=np.int64)
        y_val_np = np.array(y_val_encoded, dtype=np.int64)
        
        # Create TensorDatasets
        train_dataset = TensorDataset(torch.from_numpy(X_train_np), torch.from_numpy(y_train_np))
        val_dataset = TensorDataset(torch.from_numpy(X_val_np), torch.from_numpy(y_val_np))
        
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
        for speed in speeds:
            indices = []
            for i in range(len(test_meta)):
                if test_meta[i].get('speed') == speed:
                    indices.append(i)
            
            if indices:  # Check if there are any samples for this speed
                speed_true = [all_labels[i] for i in indices]
                speed_pred = [all_preds[i] for i in indices]
                accuracy = np.mean(np.array(speed_true) == np.array(speed_pred))
                print(f"Speed '{speed}': Accuracy = {accuracy:.4f}")
            else:
                print(f"No samples for speed '{speed}'")

    def analyse_complexity(self, all_labels, all_preds, test_meta):
        complexities = ['simple', 'complex']
        for complexity in complexities:
            indices = []
            for i in range(len(test_meta)):
                if test_meta[i].get('complexity') == complexity:
                    indices.append(i)
            
            if indices:  # Check if there are any samples for this complexity
                comp_true = [all_labels[i] for i in indices]
                comp_pred = [all_preds[i] for i in indices]
                accuracy = np.mean(np.array(comp_true) == np.array(comp_pred))
                print(f"Complexity '{complexity}': Accuracy = {accuracy:.4f}")
            else:
                print(f"No samples for complexity '{complexity}'")

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
        
        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8,6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=self.label_encoder.classes_, 
            yticklabels=self.label_encoder.classes_
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        self.analyse_speed(all_labels, all_preds, test_meta)
        self.analyse_complexity(all_labels, all_preds, test_meta)

if __name__ == "__main__":
    landmark_dir = '/users/k22001386/HAR/HAR/processed_features2'

    batch_size = 32
    
    trainer = Trainer(
        input_size=132, 
        hidden_size=256, 
        num_classes=41,
        num_layers=4,
        learning_rate=0.001,
        batch_size=batch_size
    )
    
    train_loader, val_loader, test_meta = trainer.get_data(landmark_dir)
    trainer.train(train_loader)
    trainer.evaluate(val_loader, test_meta)
    torch.save(trainer.model,'lstm_model.pt')