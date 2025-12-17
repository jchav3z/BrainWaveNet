import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from model import BrainWaveNet 

# CONFIGURACIÓN
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHTS_SAVE_PATH = 'models/brainwavenet_weights.pth' 
TIME_STEPS = 3000

# CLASES
class SleepDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# FUNCIONES
def apply_bandpass_filter(data, fs=100, lowcut=0.5, highcut=40.0):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data, axis=-1)
    return filtered_data

def load_and_prepare_data():
    # SIMULACIÓN DE DATOS
    N_SAMPLES = 1000
    X_raw = np.random.randn(N_SAMPLES, TIME_STEPS)
    y = np.random.randint(0, 5, N_SAMPLES)
    
    X_filtered = apply_bandpass_filter(X_raw)
    
    mean = X_filtered.mean(axis=1, keepdims=True)
    std = X_filtered.std(axis=1, keepdims=True)
    X_normalized = (X_filtered - mean) / (std + 1e-6)
    
    X_final = X_normalized[:, np.newaxis, :]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_final, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_val, y_train, y_val

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item() * inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='macro')
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, f1, accuracy

def train_model():
    X_train, X_val, y_train, y_val = load_and_prepare_data()
    train_dataset = SleepDataset(X_train, y_train)
    val_dataset = SleepDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = BrainWaveNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_f1 = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss, val_f1, val_acc = evaluate_model(model, val_loader, criterion)
        
        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), WEIGHTS_SAVE_PATH)
            print(f"Modelo guardado: Mejor F1-score {best_f1:.4f}")
    
if __name__ == '__main__':
    train_model()