import torch
import torch.nn as nn

class BrainWaveNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=5, time_steps=3000):
        super(BrainWaveNet, self).__init__()
        
        LINEAR_INPUT_SIZE = 64 * 19 

        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=64, stride=16, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Dropout(0.5)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(LINEAR_INPUT_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits

if __name__ == '__main__':
    input_tensor = torch.randn(1, 1, 3000) 
    model = BrainWaveNet()
    output_conv = model.conv_layers(input_tensor)
    print(f"DEBUG: Dimensiones después de CONV/POOL: {output_conv.shape}")
    output = model(input_tensor)
    print(f"Salida del modelo (logits): {output.shape}")