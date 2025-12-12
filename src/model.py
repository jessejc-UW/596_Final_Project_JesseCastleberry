import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
from pathlib import Path
import time

class HeartRateEstimator(nn.Module):
    def __init__(self, 
                 in_channels=1,
                 out_channels=1,
                 conv_channels=[(1,64), (64,128), (128,256)],
                 kernel_sizes=[7, 7, 7],
                 strides=[1, 1, 1],
                 padding_sizes=[3, 3, 3],
                 fc_sizes=[(256,128), (128,1)],
                 dropout=0.05
                ):
        super().__init__()

        self.conv = nn.ModuleList()
        for idx, channel_size in enumerate(conv_channels):
            self.conv.append(nn.Conv1d(in_channels=channel_size[0],
                                             out_channels=channel_size[1],
                                             kernel_size=kernel_sizes[idx],
                                             stride=strides[idx],
                                             padding=padding_sizes[idx]))
            self.conv.append(nn.BatchNorm1d(channel_size[1]))
            self.conv.append(nn.ReLU(inplace=True))
            self.conv.append(nn.MaxPool1d(kernel_size=2))
        self.conv = nn.Sequential(*self.conv)

        self.glob_pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.ModuleList()
        for idx, fc_size in enumerate(fc_sizes):
            self.fc.append(nn.Linear(fc_size[0], fc_size[1]))
            if idx < len(fc_sizes) - 1:
                self.fc.append(nn.ReLU(inplace=True))
        self.fc = nn.Sequential(*self.fc)

    def forward(self, x):
        x = self.conv(x)
        x = self.glob_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def save_model(self, save_dir, model_name="heart_rate_estimator.pth"):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(save_dir, model_name)
        torch.save(self.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, load_path):
        self.load_state_dict(torch.load(load_path))
        print(f"Model loaded from {load_path}")
    