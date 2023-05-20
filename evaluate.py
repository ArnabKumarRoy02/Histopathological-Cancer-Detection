import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import cv2

# Import PyTorch
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision
import torch.optim as optim

# Import useful sklearn functions
import sklearn
from sklearn.metrics import roc_auc_score, accuracy_score
from PIL import Image

# Validate the model
model.eval()
for data, target in valid_loader:
    # Mode tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda().float()

        # Forward pass: compute predicted outputs by passing inputs to the model
        target = target.view(-1, 1)
        output = model(data)
        
        # Calculate the batch loss
        loss = criterion(output, target)
        
        # Update average validation loss 
        valid_loss += loss.item()*data.size(0)
        
        # Output = output.topk()
        y_actual = target.data.cpu().numpy()
        y_pred = output[:,-1].detach().cpu().numpy()
        val_auc.append(roc_auc_score(y_actual, y_pred))        
