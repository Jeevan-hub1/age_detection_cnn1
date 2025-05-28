import torch
import torch.nn as nn
from torchvision.models import resnet18
import numpy as np
import cv2

class AgeDetectionResNet(nn.Module):
    def __init__(self):
        super(AgeDetectionResNet, self).__init__()
        self.model = resnet18(weights='IMAGENET1K_V1')
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)  # Single output for regression
        self.relu = nn.ReLU()  # Ensure non-negative output
    
    def forward(self, x):
        x = self.model(x)
        return self.relu(x)  # Enforce non-negative age

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        self.activations = output
    
    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]
    
    def generate(self, input_image, target=None):
        self.model.eval()
        output = self.model(input_image)
        
        if target is None:
            target = output  # Use predicted age for regression
        
        self.model.zero_grad()
        output.backward(gradient=target)
        
        gradients = self.gradients
        activations = self.activations
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)
        heatmap /= np.max(heatmap) + 1e-10
        return heatmap

def overlay_heatmap(heatmap, image, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + image
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img