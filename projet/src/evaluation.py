#2-entrainement_miniddsm
# on importe les modules
import torch
from torchvision.models import ResNet18_Weights
import torchvision.models as models
import torch.nn as nn


# Création du modèle et configuration GPU
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Dropout(p=dropout),  # Désactive 50% des neurones aléatoirement à chaque itération
    nn.Linear(model.fc.in_features, 2))

# Configuration GPU
device = torch.device("cuda:0")
model = model.to(device)
criterion = loss_function
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # ajout d'un weightdecay pour pénaliser l'overfitting
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, threshold=threshold)
