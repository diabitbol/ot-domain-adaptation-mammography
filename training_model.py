import torch
from torchvision.models import ResNet18_Weights
import torchvision.models as models
import numpy as np
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
from torch.utils.data import WeightedRandomSampler
from collections import Counter
from datetime import datetime


def train_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer
):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss=criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss= running_loss / len(train_loader.dataset)
    return(epoch_loss)


def validate_model(
    model, 
    val_loader, 
    criterion, 
    device
):
    model.eval()
    running_loss = 0.0
    corrects = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _,preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = corrects.double() / len(val_loader.dataset)
        return(epoch_loss, epoch_acc)


def train_and_validate(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer,
    device,
    num_epochs,
    num_class,
    scheduler
):
    model.to(device)
    global best_acc
    patience_limite=5
    patience_compteur=0
    for epoch in range(num_epochs):
        print(patience_compteur)
        print(f"\n--- Époque {epoch+1}/{num_epochs} ---")

        train_loss=train_model(model, train_loader, val_loader, criterion, optimizer)
        print(f"Loss d'entraînement: {train_loss:.4f}")

        val_loss, val_acc=validate_model(model, val_loader, criterion, device)
        print(f"Loss de validation: {val_loss:.4f} | Accuracy de validation: {val_acc:.4f}")

        scheduler.step(val_acc)

        current_lr=optimizer.param_groups[0]['lr']
        print(f"Taux d'apprentissage actuel: {current_lr}")

        if best_acc < val_acc:
            best_acc=val_acc
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save(model.state_dict(), f"best_resnet18_model_ordonnanceur_{timestamp}.pth")
            patience_compteur=0
            print("Modèle sauvegardé.")
        else:
            patience_compteur+=1
            if patience_compteur >= patience_limite:
                print("Arrêt anticipé déclenché.")
                break
    print("\nEntraînement terminé.")

    