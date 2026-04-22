import torch
from torchvision.models import ResNet18_Weights
import torchvision.models as models
import numpy as np
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split   
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import gc


def data_generation(
    resolution,
    batch_size,
    contrast,
    brightness,
    degrees,
    normalisation_choice,
    lower_lim,
    random_number,
    train_size,
    val_size,
    test_size,
    data_path
):

    #définition de la fonction de pour normaliser les images par rapport à elles mêmes
    def robust_standardize1(x):
        mask = x > lower_lim  #On crée un masque pour ignorer le fond (souvent proche de 0 ou < 0.05)
        if mask.any():
            mean = x[mask].mean()
            std = x[mask].std()
            x[mask] = (x[mask] - mean) / (std + 1e-6) 
            x[mask] = torch.clamp(x[mask], -3, 3) #on ramène les valeurs extrêmes du sein à l'intervalle -3 3
        x[~mask] = -4.0 # on fixe le reste des valeurs correspondantes au support noir à -4
        return x


    #definition de la fonction pour normaliser les images par rapport au dataset MiniDDSM et calcul de mean et std error du dataset
    def calcul_dataset_mean_std():

        def recuperation_pixel(loader_source, n_batches=20):
            # On récupère quelques données pour avoir une distribution stable
            source_pixels = []
            
            with torch.no_grad():
                # Extraction du domaine Source
                for i, (images, _) in tqdm(enumerate(loader_source)):
                    if i >= n_batches: break
                    # On prend un seul canal (gris) et on aplatit
                    source_pixels.extend(images[:, 0, :, :].flatten().numpy())
            return(source_pixels)

        def robust_mean_variance(x):
            # 1. On calcule la std error et la mean sur le dataset
            mask = (x > 0.05) & (x<0.97) 
            if mask.any():
                mean = x[mask].mean()
                std = x[mask].std()
            return (mean,std)

        train_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor()])

        train_dataset_full = ImageFolder(root=data_path, transform=train_transform)
        train_loader1 = DataLoader(train_dataset_full, batch_size=batch_size)

        (source_pixel)=recuperation_pixel(train_loader1)
        source_pixel=np.array(source_pixel)

        (mu,sigma)=robust_mean_variance(source_pixel)
        return(mu,sigma)

    def robust_standardize2(x):
        mask = x > lower_lim
        if mask.any():
            mean = mu
            std = sigma
            x[mask] = (x[mask] - mean) / (std + 1e-6)    
            x[mask] = torch.clamp(x[mask], -3, 3)
        x[~mask] = -4.0 
        return x

    if normalisation_choice==0:
        def identity(x):
            return x
        fonction = identity
    elif normalisation_choice==1:
        fonction=robust_standardize1
    else:
        (mu,sigma)=calcul_dataset_mean_std()
        fonction=robust_standardize2
        
    # Transformation des données ATTENTION A L'ORDRE DES TRANSFORMS!!!
    train_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ColorJitter(brightness=brightness, contrast=contrast), # Change légèrement la luminosité
        transforms.RandomRotation(degrees=degrees),       # Rotation légère (en mettant du noir 0 sur le côte)
        transforms.RandomVerticalFlip(p=0.5),        # Les mammo peuvent être inversées
        transforms.ToTensor(),
        transforms.Lambda(fonction), 
        # --- AJOUTS POUR LUTTER CONTRE L'OVERFITTING ---
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Lambda(fonction), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset_full = ImageFolder(root=data_path, transform=train_transform)
    val_dataset_full = ImageFolder(root=data_path, transform=transform)

    label = np.array(val_dataset_full.targets)
    print(f"Nombre total d'images : {len(label)}")
    print(f"Classes détectées : {val_dataset_full.classes}")
    print(f"Distribution des classes : {Counter(label)}")
    print(f"accuracy random val : {1-(label.sum()/len(label))}")

    #on équilibre la proportion de labels 0 et 1 dans le dataset d'entraînement
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_number)
    for train_val_index, test_index in sss1.split(np.zeros(len(label)), label):
        train_val_indices = train_val_index
        test_indices = test_index

    train_val_labels = label[train_val_indices]
    new_val_size = val_size / (train_size + val_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=new_val_size, random_state=random_number)
    for train_index, val_index in sss2.split(np.zeros(len(train_val_labels)), train_val_labels):
        train_indices = train_val_indices[train_index]
        val_indices = train_val_indices[val_index]

    print(f"Train set: {len(train_indices)}")
    print(f"Validation set: {len(val_indices)}")
    print(f"Test set: {len(test_indices)}")

    # Weighted Sampler pour équilibrer les classes
    train_labels = label[train_indices]
    class_counts = Counter(train_labels)
    print(f"Distribution dans train : {class_counts}")

    num_labels = len(train_labels)
    weight_for_class = {cl: num_labels / count for cl, count in class_counts.items()}
    sample_weights = [weight_for_class[l] for l in train_labels]
    sample_weights_tensor = torch.DoubleTensor(sample_weights)

    print(f"Poids par classe : {weight_for_class}")

    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor, 
        num_samples=len(sample_weights_tensor), 
        replacement=True
    )

    train_set = Subset(train_dataset_full, train_indices)
    val_set = Subset(val_dataset_full, val_indices)
    test_set = Subset(val_dataset_full, test_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4, pin_memory=True, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return(train_loader,val_loader,test_loader)


def configurate_model(
    device,
    num_class,
    dropout,
    factor,
    weight_decay,
    patience,
    threshold
):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout), # Désactive 50% des neurones aléatoirement à chaque itération
        nn.Linear(model.fc.in_features, num_class)
    )
    device = torch.device(device)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, threshold=threshold)
    return model, criterion, optimizer, scheduler