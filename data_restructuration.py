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


def restructure_normale_data(
    source,
    destination
):
    for filename in os.listdir(source):
        source_filename= source + "\\" + filename
        chemin_MLO_right=os.path.join(source_filename,"A_"+filename+"_1.RIGHT_MLO.jpg")
        chemin_MLO_left=os.path.join(source_filename,"A_"+filename+"_1.LEFT_MLO.jpg")
        if os.path.exists(chemin_MLO_right):
            shutil.copy2(chemin_MLO_right,destination)
        if os.path.exists(chemin_MLO_left):
            shutil.copy2(chemin_MLO_left,destination)
    print("Restructuration terminée")


def restructuration_cancer(
    source,
    destination1,
    destination2
):
    c=0
    for filename in os.listdir(source):
        #identification de la lettre correcte
        ok=False
        i=0
        l=["A_","B_","C_","D_"]
        source_filename= source + "\\" + filename
        while ok == False:
            chemin_MLO_right=os.path.join(source_filename,l[i]+filename+"_1.RIGHT_MLO.jpg")
            if os.path.exists(chemin_MLO_right):
                ok = True
            else:
                i+=1
        if i==4:
            print("erreur1 avec le fichier "+filename)

        #identification cancer gauche ou droit
        orientation=None
        if os.path.exists(os.path.join(source_filename,l[i]+filename+"_1.RIGHT_MLO.OVERLAY")):
             orientation="RIGHT"
        elif os.path.exists(os.path.join(source_filename,l[i]+filename+"_1.LEFT_MLO.OVERLAY")):
            orientation="LEFT"
        else:
             print("cancer uniquement visible sur le point de vue CC "+filename)
        #copie des images
        if orientation=="RIGHT":
            shutil.copy2(chemin_MLO_right,destination1)
            shutil.copy2(os.path.join(source_filename,l[i]+filename+"_1.LEFT_MLO.jpg"),destination2)
        if orientation=="LEFT":
            shutil.copy2(os.path.join(source_filename,l[i]+filename+"_1.LEFT_MLO.jpg"),destination1)
            shutil.copy2(chemin_MLO_right,destination2)


def restructuration_benign(
    source,
    destination1,
    destination2
):
    for filename in os.listdir(source):
        #identification de la lettre correcte
        ok=False
        i=0
        l=["A_","B_","C_","D_"]
        source_filename= source + "\\" + filename
        while ok == False:
            chemin_MLO_right=os.path.join(source_filename,l[i]+filename+"_1.RIGHT_MLO.jpg")
            if os.path.exists(chemin_MLO_right):
                ok = True
            else:
                i+=1
        if i==4:
            print("erreur1 avec le fichier "+filename)

        #identification cancer gauche ou droit
        orientation=None
        if os.path.exists(os.path.join(source_filename,l[i]+filename+"_1.RIGHT_MLO.OVERLAY")):
             orientation="RIGHT"
        elif os.path.exists(os.path.join(source_filename,l[i]+filename+"_1.LEFT_MLO.OVERLAY")):
            orientation="LEFT"
        else:
             print("cancer uniquement visible sur le point de vue CC "+filename)
        #copie des images
        if orientation=="RIGHT":
            shutil.copy2(chemin_MLO_right,destination1)
            shutil.copy2(os.path.join(source_filename,l[i]+filename+"_1.LEFT_MLO.jpg"),destination2)
        if orientation=="LEFT":
            shutil.copy2(os.path.join(source_filename,l[i]+filename+"_1.LEFT_MLO.jpg"),destination1)
            shutil.copy2(chemin_MLO_right,destination2)


def get_transformed_data(
    image_length, 
    image_heigth, 
    root
):
    transform = transforms.Compose([
        transforms.Resize((image_length,image_heigth)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data_set=ImageFolder(root=root, transform=transform)
    return dataset


def get_data_loaders(
    dataset,
    numero_random,
    batch_size,
    train_size,
    val_size,
    test_size
):
    labels = np.array(dataset.targets)
    sss1=StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=numero_random)
    for train_val_index, test_index in sss1.split(np.zeros(len(labels)), labels):
        train_val_indices=train_val_index
        test_indices=test_index
    train_val_labels=labels[train_val_indices]
    new_val_size=val_size/(train_size+val_size)
    ss2=StratifiedShuffleSplit(n_splits=1, test_size=new_val_size, random_state=numero_random)
    for train_index, val_index in ss2.split(np.zeros(len(train_val_labels)), train_val_labels):
        train_indices=train_val_indices[train_index]
        val_indices=train_val_indices[val_index]
    train_labels=labels[train_indices]
    class_counts=Counter(train_labels)
    print(f"Comptes des classes dans le jeu d'entraînement: {class_counts} \n")
    
    num_labels=len(train_labels)
    weight_for_class={cl : num_labels/count for cl, count in class_counts.items()}
    sample_weights=[weight_for_class[l] for l in train_labels]
    sample_weights_tensor=torch.DoubleTensor(sample_weights)
    print(f"poids calculés par classe :{weight_for_class} \n")
    
    sampler = WeightedRandomSampler(weights=sample_weights_tensor, num_samples=len(sample_weights_tensor), replacement=True)
    distribution_sampler=Counter(train_labels[list(sampler)])
    print(f"Distribution des classes dans le sampler : {distribution_sampler} \n")
    
    train_set=Subset(data_set, train_indices)
    val_set=Subset(data_set, val_indices)
    test_set=Subset(data_set, test_indices)
    print(f"Train set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    print(f"Test set size: {len(test_set)}")
    
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader