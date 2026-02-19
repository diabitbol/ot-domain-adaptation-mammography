#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Installation des packages et récupération des données

import sys

get_ipython().system('mc mirror s3/hlegarzic/stat_app/ /home/onyxia/data/MINI-DDSM/')

get_ipython().system('{sys.executable} -m pip install --upgrade pip')
get_ipython().system('{sys.executable} -m pip install matplotlib tqdm seaborn scikit-learn')


# In[2]:


# Imports
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

# Vérification GPU
print(f"GPU disponible : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU détecté : {torch.cuda.get_device_name(0)}")
else:
    print("Pas de GPU, utilisation du CPU")


# In[3]:


# Restructuration des données
import os
import shutil

def restructure_normale_data():
    source = "/home/onyxia/data/MINI-DDSM/MINI-DDSM-Complete-JPEG-8/Normal"
    destination = "/home/onyxia/data/MINI-DDSM/MINI-DDSM-Complete-JPEG-82/normal"
    os.makedirs(destination, exist_ok=True)

    for filename in os.listdir(source):
        source_filename = os.path.join(source, filename)
        if not os.path.isdir(source_filename):
            continue

        chemin_MLO_right = os.path.join(source_filename, "A_" + filename + "_1.RIGHT_MLO.jpg")
        chemin_MLO_left = os.path.join(source_filename, "A_" + filename + "_1.LEFT_MLO.jpg")

        if os.path.exists(chemin_MLO_right):
            shutil.copy2(chemin_MLO_right, destination)
        if os.path.exists(chemin_MLO_left):
            shutil.copy2(chemin_MLO_left, destination)

    print("✓ Restructuration Normal terminée")


def restructuration_cancer():
    source = "/home/onyxia/data/MINI-DDSM/MINI-DDSM-Complete-JPEG-8/Cancer"
    destination1 = "/home/onyxia/data/MINI-DDSM/MINI-DDSM-Complete-JPEG-82/cancer_benign"
    destination2 = "/home/onyxia/data/MINI-DDSM/MINI-DDSM-Complete-JPEG-82/normal"
    os.makedirs(destination1, exist_ok=True)
    os.makedirs(destination2, exist_ok=True)

    for filename in os.listdir(source):
        source_filename = os.path.join(source, filename)
        if not os.path.isdir(source_filename):
            continue

        # Identification de la lettre correcte
        ok = False
        i = 0
        l = ["A_", "B_", "C_", "D_"]

        while not ok and i < 4:
            chemin_MLO_right = os.path.join(source_filename, l[i] + filename + "_1.RIGHT_MLO.jpg")
            if os.path.exists(chemin_MLO_right):
                ok = True
            else:
                i += 1

        if i == 4:
            print(f"Erreur1 avec le fichier {filename}")
            continue

        # Identification cancer gauche ou droit
        orientation = None
        if os.path.exists(os.path.join(source_filename, l[i] + filename + "_1.RIGHT_MLO.OVERLAY")):
            orientation = "RIGHT"
        elif os.path.exists(os.path.join(source_filename, l[i] + filename + "_1.LEFT_MLO.OVERLAY")):
            orientation = "LEFT"
        else:
            print(f"Cancer uniquement visible sur le point de vue CC : {filename}")

        # Copie des images
        if orientation == "RIGHT":
            shutil.copy2(chemin_MLO_right, destination1)
            chemin_MLO_left = os.path.join(source_filename, l[i] + filename + "_1.LEFT_MLO.jpg")
            if os.path.exists(chemin_MLO_left):
                shutil.copy2(chemin_MLO_left, destination2)
        elif orientation == "LEFT":
            chemin_MLO_left = os.path.join(source_filename, l[i] + filename + "_1.LEFT_MLO.jpg")
            if os.path.exists(chemin_MLO_left):
                shutil.copy2(chemin_MLO_left, destination1)
            if os.path.exists(chemin_MLO_right):
                shutil.copy2(chemin_MLO_right, destination2)

    print("Restructuration Cancer terminée")


def restructuration_benign():
    source = "/home/onyxia/data/MINI-DDSM/MINI-DDSM-Complete-JPEG-8/Benign"
    destination1 = "/home/onyxia/data/MINI-DDSM/MINI-DDSM-Complete-JPEG-82/cancer_benign"
    destination2 = "/home/onyxia/data/MINI-DDSM/MINI-DDSM-Complete-JPEG-82/normal"
    os.makedirs(destination1, exist_ok=True)
    os.makedirs(destination2, exist_ok=True)

    for filename in os.listdir(source):
        source_filename = os.path.join(source, filename)
        if not os.path.isdir(source_filename):
            continue

        # Identification de la lettre correcte
        ok = False
        i = 0
        l = ["A_", "B_", "C_", "D_"]

        while not ok and i < 4:
            chemin_MLO_right = os.path.join(source_filename, l[i] + filename + "_1.RIGHT_MLO.jpg")
            if os.path.exists(chemin_MLO_right):
                ok = True
            else:
                i += 1

        if i == 4:
            print(f"Erreur1 avec le fichier {filename}")
            continue

        # Identification benign gauche ou droit
        orientation = None
        if os.path.exists(os.path.join(source_filename, l[i] + filename + "_1.RIGHT_MLO.OVERLAY")):
            orientation = "RIGHT"
        elif os.path.exists(os.path.join(source_filename, l[i] + filename + "_1.LEFT_MLO.OVERLAY")):
            orientation = "LEFT"
        else:
            print(f"Benign uniquement visible sur le point de vue CC : {filename}")

        # Copie des images
        if orientation == "RIGHT":
            shutil.copy2(chemin_MLO_right, destination1)
            chemin_MLO_left = os.path.join(source_filename, l[i] + filename + "_1.LEFT_MLO.jpg")
            if os.path.exists(chemin_MLO_left):
                shutil.copy2(chemin_MLO_left, destination2)
        elif orientation == "LEFT":
            chemin_MLO_left = os.path.join(source_filename, l[i] + filename + "_1.LEFT_MLO.jpg")
            if os.path.exists(chemin_MLO_left):
                shutil.copy2(chemin_MLO_left, destination1)
            if os.path.exists(chemin_MLO_right):
                shutil.copy2(chemin_MLO_right, destination2)

    print("Restructuration Benign terminée")


# Exécution de la restructuration
print("Début de la restructuration des données...")
print("=" * 60)

restructure_normale_data()
restructuration_cancer()
restructuration_benign()

print("=" * 60)
print("Restructuration complète terminée !")

# Vérification des résultats
print("\n Vérification des données restructurées :")
base_path = "/home/onyxia/data/MINI-DDSM/MINI-DDSM-Complete-JPEG-82"
for classe in ["normal", "cancer_benign"]:
    chemin = os.path.join(base_path, classe)
    if os.path.exists(chemin):
        nb_images = len([f for f in os.listdir(chemin) if f.endswith('.jpg')])
        print(f"  - {classe}: {nb_images} images")
    else:
        print(f"  - {classe}: dossier non trouvé")


# In[6]:


# Génération des données équilibrées
# Transformation des données
transform = transforms.Compose([
    transforms.Resize((700, 700)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_set = ImageFolder(root="/home/onyxia/data/MINI-DDSM/MINI-DDSM-Complete-JPEG-8", transform=transform)
label = np.array(data_set.targets)

print(f"Nombre total d'images : {len(label)}")
print(f"Classes détectées : {data_set.classes}")
print(f"Distribution des classes : {Counter(label)}")

# Split stratifié des données
numero_random = 42
train_size = 0.7
val_size = 0.15
test_size = 0.15

sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=numero_random)
for train_val_index, test_index in sss1.split(np.zeros(len(label)), label):
    train_val_indices = train_val_index
    test_indices = test_index

train_val_labels = label[train_val_indices]
new_val_size = val_size / (train_size + val_size)
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=new_val_size, random_state=numero_random)
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

train_set = Subset(data_set, train_indices)
val_set = Subset(data_set, val_indices)
test_set = Subset(data_set, test_indices)

bs = 32
train_loader = DataLoader(train_set, batch_size=bs, sampler=sampler)
val_loader = DataLoader(val_set, batch_size=bs, shuffle=False)
test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)


# In[7]:


# Entraînement du modèle
# Création du modèle et configuration GPU
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_class = len(data_set.classes)
model.fc = torch.nn.Linear(model.fc.in_features, num_class)

# Configuration GPU
device = torch.device("cuda:0")
print(f"Device utilisé : {device}")
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.1, patience=3, threshold=0.0001
)

print(f"Modèle configuré avec {num_class} classes")

# Fonctions d'entraînement et validation
def train_model(model, train_loader, val_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for inputs, labels in tqdm(train_loader, desc="Entraînement"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    corrects = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = corrects.double() / len(val_loader.dataset)

    return epoch_loss, epoch_acc

# Boucle d'entraînement avec early stopping
num_epochs = 10  
best_acc = 0.0
patience_limite = 5
patience_compteur = 0

for epoch in range(num_epochs):
    print(f"\n{'='*50}")
    print(f"Époque {epoch+1}/{num_epochs}")
    print(f"{'='*50}")

    # Entraînement
    train_loss = train_model(model, train_loader, val_loader, criterion, optimizer)
    print(f"Loss d'entraînement: {train_loss:.4f}")

    # Validation
    val_loss, val_acc = validate_model(model, val_loader, criterion, device)
    print(f"Loss de validation: {val_loss:.4f} | Accuracy: {val_acc:.4f}")

    # Scheduler
    scheduler.step(val_acc)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Learning rate: {current_lr}")

    # Sauvegarde du meilleur modèle
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "/home/onyxia/best_resnet18_model.pth")
        # Sauvegarder sur S3
        get_ipython().system('mc cp /home/onyxia/best_resnet18_model.pth s3/hlegarzic/stat_app/')
        print(f"Nouveau meilleur modèle sauvegardé (acc: {best_acc:.4f})")
        patience_compteur = 0
    else:
        patience_compteur += 1
        print(f"Patience: {patience_compteur}/{patience_limite}")

        if patience_compteur >= patience_limite:
            print("\n Early stopping déclenché")
            break

print(f"\n{'='*50}")
print(f"Entraînement terminé. Meilleure accuracy: {best_acc:.4f}")
print(f"{'='*50}")


# In[ ]:


# Évaluation sur le test set
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_predictions.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

predictions_np = np.array(all_predictions)
labels_np = np.array(all_labels)

accuracy = np.mean(predictions_np == labels_np)
print(f'Test_Accuracy : {accuracy*100:.4f}%')

# Matrice de confusion
class_names = ['cancer', 'benign', 'normal']
cm = confusion_matrix(labels_np, predictions_np)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('/home/onyxia/confusion_matrix.png')
plt.show()

# Sauvegarder la matrice sur S3
get_ipython().system('mc cp /home/onyxia/confusion_matrix.png s3/hlegarzic/stat_app/')

# Rapport de classification
report = classification_report(labels_np, predictions_np, target_names=class_names)
print("\n" + "="*50)
print("RAPPORT DE CLASSIFICATION")
print("="*50)
print(report)

with open('/home/onyxia/classification_report.txt', 'w') as f:
    f.write("RAPPORT DE CLASSIFICATION\n")
    f.write("="*50 + "\n")
    f.write(report)

get_ipython().system('mc cp /home/onyxia/classification_report.txt s3/hlegarzic/stat_app/')
print("Rapport sauvegardé sur S3")

# Calcul de la courbe ROC et de l'AUC
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

num_classes = len(class_names)

if num_classes == 2:
    # Cas binaire : courbe ROC simple
    # Probabilités pour la classe positive (classe 0 ou 1 selon votre choix)
    probabilities_positive_class = probabilities_np[:, 1]

    fpr, tpr, thresholds = roc_curve(labels_np, probabilities_positive_class, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aléatoire')
    plt.xlabel('False Positive Rate (1 - Spécificité)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensibilité)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig('/home/onyxia/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

    get_ipython().system('mc cp /home/onyxia/roc_curve.png s3/hlegarzic/stat_app/')
    print("✓ Courbe ROC sauvegardée sur S3")

else:
    # Cas multi-classe : courbe ROC pour chaque classe (One-vs-Rest)
    labels_binarized = label_binarize(labels_np, classes=range(num_classes))

    plt.figure(figsize=(10, 8))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_binarized[:, i], probabilities_np[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aléatoire')
    plt.xlabel('False Positive Rate (1 - Spécificité)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensibilité)', fontsize=12)
    plt.title('Courbes ROC Multi-classes (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig('/home/onyxia/roc_curve_multiclass.png', dpi=300, bbox_inches='tight')
    plt.show()

    get_ipython().system('mc cp /home/onyxia/roc_curve_multiclass.png s3/hlegarzic/stat_app/')
    print("✓ Courbes ROC multi-classes sauvegardées sur S3")

# Visualisation des erreurs de classification
misclassified_indices = np.where(predictions_np != labels_np)[0]
print(f"\nNombre d'erreurs de classification : {len(misclassified_indices)}")

if len(misclassified_indices) > 0:
    # Afficher les 9 premières erreurs
    num_display = min(9, len(misclassified_indices))

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()

    for idx, ax in enumerate(axes):
        if idx < num_display:
            # Récupérer l'indice global dans le test_set
            error_idx = misclassified_indices[idx]

            # Charger l'image depuis le test_set
            image, true_label = test_set[error_idx]
            predicted_label = predictions_np[error_idx]

            # Dénormaliser l'image pour l'affichage
            image_display = image.permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_display = std * image_display + mean
            image_display = np.clip(image_display, 0, 1)

            ax.imshow(image_display)
            ax.set_title(f'Vrai: {class_names[true_label]}\nPrédit: {class_names[predicted_label]}', 
                        fontsize=10, color='red')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('/home/onyxia/misclassified_examples.png', dpi=300, bbox_inches='tight')
    plt.show()

    get_ipython().system('mc cp /home/onyxia/misclassified_examples.png s3/hlegarzic/stat_app/')
    print("Exemples d'erreurs sauvegardés sur S3")
else:
    print("Aucune erreur de classification")

