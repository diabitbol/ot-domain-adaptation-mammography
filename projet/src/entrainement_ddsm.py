# on importe les modules
import torch
import numpy as np
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
from torch.utils.data import WeightedRandomSampler
from collections import Counter
from torch.cuda.amp import GradScaler, autocast
import os


# Génération des données équilibrées (standardisation robuste par image)


def generation_donnees(chemin_data, limite_basse, bs, resolution, choix_normalisation_image, brightness, contrast, degrees, test_size, val_size, train_size, numero_random):

    # définition de la fonction de pour normaliser les images par rapport à elles mêmes
    def robust_standardize1(x):
        mask = x > limite_basse         # On crée un masque pour ignorer le fond (souvent proche de 0 ou < 0.05)
        if mask.any():
            mean = x[mask].mean()
            std = x[mask].std()
            x[mask] = (x[mask] - mean) / (std + 1e-6) 
            x[mask] = torch.clamp(x[mask], -3, 3)  # on ramène les valeurs extrêmes du sein à
# l'intervalle -3 3
        x[~mask] = -4.0  # on fixe le reste des valeurs correspondantes au support noir à -4
        return x

    # definition de la fonction pour normaliser les images par rapport au dataset MiniDDSM 
    # et calcul de mean et std error du dataset
    def calcul_dataset_mean_std():

        def recuperation_pixel(loader_source, n_batches=20):
            # On récupère quelques données pour avoir une distribution stable
            source_pixels = []
            
            with torch.no_grad():
                # Extraction du domaine Source
                for i, (images, _) in tqdm(enumerate(loader_source)):
                    if i >= n_batches: 
                        break
                    # On prend un seul canal (gris) et on aplatit
                    source_pixels.extend(images[:, 0, :, :].flatten().numpy())
            return (source_pixels)

        def robust_mean_variance(x):
            # 1. On calcule la std error et la mean sur le dataset
            mask = (x > 0.05) & (x < 0.97) 
            if mask.any():
                mean = x[mask].mean()
                std = x[mask].std()
            return (mean, std)

        train_transform = transforms.Compose([transforms.Resize((resolution, resolution)), transforms.ToTensor()])

        train_dataset_full = ImageFolder(root=chemin_data, transform=train_transform)
        train_loader1 = DataLoader(train_dataset_full, batch_size=bs)

        (source_pixel) = recuperation_pixel(train_loader1)
        source_pixel = np.array(source_pixel)

        (mu, sigma) = robust_mean_variance(source_pixel)
        return (mu, sigma)

    def robust_standardize2(x):
        mask = x > limite_basse
        if mask.any():
            mean = mu
            std = sigma
            x[mask] = (x[mask] - mean) / (std + 1e-6)    
            x[mask] = torch.clamp(x[mask], -3, 3)
        x[~mask] = -4.0 
        return x

    if choix_normalisation_image == 0:
        def identity(x):
            return x
        fonction = identity
    elif choix_normalisation_image == 1:
        fonction = robust_standardize1
    else:
        (mu, sigma) = calcul_dataset_mean_std()
        fonction = robust_standardize2
        
    # Transformation des données ATTENTION A L'ORDRE DES TRANSFORMS!!!
    train_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ColorJitter(brightness=brightness, contrast=contrast),  # Change légèrement la luminosité
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

    train_dataset_full = ImageFolder(root=chemin_data, transform=train_transform)
    val_dataset_full = ImageFolder(root=chemin_data, transform=transform)

    label = np.array(val_dataset_full.targets)
    print(f"Nombre total d'images : {len(label)}")
    print(f"Classes détectées : {val_dataset_full.classes}")
    print(f"Distribution des classes : {Counter(label)}")
    print(f"accuracy random val : {1-(label.sum()/len(label))}")

    # on équilibre la proportion de labels 0 et 1 dans le dataset d'entraînement
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

    train_set = Subset(train_dataset_full, train_indices)
    val_set = Subset(val_dataset_full, val_indices)
    test_set = Subset(val_dataset_full, test_indices)

    train_loader = DataLoader(train_set, batch_size=bs, num_workers=4, pin_memory=True, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    return (train_loader, val_loader, test_loader)


if __name__ == "__main__":
    generation_donnes()


# Entraînement du modèle

def entrainement_model(base_dir, chemin_modele_local, chemin_modele_s3, patience, num_epochs, train_loader, val_loader, patience_limite, criterion, nom_modele_sauvegarde, optimizer, scheduler, device, model):

    # Fonctions d'entraînement et validation
    def train_model(model, train_loader, val_loader, criterion, optimizer):
        model.train()
        running_loss = 0.0
        scaler = GradScaler()  # Le scaler reste identique
        
        for inputs, labels in tqdm(train_loader, desc="Entraînement"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
        
            with autocast(): 
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # 4. On modifie la phase de rétropropagation
            scaler.scale(loss).backward()  # On "scale" la loss pour pas qu'elle soit trop petite
            scaler.step(optimizer)         # L'optimizer fait son pas
            scaler.update()                # On met à jour le scaler pour le prochain tour
            
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
    best_acc = 0.0  # initialisation accuracy
    patience_compteur = 0  # initialisation patience
    best_loss = 100  # initialisation Loss

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
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr}")
        
        # Sauvegarde du meilleur modèle Validation Loss
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(base_dir, "projet", "model") + nom_modele_sauvegarde + ".pth")
            # Sauvegarder sur S3
            cmd = f"mc mirror {chemin_modele_local} {chemin_modele_s3}"
            exit_code = os.system(cmd)
            if exit_code == 0:
                print("Synchronisation réussie !")
            else:
                print(f"Erreur lors de la synchronisation. Code : {exit_code}")
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


if __name__ == "__main__":
    entrainement_model()

