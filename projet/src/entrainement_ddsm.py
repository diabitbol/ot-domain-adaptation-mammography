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
import cv2

# Génération des données équilibrées (standardisation robuste par image)


def generation_donnees(chemin_data, limite_basse=0., bs=32, resolution=700, choix_fonction=0, 
                       brightness=0, contrast=0, degrees=0, test_size=0.15, val_size=0.15, 
                       train_size=0.7, 
                       numero_random=42, normalize=None, mu_ddsm=None, sigma_ddsm=None, 
                       mu_vindr=None, sigma_vindr=None, value_quantile_target=None, 
                       value_quantile_source=None, clip_limit=2.0, tile_size=(8, 8)):

    # définition de la fonction de pour normaliser les images par rapport à elles mêmes
    def robust_standardize1(x):
        mask = x > limite_basse  # On crée un masque pour ignorer le fond (proche de 0 ou <0.05)
        if mask.any():
            mean = x[mask].mean()
            std = x[mask].std()
            x[mask] = (x[mask] - mean) / (std + 1e-6) 
            x[mask] = torch.clamp(x[mask], -3, 3)  # on ramène les valeurs extrêmes du sein à
# l'intervalle -3 3
        x[~mask] = -4.0  # on fixe le reste des valeurs correspondantes au support noir à -4
        return x

    # definition de la fonction pour normaliser les images par rapport au dataset MiniDDSM 
    def robust_standardize2(x):
        mask = x > limite_basse
        if mask.any():
            x[mask] = ((((x[mask] - mu_vindr) / (sigma_vindr + 1e-6))) *  sigma_ddsm)  + mu_ddsm
        return x

    def apply_ot_mapping(image_tensor):
        image_flatten = image_tensor.flatten()
        img_flat_np = image_flatten.numpy()
        img_flat_np = np.interp(
            img_flat_np, 
            value_quantile_target.numpy(), 
            value_quantile_source.numpy()
        )
        return (torch.from_numpy(img_flat_np).view_as(image_tensor).float())

    def apply_clahe(image_tensor, clip_limit=2.0, tile_size=(8, 8)):
        # 1. On extrait le tenseur en numpy
        img_np = image_tensor.numpy()
        
        # 2. Sécurité absolue : on force une matrice 2D parfaite (H, W)
        if img_np.ndim == 3:
            img_np = img_np[0]  # On prend le premier canal
            
        # On écrase les dimensions fantômes éventuelles (ex: [700, 700, 1] devient [700, 700])
        img_np = np.squeeze(img_np)
        
        # 3. Mise à l'échelle (0 à 255)
        img_scaled = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        
        # 4. LE CORRECTIF MAGIQUE : On force un bloc mémoire propre pour OpenCV
        img_scaled = np.ascontiguousarray(img_scaled)
        
        # 5. Création et application du filtre CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        img_clahe = clahe.apply(img_scaled)
    
        # 6. On reconvertit l'image en Float 32 bits (le format PyTorch)
        img_rescaled = img_clahe.astype(np.float32) / 255.0
        
        # 7. On repasse en Tenseur et on recrée la dimension du canal -> [1, 700, 700]
        tensor_clahe = torch.from_numpy(img_rescaled).unsqueeze(0)
        
        # 8. Sécurité ResNet : Si l'image d'origine avait 3 canaux, on duplique
        if image_tensor.shape[0] == 3:
            tensor_clahe = tensor_clahe.repeat(3, 1, 1)
        return tensor_clahe
        
    if choix_fonction == 0:
        def identity(x):
            return x
        fonction = identity
    elif choix_fonction == 1:
        fonction = robust_standardize1
    elif choix_fonction == 2:
        fonction = robust_standardize2
    elif choix_fonction == 3:
        fonction = apply_ot_mapping
    else:
        fonction = apply_clahe

    # Transformation des données ATTENTION A L'ORDRE DES TRANSFORMS!!!
    train_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ColorJitter(brightness=brightness, contrast=contrast), 
        transforms.RandomRotation(degrees=degrees),     
        transforms.RandomVerticalFlip(p=0.5),  # Les mammo peuvent être inversées
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
    
    transform_brut = transforms.Compose([transforms.Resize((resolution, resolution)),
                                         transforms.ToTensor(),
                                         transforms.Lambda(fonction)])

    train_dataset_full = ImageFolder(root=chemin_data, transform=train_transform)
    val_dataset_full = ImageFolder(root=chemin_data, transform=transform)
    val_dataset_brut = ImageFolder(root=chemin_data, transform=transform_brut)

    label = np.array(val_dataset_full.targets)
    print(f"Nombre total d'images : {len(label)}")
    print(f"Classes détectées : {val_dataset_full.classes}")
    print(f"Distribution des classes : {Counter(label)}")
    print(f"accuracy random val : {max(1-(label.sum()/len(label)), (label.sum()/len(label)))}")

    if test_size == 1 or test_size == 1.0:
        print("\n---> Mode Inférence activé : 100% des données assignées au Test Set.")
        test_indices = np.arange(len(label))
        test_set = Subset(val_dataset_full, test_indices)
        test_set_brut = Subset(val_dataset_brut, test_indices)
   
        # On ne crée que le loader de test
        test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=4, 
                                 pin_memory=True)
        test_loader_brut = DataLoader(test_set_brut, batch_size=bs, shuffle=False, num_workers=4, 
                                      pin_memory=True)
        
        if normalize == 1:
            # On retourne None pour le train et la val
            return (None, None, test_loader)
        else:
            return (None, None, test_loader_brut)
        
    else:
        # on équilibre la proportion de labels 0 et 1 dans le dataset d'entraînement
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=numero_random)
        for train_val_index, test_index in sss1.split(np.zeros(len(label)), label):
            train_val_indices = train_val_index
            test_indices = test_index

        train_val_labels = label[train_val_indices]
        new_val_size = val_size / (train_size + val_size)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=new_val_size, 
                                      random_state=numero_random)
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
        sample_weights = [weight_for_class[l1] for l1 in train_labels]
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

        train_loader = DataLoader(train_set, batch_size=bs, num_workers=4, 
                                  pin_memory=True, sampler=sampler)
        val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, 
                                num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, 
                                 num_workers=4, pin_memory=True)
        return (train_loader, val_loader, test_loader)


if __name__ == "__main__":
    generation_donnees()


# Entraînement du modèle

def entrainement_model(base_dir, chemin_modele_local, chemin_modele_s3, patience, num_epochs, 
                       train_loader, val_loader, patience_limite, criterion, nom_modele_sauvegarde, 
                       optimizer, scheduler, device, model, EWC=False, ewc_loss1=None,
                       fischer=None, opt_weights=None, lam=400):
    # Pour le calul de la loss
    def ewc_loss(current_loss, fisher=fischer, opt_weights=opt_weights, lam=lam, batch_idx=[0]):
        ewc_penalty = 0
        for n, p in model.named_parameters():
            if n in fisher:
                # Le "ressort" : importance * (poids_actuel - poids_DDSM)^2
                ewc_penalty += (fisher[n] * (p - opt_weights[n])**2).sum()
        penalite_finale = (lam / 2) * ewc_penalty
        # On affiche la pénalité de temps en temps (les 5 premiers batchs par ex)
        if batch_idx[0] < 5:
            print(f"\n[DEBUG] BCE Loss: {current_loss.item():.4f} | EWC Penalty: {penalite_finale.item():.6f}")
            batch_idx[0] += 1
        return current_loss + penalite_finale

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
                if EWC:
                    current_loss = criterion(outputs.squeeze(1), labels.float())
                    loss = ewc_loss(current_loss=current_loss)
                else:
                    loss = criterion(outputs.squeeze(1), labels.float())

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
                loss = criterion(outputs.squeeze(1), labels.float())

                running_loss += loss.item() * inputs.size(0)
                probs = torch.sigmoid(outputs.squeeze(1))
                preds = (probs >= 0.5).long()

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
            best_acc = val_acc
            torch.save(model.state_dict(), chemin_modele_local)
            # Sauvegarder sur S3
            cmd = f"mc cp {chemin_modele_local} {chemin_modele_s3}"
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

