import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tqdm import tqdm
import gc


def evaluation(model, test_loader, device):
    """
    Fait passer les données de test dans le modèle et retourne les arrays nécessaires aux métriques.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            logits = outputs.squeeze(1)
            
            probabilities = torch.sigmoid(logits)
            
            preds = (probabilities >= 0.5).long()
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
    predictions_np = np.array(all_predictions)
    labels_np = np.array(all_labels)
    probabilities_np = np.array(all_probabilities)

    accuracy = np.mean(predictions_np == labels_np)
    print(f'Test_Accuracy : {accuracy*100:.4f}%')

    # Nettoyage de la VRAM
    gc.collect()
    torch.cuda.empty_cache()

    return labels_np, predictions_np, probabilities_np


def matrice_confusion(labels_np, predictions_np, class_names):
    """
    Affiche la matrice de confusion.
    """
    cm = confusion_matrix(labels_np, predictions_np)

    plt.figure(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


def rapport_classification(labels_np, predictions_np, class_names):
    """
    Affiche le rapport détaillé des métriques (Precision, Recall, F1-Score).
    """
    report = classification_report(labels_np, predictions_np, target_names=class_names)
    print("\n" + "="*50)
    print("RAPPORT DE CLASSIFICATION")
    print("="*50)
    print(report)


def courbe_roc_auc(labels_np, probabilities_np):
    """
    Affiche la courbe ROC et calcule l'AUC.
    """
    fpr, tpr, thresholds = roc_curve(labels_np, probabilities_np, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 3))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aléatoire')
    plt.xlabel('False Positive Rate (1 - Spécificité)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensibilité)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()