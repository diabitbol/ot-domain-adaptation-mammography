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
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
from torch.utils.data import WeightedRandomSampler
from collections import Counter

def evaluate_model(
    path,
    test_loader
):
    model.load_state_dict(torch.load(path))
    model.eval()
    all_predictions = []
    all_labels= []
    all_probabilities = []
    with torch.no_grad():
        for inputs, labels in tqdm(choose_data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            _,preds=torch.max(outputs,1)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            positive_class_probs = probabilities[:, 0].cpu().numpy()
            all_probabilities.extend(positive_class_probs)
    predictions_np=np.array(all_predictions)
    labels_np=np.array(all_labels)
    probabilities_np=np.array(all_probabilities)
    accuracy = np.mean(predictions_np == labels_np)
    return predictions_np, labels_np, probabilities_np, accuracy


def confusion_matrix(
    class_names,
    labels,
    predictions
):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')   
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


def roc_auc(
    labels_np, 
    probabilities_np
):
    fpr, tpr, thresholds = roc_curve(labels_np, probabilities_np, pos_label=0)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(4,4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate -Spécificity')
    plt.ylabel('True Positive Rate - Sensitivity')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def evaluation_report(
    labels_np, 
    predictions_np, 
    target_names
):
    report = classification_report(labels_np, predictions_np, target_names=class_names)
    print("Classification report:")
    print(report)