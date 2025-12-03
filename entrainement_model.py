import torch
from torchvision.models import ResNet18_Weights
import torchvision.models as models

#model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model=models.resnet18(weights=None)
model.fc = torch.nn.Linear(512,3)
model.load_state_dict(torch.load("best_resnet18_model.pth"))


#définition du nombre de classes, ici 3 : cancer, bénin, normal
num_class=3
model.fc = torch.nn.Linear(model.fc.in_features, num_class)

from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split   
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

#standardisation de la transformation des images pour coller au modèle restnet18
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

data_set=ImageFolder(root=".\\MINI-DDSM-Complete-JPEG-81", transform=transform)

print("imagefolder ok")
total_size=len(data_set)
train_size=int(0.7*total_size)
val_size=int(0.15*total_size)
test_size=total_size-train_size-val_size

print(total_size, train_size, val_size, test_size)

train_set, val_set, test_set = random_split(data_set, [train_size, val_size, test_size])
print("random_split ok")

bs=64
train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
print("train_loader ok")
val_loader = DataLoader(val_set, batch_size=bs, shuffle=False)
print("val_loader ok")
test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)
print("test_loader ok")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001) 

import torch.optim as optim
from tqdm import tqdm


num_epochs=5
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


def train_model(model, train_loader, val_loader, criterion, optimizer):
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

def validate_model(model, val_loader, criterion, device):
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
    


def entrain_and_validate():
    best_acc=0.9056
    for epoch in range(num_epochs):
        print(f"\n--- Époque {epoch+1}/{num_epochs} ---")

        train_loss=train_model(model, train_loader, val_loader, criterion, optimizer)
        print(f"Loss d'entraînement: {train_loss:.4f}")

        val_loss, val_acc=validate_model(model, val_loader, criterion, device)
        print(f"Loss de validation: {val_loss:.4f} | Accuracy de validation: {val_acc:.4f}")

        if best_acc < val_acc:
            best_acc=val_acc
            torch.save(model.state_dict(), "best_resnet18_model.pth")
            print("Modèle sauvegardé.")

    print("\nEntraînement terminé.")

#entrain_and_validate()

#meilleur accuracy sur validation :

def test_model():
    model.load_state_dict(torch.load("best_resnet18_model.pth"))
    test_loss, test_acc=validate_model(model, test_loader, criterion, device)
    print(f"Loss de test: {test_loss:.4f} | Accuracy de test: {test_acc:.4f}")

test_model()