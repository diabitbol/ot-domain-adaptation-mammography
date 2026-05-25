import torch


# Calcul de l'information de Fischer
def compute_fisher(model, dataloader, device):
    fisher = {}
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    
    # Initialiser Fisher à zéro
    for n, p in params.items():
        fisher[n] = torch.zeros_like(p.data)

    model.eval()
    for images, labels in dataloader:
        images = images.to(device)
        model.zero_grad()
        
        outputs = model(images)
        # On calcule la log-vraisemblance (BCE)
        prob = torch.sigmoid(outputs)
        # On utilise les prédictions du modèle pour calculer Fisher (selon l'article)
        log_likelihood = torch.nn.functional.binary_cross_entropy_with_logits(outputs, prob, 
                                                                              reduction='sum')
        
        log_likelihood.backward()
        for n, p in params.items():
            if p.grad is not None:
                fisher[n] += (p.grad ** 2) / len(dataloader.dataset)
                
    return fisher


# Juste avant de passer à VinDr :
def poids_optimaux(model):
    opt_weights = {}
    for name, param in model.named_parameters():
        # .clone() est crucial pour avoir une copie indépendante
        # .detach() assure que ces poids ne seront pas modifiés par le gradient
        opt_weights[name] = param.data.clone().detach()
    return (opt_weights)

