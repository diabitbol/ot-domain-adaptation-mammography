import matplotlib.pyplot as plt
import torch
import numpy as np


def distribution_pixels(loader_source, loader_target, n_batches=5):
    source_pixels = []
    target_pixels = []
    with torch.no_grad():
        # Extraction du domaine Source
        for i, (images, _) in enumerate(loader_source):
            if i >= n_batches: 
                break
            # On prend un seul canal (gris) et on aplatit
            source_pixels.extend(images[:, 0, :, :].flatten().numpy())
            
        # Extraction du domaine Cible (VinDr)
        for i, (images, _) in enumerate(loader_target):
            if i >= n_batches: 
                break
            target_pixels.extend(images[:, 0, :, :].flatten().numpy())
    return (source_pixels, target_pixels)


if __name__ == "__main__":
    distribution_pixels()


def plot_distributions(source_pixels, target_pixels, max_samples=100000):
    """
    Affiche la distribution en sous-échantillonnant si les listes sont trop grandes.
    max_samples: Nombre maximum de pixels à utiliser pour tracer l'histogramme.
    """
    source_arr = np.array(source_pixels)
    target_arr = np.array(target_pixels)
    
    if len(source_arr) > max_samples:
        source_arr = np.random.choice(source_arr, size=max_samples, replace=False)
    
    if len(target_arr) > max_samples:
        target_arr = np.random.choice(target_arr, size=max_samples, replace=False)
    
    plt.figure(figsize=(7, 4))
    plt.hist(source_arr, bins=100, alpha=0.5, label='Source (DDSM)', color='blue', density=True)
    plt.hist(target_arr, bins=100, alpha=0.5, label='Target (VinDr)', color='red', density=True)
    
    plt.title("Comparaison des distributions d'intensité (Pixels)")
    plt.xlabel("Valeur du pixel")
    plt.ylabel("Densité")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    return ()


if __name__ == "__main__":
    plot_distributions()


def robust_mean_variance(x):
    x_arr = np.array(x)
    mask = (x_arr > 0.05) & (x_arr < 0.97) 
    if mask.any():
        mean = x_arr[mask].mean()
        std = x_arr[mask].std() 
        return mean, std


# matching quantile : calcul des quantiles et représentations

def calcul_quantile(loader_source, loader_target, n_batches=5):
    # récupération des datasets Vindr et miniDDSM transformé 
    (source_pixel, target_pixel) = distribution_pixels(loader_source, loader_target, n_batches=5)
    source_pixel = np.array(source_pixel)
    target_pixel = np.array(target_pixel)
    liste_quantile_np = np.linspace(0, 1, 1000)

    print("Calcul des quantiles Source...")
    value_quantile_source = torch.from_numpy(
        np.quantile(source_pixel, liste_quantile_np)
    ).float()

    print("Calcul des quantiles Target...")
    value_quantile_target = torch.from_numpy(
        np.quantile(target_pixel, liste_quantile_np)
    ).float()

    return (value_quantile_source, value_quantile_target)


if __name__ == "__main__":
    calcul_quantile()




