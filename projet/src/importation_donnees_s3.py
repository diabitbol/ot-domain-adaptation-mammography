import os


def sync_s3_to_local(s3_path, local_path):
    """
    Synchronise les données du S3 vers le disque local si elles n'existent pas.
    """    
    # On crée le dossier local s'il n'existe pas pour éviter les erreurs
    os.makedirs(s3_path, exist_ok=True)
    
    # La commande de miroir
    # On utilise f-string pour injecter tes chemins définis dans config.py
    cmd = f"mc mirror {s3_path} {local_path}"
    
    print(f"Exécution de : {cmd}")
    exit_code = os.system(cmd)
    
    if exit_code == 0:
        print("Synchronisation réussie !")
    else:
        print(f"Erreur lors de la synchronisation. Code : {exit_code}")


if __name__ == "__main__":
    sync_s3_to_local()

