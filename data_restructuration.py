import os
import shutil
import cv2 
import numpy as np
from pathlib import Path
from tqdm import tqdm
import seaborn as sns


def restructure_normale_data(
    data_local,
    destination
):
    source = data_local+"Normal/"
    destination = destination+"0-normal/"
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


def restructuration_cancer(
    data_local,
    destination
):
    source = data_local+"Cancer/"
    destination1 = destination+"1-cancer_benign/"
    destination2 = destination+"0-normal/"
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


def restructuration_benign(
    data_local,
    destination
):
    source = data_local+"Benign/"
    destination1 = destination+"1-cancer_benign/"
    destination2 = destination+"0-normal/"
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


def process_mammo(
    image_path,
    rh,
    rb,
    valeur_de_coupure,
    rhp
):
    img=cv2.imread(image_path,0) #on récupère l'image
    hauteur,largeur=img.shape
    img=img[rh:hauteur-rb,:] #on rogne le haut et le bas de 20 pixels ccar ils sont trop clairs et ils modifient le masque
    _,thresh=cv2.threshold(img,valeur_de_coupure,255,cv2.THRESH_BINARY) #on isole les formes
    contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #on récupère les contours
    large_cnt=max(contours,key=cv2.contourArea) #on récupère le plus grand contours correspondant au sein
    mask=np.zeros_like(img)
    cv2.drawContours(mask,[large_cnt],-1,255,-1) #on créée une image noir avec le contour du sein rempli de blanc 
    clean_img=cv2.bitwise_and(img,mask) # -> ce qui est rempli de blanc reste, ce qui est rempli de noir devient noir
    x, y, w, h = cv2.boundingRect(large_cnt) # on récupère les valeurs extrèmes du contour
    cropped_img = clean_img[y:y+h, x:x+w] # pour rogner 
    new_hauteur, new_largeur=cropped_img.shape #maintenant on met aux proportion du vindr en 1520*912 
    (vindr_hauteur, vindr_largeur)=(1520,912)  #pour ne pas avoir de déformation différente des images de celles du vindr
    rapport_vindr=vindr_hauteur/vindr_largeur   #pour ce faire, comme la largeur est fixée de la même manière dans les deux datasets (début et bout du sein en largeur)
    new_hauteur_goal=rapport_vindr*new_largeur  #on a juste à ajuster la hauteur
    difference_pixel=int(new_hauteur-new_hauteur_goal) #sinon on ajuste pas et on voit s'il n'y a pas trop d'images sur lesquelles on ne réajuste pas les proportions
    proportion_pas_ok=1
    if difference_pixel>0:
        rognage_en_haut=int(rhp*difference_pixel)
        rognage_en_bas=difference_pixel-rognage_en_haut
        cropped_img=cropped_img[rognage_en_haut:new_hauteur-rognage_en_bas,:]
        proportion_pas_ok=0 #pour connaître le nombre d'image pas aux proportions de celles du vindr
    return((cropped_img,proportion_pas_ok))    



def traitement_image(
    chemin_dossier_destination,
    source_data_normal,
    source_data_cancer_benign,
    rh,
    rb,
    valeur_de_coupure,
    rhp,
    res_min
):
    path_obj_normal = Path(source_data_normal)
    path_obj_cancer_benign= Path(source_data_cancer_benign)
    image_normal_paths = list(path_obj_normal.glob("*.jpg")) 
    image_cancer_paths = list(path_obj_cancer_benign.glob("*.jpg"))
    
    destination_cancer_benign= os.path.join(chemin_dossier_destination,"1-cancer_benign")
    destination_normal=os.path.join(chemin_dossier_destination,"0-normal")
    os.makedirs(chemin_dossier_destination, exist_ok=True)
    os.makedirs(destination_cancer_benign, exist_ok=True)
    os.makedirs(destination_normal, exist_ok=True)
    compteur_proportion_pas_ok=0
    i=0

    for image in tqdm(image_normal_paths):
        (new_image,b)=process_mammo(
            image,
            rh,
            rb,
            valeur_de_coupure,
            rhp
        )
        (hauteur,largeur)=new_image.shape
        if b==0 and hauteur>=res_min and largeur >=res_min:
            cv2.imwrite(os.path.join(destination_normal, f"image_{i}.jpg"), new_image)
        else:
            compteur_proportion_pas_ok=compteur_proportion_pas_ok+b
        i+=1
    for image in tqdm(image_cancer_paths):
        (new_image,b)=process_mammo(
            image,
            rh,
            rb,
            valeur_de_coupure,
            rhp
        )
        (hauteur,largeur)=new_image.shape
        if b==0 and hauteur>=res_min and largeur >=res_min:
            cv2.imwrite(os.path.join(destination_cancer_benign, f"image_{i}.jpg"), new_image)
        else:
            compteur_proportion_pas_ok=compteur_proportion_pas_ok+b
        i+=1
    return(f"traitement_et_enregistrement_terminé au chemin {chemin_dossier_destination} \n proportion d'images non aux proportions vindr : {compteur_proportion_pas_ok/i} \n ATTENTION : penser à enregistrer sur onyxia")
