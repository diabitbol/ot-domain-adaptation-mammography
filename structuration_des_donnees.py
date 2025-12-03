
#restructuration des données
import os
import shutil
from PIL import Image

def restructure_normal():
    source="./MINI-DDSM-Complete-JPEG-8/Normal"
    destination="./MINI-DDSM-Complete-JPEG-81/Normal1"
    i=0
    for filename in os.listdir(source):
        i+=1
        if i%100==0:
            print(str(i)+" fichiers traités" + filename)
        source_filename= source + "\\" + filename
        chemin_MLO_right=os.path.join(source_filename,"A_"+filename+"_1.RIGHT_MLO.jpg")
        chemin_MLO_left=os.path.join(source_filename,"A_"+filename+"_1.LEFT_MLO.jpg")
        if os.path.exists(chemin_MLO_right):
            shutil.copy2(chemin_MLO_right,destination)
        #if os.path.exists(chemin_MLO_left):
        #        original_image=Image.open(chemin_MLO_left)
        #        flipped_image=original_image.transpose(Image.FLIP_LEFT_RIGHT)
        #        destination2=os.path.join(destination,"A_"+filename+"_1.LEFT_MLO.jpg")
        #        flipped_image.save(destination2, format="JPEG")
        else:
            source_filename= source + "\\" + filename
            chemin_MLO_right=os.path.join(source_filename,"D_"+filename+"_1.RIGHT_MLO.jpg")
            chemin_MLO_left=os.path.join(source_filename,"D_"+filename+"_1.LEFT_MLO.jpg")
            if os.path.exists(chemin_MLO_right):
                shutil.copy2(chemin_MLO_right,destination)
            #if os.path.exists(chemin_MLO_left):
            #        original_image=Image.open(chemin_MLO_left)
            #        flipped_image=original_image.transpose(Image.FLIP_LEFT_RIGHT)
            #        destination2=os.path.join(destination,"A_"+filename+"_1.LEFT_MLO.jpg")
            #        flipped_image.save(destination2, format="JPEG")
                
    print("Restructuration terminée")

restructure_normal()
print("c'est fait pour les normales")
def restructuration_cancer():
    source="./MINI-DDSM-Complete-JPEG-8/Cancer"
    destination="./MINI-DDSM-Complete-JPEG-81/Cancer1"
    c=0
    for filename in os.listdir(source):
        #identification de la lettre correcte
        ok=False
        i=0
        l=["A_","B_","C_","D_"]
        source_filename= source + "\\" + filename
        while ok == False:
            chemin_MLO_right=os.path.join(source_filename,l[i]+filename+"_1.RIGHT_MLO.jpg")
            if os.path.exists(chemin_MLO_right):
                ok = True
            else:
                i+=1
        if i==4:
            print("erreur1 avec le fichier "+filename)

        #identification cancer gauche ou droit
        orientation=None
        if os.path.exists(os.path.join(source_filename,l[i]+filename+"_1.RIGHT_MLO.OVERLAY")):
             orientation="RIGHT"
        elif os.path.exists(os.path.join(source_filename,l[i]+filename+"_1.LEFT_MLO.OVERLAY")):
            orientation="LEFT"
        else:
             print("cancer uniquement visible sur le point de vue CC "+filename)

        #copie des images
        if orientation=="RIGHT":
            if i==0:
                shutil.copy2(chemin_MLO_right,destination)
            elif i<4:
                original_image=Image.open(chemin_MLO_right)
                flipped_image=original_image.transpose(Image.FLIP_LEFT_RIGHT)
                destination2=os.path.join(destination,l[i]+filename+"_1.RIGHT_MLO.jpg")
                flipped_image.save(destination2, format="JPEG")
                 
        elif orientation=="LEFT":
            if i==0:
                original_image=Image.open(os.path.join(source_filename,l[i]+filename+"_1.LEFT_MLO.jpg"))
                flipped_image=original_image.transpose(Image.FLIP_LEFT_RIGHT)
                destination2=os.path.join(destination,l[i]+filename+"_1.LEFT_MLO.jpg")
                flipped_image.save(destination2, format="JPEG")
            elif i<4:
                shutil.copy2(os.path.join(source_filename,l[i]+filename+"_1.LEFT_MLO.jpg"),destination)
        c+=1
        if c%100==0:
            print(str(c)+" fichiers traités")   

    print("Restructuration terminée")



def restructuration_benign():
    source="./MINI-DDSM-Complete-JPEG-8/Benign"
    destination="./MINI-DDSM-Complete-JPEG-81/Benign1"
    c=0
    for filename in os.listdir(source):
        #identification de la lettre correcte
        ok=False
        i=0
        l=["A_","B_","C_","D_"]
        source_filename= source + "\\" + filename
        while ok == False:
            chemin_MLO_right=os.path.join(source_filename,l[i]+filename+"_1.RIGHT_MLO.jpg")
            if os.path.exists(chemin_MLO_right):
                ok = True
            else:
                i+=1
        if i==4:
            print("erreur1 avec le fichier "+filename)

        #identification tumeur bénigne gauche ou droit
        orientation=None
        if os.path.exists(os.path.join(source_filename,l[i]+filename+"_1.RIGHT_MLO.OVERLAY")):
             orientation="RIGHT"
        elif os.path.exists(os.path.join(source_filename,l[i]+filename+"_1.LEFT_MLO.OVERLAY")):
            orientation="LEFT"
        else:
             print("signe bénin uniquement visible sur le point de vue CC "+filename)

        #copie des images
        if orientation=="RIGHT":
            if i==0:
                shutil.copy2(chemin_MLO_right,destination)
            elif i<4:
                original_image=Image.open(chemin_MLO_right)
                flipped_image=original_image.transpose(Image.FLIP_LEFT_RIGHT)
                destination2=os.path.join(destination,l[i]+filename+"_1.RIGHT_MLO.jpg")
                flipped_image.save(destination2, format="JPEG")
                 
        elif orientation=="LEFT":
            if i==0:
                original_image=Image.open(os.path.join(source_filename,l[i]+filename+"_1.LEFT_MLO.jpg"))
                flipped_image=original_image.transpose(Image.FLIP_LEFT_RIGHT)
                destination2=os.path.join(destination,l[i]+filename+"_1.LEFT_MLO.jpg")
                flipped_image.save(destination2, format="JPEG")
            elif i<4:
                shutil.copy2(os.path.join(source_filename,l[i]+filename+"_1.LEFT_MLO.jpg"),destination)
        c+=1
        if c%100==0:
            print(str(c)+" fichiers traités")   

    print("Restructuration terminée")

