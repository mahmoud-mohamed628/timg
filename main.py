import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog

# --- 1. SÉLECTION DE LA VIDÉO ---
# Initialiser tkinter et cacher la fenêtre principale
root = tk.Tk()
root.withdraw()

# Ouvrir une fenêtre pour choisir la vidéo
print("Ouverture de la fenêtre pour choisir une vidéo...")
video_path = filedialog.askopenfilename(
    title="Choisis une vidéo à analyser",
    filetypes=[("Fichiers Vidéo", "*.mp4 *.avi *.mov *.mkv"), ("Tous les fichiers", "*.*")]
)

# Vérifier si l'utilisateur a choisi une vidéo ou a annulé
if not video_path:
    print("Erreur: Aucune vidéo n'a été sélectionnée. Fin du programme.")
    exit()

# Extraire le nom de la vidéo sans l'extension (ex: 'bolt' à partir de 'C:/dossier/bolt.mp4')
video_filename = os.path.basename(video_path)
video_name_no_ext = os.path.splitext(video_filename)[0]
print(f"Vidéo sélectionnée : {video_filename}")

# --- 2. CRÉATION DU DOSSIER DE SORTIE ---
output_dir = 'rapports_graphes'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Dossier '{output_dir}' créé avec succès.")

# Charger la vidéo
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Erreur: Impossible de lire le fichier video '{video_path}'.")
    exit()

# Paramètres de Lucas-Kanade
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Paramètres de Shi-Tomasi
feature_params = dict(maxCorners=50,
                       qualityLevel=0.01,
                       minDistance=10,
                       blockSize=3)

# Lire la première frame
ret, old_frame = cap.read()
if not ret:
    print("Erreur: Vidéo vide.")
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Initialisation des tableaux
trajectory_x = []
trajectory_y = []
speeds = []
directions = []
frames_list = []
frame_idx = 0

print("Analyse du mouvement en cours... Patiente jusqu'à la fin de la vidéo. (Appuie sur 'q' pour arrêter)")

while True:
    ret, frame = cap.read()
    if not ret:
        break # Fin de la vidéo
        
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_idx += 1
    
    if p0 is not None and len(p0) > 0:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        if len(good_new) > 0:
            center_x = np.mean(good_new[:, 0])
            center_y = np.mean(good_new[:, 1])
            trajectory_x.append(center_x)
            trajectory_y.append(center_y)
            frames_list.append(frame_idx)
            
            delta_x = np.mean(good_new[:, 0] - good_old[:, 0])
            delta_y = np.mean(good_new[:, 1] - good_old[:, 1])
            
            speed = np.sqrt(delta_x**2 + delta_y**2)
            speeds.append(speed)
            
            direction = np.degrees(np.arctan2(-delta_y, delta_x))
            directions.append(direction)
            
            mask_draw = np.zeros_like(frame)
            for new, old in zip(good_new, good_old):
                a, b = int(new[0]), int(new[1])
                c, d = int(old[0]), int(old[1])
                cv2.arrowedLine(mask_draw, (c, d), (a, b), (0, 255, 0), 2, tipLength=0.3)
                cv2.circle(frame, (a, b), 4, (0, 0, 255), -1)
                
            output_img = cv2.addWeighted(frame, 0.8, mask_draw, 1, 0)
            cv2.imshow(f'Estimation Mouvement - {video_name_no_ext}', output_img)
            
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        else:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
    else:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        old_gray = frame_gray.copy()

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- 3. GÉNÉRATION ET SAUVEGARDE DES FIGURES ---
print("Génération des figures...")
plt.figure(figsize=(15, 4))

# Figure 1: Trajectoire globale 2D
plt.subplot(1, 3, 1)
plt.plot(trajectory_x, trajectory_y, color='blue', linewidth=2)
plt.gca().invert_yaxis()
plt.title(f"1. Trajectoire Globale ({video_name_no_ext})")
plt.xlabel("Position X (pixels)")
plt.ylabel("Position Y (pixels)")
plt.grid(True)

# Figure 2: Vitesse scalaire
plt.subplot(1, 3, 2)
plt.plot(frames_list, speeds, color='red', linewidth=2)
plt.title(f"2. Vitesse ({video_name_no_ext})")
plt.xlabel("Frames (Temps)")
plt.ylabel("Vitesse (pixels/frame)")
plt.grid(True)

# Figure 3: Direction en degrés
plt.subplot(1, 3, 3)
plt.plot(frames_list, directions, color='green', linewidth=2)
plt.title(f"3. Direction ({video_name_no_ext})")
plt.xlabel("Frames (Temps)")
plt.ylabel("Direction (Degrés)")
plt.grid(True)

plt.tight_layout()

# Sauvegarde avec le nom dynamique dans le dossier spécifique
output_file_path = os.path.join(output_dir, f"{video_name_no_ext}_graphes.png")
plt.savefig(output_file_path) 
plt.show()

print(f"Terminé avec succès ! L'image a été enregistrée ici : {output_file_path}")