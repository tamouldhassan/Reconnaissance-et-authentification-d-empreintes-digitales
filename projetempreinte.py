import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from skimage.morphology import skeletonize
from scipy.spatial.distance import euclidean

class SystemeReconnaissanceEmpreintes:
    def __init__(self, fenetre_principale):
        self.fenetre_principale = fenetre_principale
        self.fenetre_principale.title("Système de Reconnaissance d'Empreintes Digitales")
        self.fenetre_principale.geometry("1400x900")
        self.fenetre_principale.configure(bg='#f0f0f0')
        
        # Variables
        self.dataset_empreintes = {}
        self.empreinte_courante = None
        self.image_originale = None
        self.image_binarisee = None
        self.image_squelettisee = None
        self.minuties_courantes = []
        
        self.creer_interface()
    
    def creer_interface(self):
        # fenetre
        frame_principal = tk.Frame(self.fenetre_principale, bg='#f0f0f0')
        frame_principal.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        #  les boutons
        frame_boutons = tk.Frame(frame_principal, bg='#f0f0f0')
        frame_boutons.pack(fill=tk.X, pady=(0, 10))
        
       
        btn_charger_dataset = tk.Button(
            frame_boutons, 
            text="Charger Dataset", 
            command=self.charger_dataset,
            bg='#4CAF50', 
            fg='white', 
            font=('Arial', 12, 'bold'),
            padx=20
        )
        btn_charger_dataset.pack(side=tk.LEFT, padx=(0, 10))
        
        btn_charger_empreinte = tk.Button(
            frame_boutons, 
            text="Charger Empreinte", 
            command=self.charger_empreinte,
            bg='#2196F3', 
            fg='white', 
            font=('Arial', 12, 'bold'),
            padx=20
        )
        btn_charger_empreinte.pack(side=tk.LEFT, padx=(0, 10))
        
        btn_analyser = tk.Button(
            frame_boutons, 
            text="Analyser Empreinte", 
            command=self.analyser_empreinte,
            bg='#FF9800', 
            fg='white', 
            font=('Arial', 12, 'bold'),
            padx=20
        )
        btn_analyser.pack(side=tk.LEFT, padx=(0, 10))
        
        self.label_statut = tk.Label(
            frame_boutons, 
            text="Prêt - Chargez un dataset et une empreinte", 
            bg='#f0f0f0',
            font=('Arial', 10)
        )
        self.label_statut.pack(side=tk.RIGHT)
        
        frame_images = tk.Frame(frame_principal, bg='#f0f0f0')
        frame_images.pack(fill=tk.X, pady=(0, 10))
        
        self.creer_frame_image(frame_images, "Image Originale", 0)
        self.creer_frame_image(frame_images, "Image Binarisée", 1)
        self.creer_frame_image(frame_images, "Image Squelettisée", 2)
        
        frame_comparaison = tk.Frame(frame_principal, bg='#f0f0f0')
        frame_comparaison.pack(fill=tk.BOTH, expand=True)
        
        label_comparaison = tk.Label(
            frame_comparaison, 
            text="Tableau de Comparaison", 
            bg='#f0f0f0',
            font=('Arial', 14, 'bold')
        )
        label_comparaison.pack(pady=(0, 10))
        
        colonnes = ("Nom", "Score de Similarité", "Statut")
        self.arbre_comparaison = ttk.Treeview(frame_comparaison, columns=colonnes, show='headings', height=10)
        
        for col in colonnes:
            self.arbre_comparaison.heading(col, text=col)
            self.arbre_comparaison.column(col, width=350, anchor='center')
        
        scrollbar = ttk.Scrollbar(frame_comparaison, orient=tk.VERTICAL, command=self.arbre_comparaison.yview)
        self.arbre_comparaison.configure(yscrollcommand=scrollbar.set)
        
        self.arbre_comparaison.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def creer_frame_image(self, parent, titre, colonne):
        frame = tk.Frame(parent, bg='white', relief=tk.RAISED, bd=2)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        label_titre = tk.Label(frame, text=titre, bg='white', font=('Arial', 12, 'bold'))
        label_titre.pack(pady=5)
        
        label_image = tk.Label(frame, bg='white')
        label_image.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        if colonne == 0:
            self.label_originale = label_image
        elif colonne == 1:
            self.label_binarisee = label_image
        elif colonne == 2:
            self.label_squelettisee = label_image
    
    def redimensionner_image_pour_affichage(self, image, taille_max=(350, 300)):
        h, w = image.shape[:2]
        
        ratio = min(taille_max[0]/w, taille_max[1]/h)
        nouvelle_largeur = int(w * ratio)
        nouvelle_hauteur = int(h * ratio)
        
        return cv2.resize(image, (nouvelle_largeur, nouvelle_hauteur))
    
    def charger_dataset(self):
        dossier = filedialog.askdirectory(title="Sélectionner le dossier du dataset")
        if dossier:
            self.dataset_empreintes = {}
            formats_supportes = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
            
            for fichier in os.listdir(dossier):
                if fichier.lower().endswith(formats_supportes):
                    chemin_complet = os.path.join(dossier, fichier)
                    nom_empreinte = os.path.splitext(fichier)[0]
                    
                    image = cv2.imread(chemin_complet, cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        minuties = self.extraire_minuties(image)
                        self.dataset_empreintes[nom_empreinte] = {
                            'minuties': minuties,
                            'image': image
                        }
            
            self.mettre_a_jour_statut(f"Dataset chargé: {len(self.dataset_empreintes)} empreintes")
    
    def charger_empreinte(self):
        fichier = filedialog.askopenfilename(
            title="Sélectionner une empreinte",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if fichier:
            self.empreinte_courante = fichier
            self.image_originale = cv2.imread(fichier, cv2.IMREAD_GRAYSCALE)
            self.afficher_image_originale()
            self.mettre_a_jour_statut("Empreinte chargée - Cliquez sur 'Analyser' pour traiter")
    
    def afficher_image_originale(self):
        if self.image_originale is not None:
            image_redimensionnee = self.redimensionner_image_pour_affichage(self.image_originale)
            image_pil = Image.fromarray(image_redimensionnee)
            image_tk = ImageTk.PhotoImage(image_pil)
            
            self.label_originale.configure(image=image_tk)
            self.label_originale.image = image_tk
    
    def analyser_empreinte(self):
        if self.image_originale is None:
            messagebox.showwarning("Attention", "Veuillez d'abord charger une empreinte")
            return
        
        self.mettre_a_jour_statut("Analyse en cours...")
        
        # Binarisation
        self.image_binarisee = self.binariser_image(self.image_originale)
        self.afficher_image_binarisee()
        
        # Squelettisation
        self.image_squelettisee = self.squelettiser_image(self.image_binarisee)
        self.afficher_image_squelettisee()
        
        #  Extraction des minuties
        self.minuties_courantes = self.extraire_minuties(self.image_originale)
        
        # Comparaison avec le dataset
        if self.dataset_empreintes:
            self.comparer_avec_dataset()
        else:
            messagebox.showwarning("Attention", "Aucun dataset chargé pour la comparaison")
        
        self.mettre_a_jour_statut(f"Analyse terminée - {len(self.minuties_courantes)} minuties détectées")
    
    def binariser_image(self, image):
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        image_amelioree = clahe.apply(image)
        
        # Filtre gaussien 
        image_lissee = cv2.GaussianBlur(image_amelioree, (3, 3), 0)
        
        # Binarisation adaptative
        image_binarisee = cv2.adaptiveThreshold(
            image_lissee, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 5
        )
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        image_binarisee = cv2.morphologyEx(image_binarisee, cv2.MORPH_CLOSE, kernel)
        image_binarisee = cv2.morphologyEx(image_binarisee, cv2.MORPH_OPEN, kernel)
        
        return image_binarisee
    
    def afficher_image_binarisee(self):
        if self.image_binarisee is not None:
        
            image_affichage = cv2.bitwise_not(self.image_binarisee)
            image_redimensionnee = self.redimensionner_image_pour_affichage(image_affichage)
            image_pil = Image.fromarray(image_redimensionnee)
            image_tk = ImageTk.PhotoImage(image_pil)
            
            self.label_binarisee.configure(image=image_tk)
            self.label_binarisee.image = image_tk
    
    def squelettiser_image(self, image_binarisee):

        image_bool = image_binarisee.astype(bool)
        squelette = skeletonize(image_bool)
        return (squelette * 255).astype(np.uint8)
    
    def afficher_image_squelettisee(self):
        if self.image_squelettisee is not None:
            image_affichage = cv2.bitwise_not(self.image_squelettisee)
            image_redimensionnee = self.redimensionner_image_pour_affichage(image_affichage)
            image_pil = Image.fromarray(image_redimensionnee)
            image_tk = ImageTk.PhotoImage(image_pil)
            self.label_squelettisee.configure(image=image_tk)
            self.label_squelettisee.image = image_tk
    def extraire_minuties(self, image):
        minuties = []
        
        try:
            image_preprocessed = self.preprocesser_image(image)
            
            # Binarisation
            image_bin = self.binariser_image(image_preprocessed)
            
            # Squelettisation
            image_bool = image_bin.astype(bool)
            squelette = skeletonize(image_bool)
            squelette_uint8 = (squelette * 255).astype(np.uint8)
            
            # Détection des minuties avec une approche améliorée
            minuties = self.detecter_minuties_avancees(squelette_uint8)
            
        except Exception as e:
            print(f"Erreur lors de l'extraction des minuties: {e}")
            minuties = []
        
        return minuties
    
    def preprocesser_image(self, image):
        # Normalisation de l'image
        image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        
        # Filtre de Gabor pour améliorer les crêtes
        kernel = cv2.getGaborKernel((21, 21), 8, np.pi/4, 2*np.pi, 0.5, 0, ktype=cv2.CV_32F)
        filtered_img = cv2.filter2D(image_norm, cv2.CV_8UC3, kernel)
        
        return filtered_img
    
    def detecter_minuties_avancees(self, squelette):
        minuties = []
        h, w = squelette.shape
        for y in range(10, h-10):
            for x in range(10, w-10):
                if squelette[y, x] == 255:  # Point du squelette
                    # Analyser le voisinage 3x3
                    voisinage = squelette[y-1:y+2, x-1:x+2]
                    voisins_actifs = np.sum(voisinage == 255)
                    
                    # Point de terminaison (1 voisin) ou bifurcation (3+ voisins)
                    if voisins_actifs == 2:  # Point de terminaison
                        minuties.append((x, y, 'terminaison'))
                    elif voisins_actifs >= 4:  # Point de bifurcation
                        minuties.append((x, y, 'bifurcation'))
        
        # Filtrage pour éviter les minuties trop proches
        minuties_filtrees = self.filtrer_minuties_proches(minuties)
        
        return minuties_filtrees[:25]  # Limiter à 25 minuties de qualité
    
    def filtrer_minuties_proches(self, minuties, distance_min=15):
        minuties_filtrees = []
        
        for minutie in minuties:
            x1, y1 = minutie[0], minutie[1]
            trop_proche = False
            
            for minutie_existante in minuties_filtrees:
                x2, y2 = minutie_existante[0], minutie_existante[1]
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                
                if distance < distance_min:
                    trop_proche = True
                    break
            
            if not trop_proche:
                minuties_filtrees.append(minutie)
        
        return minuties_filtrees
    
    def calculer_similarite_avancee(self, minuties1, minuties2):
        if not minuties1 or not minuties2:
            return 0.0
        
        score_total = 0.0
        correspondances = 0
        seuil_distance = 25
        
        # Matrice de correspondance
        matrice_distances = []
        for i, m1 in enumerate(minuties1):
            distances_ligne = []
            for j, m2 in enumerate(minuties2):
                distance = euclidean((m1[0], m1[1]), (m2[0], m2[1]))
                distances_ligne.append(distance)
            matrice_distances.append(distances_ligne)
        
        # Trouver les meilleures correspondances
        minuties1_utilisees = set()
        minuties2_utilisees = set()
        
        for _ in range(min(len(minuties1), len(minuties2))):
            min_distance = float('inf')
            meilleure_paire = None
            
            for i in range(len(minuties1)):
                if i in minuties1_utilisees:
                    continue
                for j in range(len(minuties2)):
                    if j in minuties2_utilisees:
                        continue
                    
                    if matrice_distances[i][j] < min_distance:
                        min_distance = matrice_distances[i][j]
                        meilleure_paire = (i, j)
            
            if meilleure_paire and min_distance < seuil_distance:
                i, j = meilleure_paire
                minuties1_utilisees.add(i)
                minuties2_utilisees.add(j)
                correspondances += 1
                score_total += (seuil_distance - min_distance) / seuil_distance
        
        if correspondances == 0:
            return 0.0
        
        score_correspondances = correspondances / max(len(minuties1), len(minuties2))
        score_qualite = score_total / correspondances if correspondances > 0 else 0
        
        score_final = (score_correspondances * 0.6 + score_qualite * 0.4) * 100
        return min(score_final, 100.0)
    
    def comparer_avec_dataset(self):
        for item in self.arbre_comparaison.get_children():
            self.arbre_comparaison.delete(item)
        
        if not self.minuties_courantes:
            messagebox.showwarning("Attention", "Aucune minutie détectée dans l'empreinte courante")
            return
        
        resultats = []
        self.mettre_a_jour_statut("Comparaison en cours...")
        
        for nom_empreinte, data in self.dataset_empreintes.items():
            minuties_dataset = data['minuties']
            score = self.calculer_similarite_avancee(self.minuties_courantes, minuties_dataset)
            resultats.append((nom_empreinte, score))
       
        resultats.sort(key=lambda x: x[1], reverse=True)
        
        for nom, score in resultats:
            if score == 100:
                statut = "Simillaire"
                tags = ('simillaire',)
            else:
                statut = "DIFFÉRENT"
                tags = ('different',)
            
            self.arbre_comparaison.insert('', 'end', values=(
                nom, 
                f"{score:.2f}%", 
                statut
            ), tags=tags)
        
        # des couleurs
        self.arbre_comparaison.tag_configure('match', background='#c8e6c9', foreground='#2e7d32')
        self.arbre_comparaison.tag_configure('similaire', background='#fff3e0', foreground='#f57c00')
        self.arbre_comparaison.tag_configure('different', background='#ffebee', foreground='#c62828')
        
        self.mettre_a_jour_statut(f"Comparaison terminée - {len(resultats)} empreintes comparées")
    
    def mettre_a_jour_statut(self, message):
        self.label_statut.configure(text=message)
        self.fenetre_principale.update_idletasks()

def main():
    fenetre = tk.Tk()
    app = SystemeReconnaissanceEmpreintes(fenetre)
    fenetre.mainloop()

if __name__ == "__main__":
    main()