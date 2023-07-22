# Blob detectron inference

 Le projet vise à développer un outil pour détecter et simuler automatiquement le réseau créé par le Physarum polycephalum (blob) à partir d'une image d'entrée. 

Le CNRS réalise des expériences avec le blob dans des boîtes de pétri, et les photographies. 

![Physarum boite de petri](images_readme/boitepetri.png) 


Pour ce faire, nous déclarons 3 classes différentes :  

    - Les nœuds principaux (Np) : Point d’attirance, souvent un flocon d’avoine.  

    - Les liens (Lp) : Le chemin parcouru par le blob entre deux nœuds. 

    - Les nœuds secondaires (Ns) : croisement entre plusieurs liens. 

Afin de détecter automatique les différents éléments sur une image, nous utilisons un modèle de machine learning de détection d’objets. 

Nous exploitons ensuite les données identifiées dans un script afin d’obtenir un fichier JSON contenant les informations de position des différents objets.  

# Entrainement du modèle (Blob_training.ipynb) 

Le dataset est composé de 192 images de boite de pétri, annotées avec l’outil Supervisely. 

![Annotation avec supervisely](images_readme/supervisely.png)  

Les annotations sont exportées au format coco, et séparées en données d’entrainement "physarum_train" (80%) et de test "physarum_test" (20%). 

Le choix du modèle s’est porté sur Mask R-CNN, une extension de Faster RCNN, un modèle utilisant l’API de Detectron2. 

L'entrainement du modèle se fait sur le notebook Blob_training.ipynb

Faster R-CNN est un réseau de neurones, il fonctionne sur deux étages, d’abord il identifie les zones d’intéret, puis les envoie a un réseau neuronal convolutif. Les résultats de sortie sont passés à une Machine à vecteurs de support (SVM)  pour les classifier. 

Une régression est ensuite effectuée pour ajuster la boîte englobante prédite à la position et à la taille exactes de l'objet réel dans la région d'intérêt. 

Mask R-CNN est capable de réaliser à la fois la détection d'objets et la segmentation sémantique des instances détectées, ce qui nous permet d’obtenir des masques des objets détectés, plutôt que de simples box. 

## Réglages d’entrainement du modèle :  

Pour accélérer l'entraînement et améliorer les performances du modèle, le poids initial du réseau est chargé à partir du modèle pré-entraîné "R-50.pkl" sur ImageNet. 

cfg.SOLVER.IMS_PER_BATCH = 4 définit le nombre d'images utilisées pour chaque mise à jour de gradient. Une taille de lot plus grande permet d'exploiter davantage les ressources GPU, accélérant ainsi l'entraînement. 

Le modèle est entraîné pendant cfg.SOLVER.MAX_ITER = 3000 itérations. Le nombre d'itérations contrôle la durée de l'entraînement et permet au modèle de converger vers une solution optimale. 

Taille du lot par image : Le paramètre cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8 spécifie le nombre de propositions de région (ROI) pour chaque image d'entraînement. 

Seuil de score de détection : Le seuil de score de détection est fixé à cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.45. Cela signifie que seules les détections dont le score est supérieur à 0.45 seront prises en compte lors de l'inférence. 

## Mesures de performance du modèle

Pour mesurer les performances du modèle, on visualise certaines métriques de performance :

![Métriques de performance](images_readme/evaluation.png)

On évalue la performance du modèle :  
- Average Precision : en comparant ses prédictions avec les annotations réelles à différentes valeurs seuil d'Intersection over Union (IoU : mesure le degré de chevauchement entre la prédiction du modèle et l'annotation réelle d'un objet).
- Average Recall : indique quelle proportion des objets réellement présents dans l'image a été correctement détectée par le modèle.

On visualise aussi les résultats pour mesurer les performances au fur et a mesure des réglages : 

![Visualisation Résultats](images_readme/visualisation1.png)  ![Visualisation NetworkX](images_readme/visualisation2.png)


# Fonctionnement du script :  

Le script Blob_detection.py prends une image en entrée et donne en sortie un fichier Json contenant les positions des objets prédits. 

Pour ce faire, le script est composé de 6 fonctions :  

## prediction(img_name, data_dir): 

Cette fonction effectue une prédiction des objets présents sur l’image, et donne en sortie les informations à leur sujet.  

Cette prédiction s’effectue à l’aide du modèle pré-entrainé, enregistré en modèle Torchscript. 

Params:  
    
    - img_name : nom du fichier d el’image à détecter. 

    - data_dir : chemin du dossier ou se trouvent le script. 

Returns: 

    - outputs : variable qui contient les informations des prédictions. 

    - img : l’image d’entrée. 
        
Dans le processus, l’image est transformée de RGB (Red Green Blue) à BGR pour que le modèle fonctionne correctement.  

## extractionOutputs(outputs): 

Cette fonction extrait les données des ojbets prédits et les places dans des listes. 

Params :  
    
    - outputs : variable qui contient les informations des prédictions. 

Returns : 
   
    - nœuds : liste des nœuds détectés. 

    - positions : dictionnaire contenant les positions (x,y) de chaque nœud. 

    - positionLiens : dictionnaire contenant les positions (x,y) de chaque lien. 

La fonction parcourt la variable output, prend le premier tensor (output[0][i]) qui contient les box de prédictions. 

On prend le centre de chaque boite pour identifier la position des objets.  

On vérifie la classe prédite (pred_class) afin de placer l’objet dans la bonne liste. 

Pour les nœuds secondaires, on vérifie qu’aucun autre nœud ne se situe à moins de 6, car il arrive au modèle de détecter deux nœuds secondaires au même endroit. 

## detectionLiens(positions, positionLiens, seuil, marge): 

Cette fonction détecte si un lien est présent entre deux nœuds à l’aide des listes précédemment produites. 

Params : 
   
    - positions : dictionnaire contenant les positions (x,y) de chaque nœud. 

    - positionLiens : dictionnaire contenant les positions (x,y) de chaque lien. 

Returns :  
   
    - listeLiensFinal : liste des liens détectés entre deux nœuds, composés de tuples de nœuds. 

Une double boucle parcourt la liste de nœuds, trace une droite entre chaque paire de nœuds, puis parcourt la liste de liens. On calcule la distance entre le lien et la droite, si elle est inférieure au “seuil”, on considère que le lien est prédsent entre deux nœuds.  

De plus, on vérifie que les x et y du lien, sont placés entre les x et y des nœuds (avec une “marge”). 

Ensuite, on parcourt nos résultats, pour chaque paire de nœuds reliée par un lien, on vérifie qu’il n’existe pas une paire de nœuds plus rapprochés.  

## drawGraph(nœuds, positions, listeLiensFinal): 

Cette fonction crée un graphe NetworkX à partir des listes de nœuds et de liens. 

Params : 
   
    - nœuds:  liste des nœuds. 

    - positions : dictionnaire contenant les positions (x,y) de chaque nœud. 

    - listeLiensFinal: Liste des paires de nœuds connectés par des liens. 

## conversionBlobRecorder(image,graph,positions,echelle,img_name): 

Cette fonction converti le graphe NetworkX en JSON exploitable par l’outil Blob Recorder. 

Params :  
   
    - image: c’est l’image d’origine. 

    - graph: le graphe NetworkX. 

    - positions: le dictionnaire contennant les positions des différents nœuds. 

    - echelle: une echelle pour convenir au format d’image de Blob Recorder.  

    - img_name: le nom de l’image 

La fonction parcourt les listes de nœuds et liens, pour rentrer les valeurs dans un dictionnaire au format attendu par Blob Recorder. 

Ensuite elle enregistre le résultat dans un fichier JSON au même nom que l’image. 

## blobDetection(img_name, data_dir): 


Cette dernière fonction permet de lancer les autres fonctions.  


Params : 
   
    - img_name : le nom de l’image à détecter. 
    
    - data_dir : le chemin du répertoire ou se trouvent les fichiers. 

 

# Limites/Problèmes rencontrés

Quelques fois, le blob trace un lien avec une trop forte courbure, et la fonction detectionLiens() utilise son centre pour détecter sa présence, ce qui peut amener à des erreurs, c'est à ca que servent "seuil" et "marge", mais cela ne suffit pas toujours : 

![Erreur detection lien](images_readme/pb_lien.png)

Il arrive que le modèle détecte 2 Noeuds secondaires collés l'un à l'autre au lieu d'un seul. Ce problème a été réglé en post-traitement dans extractionOutputs() : 

![Erreur detection Ns](images_readme/pb_ns.png)

