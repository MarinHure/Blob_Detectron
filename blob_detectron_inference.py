# !pip install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
# !pip install git+https://github.com/facebookresearch/fvcore.git

import torch
import torchvision
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import bbox_artist
import math
import networkx as nx
import json
import matplotlib.pyplot as plt

def prediction(img_name, data_dir):
  """
  This function predicts objects in an image, using a pretrained torchscript model.

  :param img_name: This is the name of the image file
  :param data_dir: The path of the directory

  :returns: Returns the predictions and the image (numpy array)
  """
  SEP = os.path.sep
  tst_imgs = os.path.join(data_dir, "test_images")  # Chemin du dossier des images pour tester
  model_dir_ts = os.path.join(data_dir, "model_ts") # Chemin du dossier ou est stocké le modèle
  model_path_ts = os.path.join(model_dir_ts, "detectron2_physarum_complet.pt")
  img_test = os.path.join(tst_imgs, img_name)
  pil_img = Image.open(img_test) # Charger l'image
  img = np.array(pil_img)[:,:,::-1] # BGR
  # loaded_model = torch.jit.load(model_path_ts) # Charger le modèle
  # _ = loaded_model.to('cpu')
  loaded_model = torch.load(model_path_ts, map_location=torch.device('cpu'))
  outputs = loaded_model(torch.as_tensor(img.astype('float32').transpose(2, 0, 1))) # Faire les prédictions
  return outputs, img


def extractionOutputs(outputs):
  """
  This function extracts the prediction informations, and put them in arrays.

  :param outputs: This is the output for the predictions
  :returns: Returns arrays that contains the nodes informations (name, position) and their links (name, position)
  """
  nodes = []
  positions = {}
  liens = []
  linkPositions = {}
  c= c1= c2=1 # Compteurs (Np, Liens, Ns)

  for i in range(len(outputs[0])):
    bbox = outputs[0][i]
    bbox = bbox.detach().cpu().numpy()
    pred_classes = outputs[1][i]
    x=((bbox[0]+bbox[2])/2) #  On prend le centre de la boite
    y=((bbox[1]+bbox[3])/2)

    if pred_classes == 0: # Noeud principal
      bbox_str = 'Np' + str(c) # Nommage avec compteur
      c=c+1
      nodes.append(bbox_str) # Ajout du nœud à la liste des nœuds
      positions[bbox_str] = [x,y] 

    elif pred_classes == 1: #Liens
      bbox_str = 'L' + str(c1)
      c1=c1+1
      liens.append(bbox_str)
      linkPositions[bbox_str] = [x,y]

    elif pred_classes == 2: # Noeud secondaire
      verif= True
      bbox_str = 'Ns' + str(c2)
      for j in positions:
        if bbox_str!=j:
          x1=positions[j][0]
          y1=positions[j][1]
          distance=math.sqrt((x1-x)**2 + (y1-y)**2)  #  On vérifie qu'il n'y a pas de Ns détectés collés (le modèle fait l'erreur qq fois)
          if distance<6 :
             verif=False
             print(bbox_str,j,distance)
             break
      if verif :
        c2=c2+1
        nodes.append(bbox_str)
        positions[bbox_str] = [x,y]
  return nodes, positions, linkPositions


def detectionLiens(positions, linkPositions):
  """
  This function detects if there is a link between two nodes.

  :param positions: Contains the positions of the nodes.
  :param linkPositions: Contains the positions of the links.
  :returns: Returns the list of links, represented by a tuple of nodes.
  """
  threshold=25 # threshold distance lien / droite entre deux nodes
  margin=25 # margin POINT(lien) COMPRIS ENTRE X1 X2
  listeLiens = []  # Liste de liens détectés entre deux nodes (avec distance entre les nodes)
  for i in positions:
    for j in positions:
      if i!= j:
        x1= positions[i][0]
        x2= positions[j][0]
        y1= positions[i][1]
        y2= positions[j][1]
        ecart = math.sqrt((x2-x1)**2 + (y2-y1)**2) #  Distance entre 2 nodes
        for k in linkPositions:
          x= linkPositions[k][0]
          y= linkPositions[k][1]
          distance = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) # Distance point - ligne
          if (distance <= threshold) and ((y1-margin <= y <= y2+margin) or (y2-margin<= y <= y1+margin)) and ((x1-margin<= x <=x2+margin) or (x2-margin<= x <= x1+margin)) : # Vérifie si lien présent à distance (threshold) de la droite tracée par deux nodes
            listeLiens.append((k,i,j,ecart))

  resultats = {} # On ne garde que les paires de nodes les plus proches
  for k, i, j, ecart in listeLiens:
      if k not in resultats:  # Si lien pas encore dans le dictionnaire
          resultats[k] = (i, j, ecart)
      else:
          _, _, min_ecart = resultats[k] #  Ecart minimum enregistré
          if ecart < min_ecart: # Comparaison
              resultats[k] = (i, j, ecart)

  listeLiensFinal = [] # Liste finale pour NetworkX
  for k in resultats:
    i, j, _ = resultats[k]
    listeLiensFinal.append((i,j))
  return listeLiensFinal


def drawGraph(nodes, positions, listeLiensFinal, image):
  """
  This function creates a NetworkX graph from the nodes/link informations and overlay the graph on the base image.

  :param nodes: This is a list of the existing nodes.
  :param positions: This is a dictionnary of the coordinates of each node.
  :param listeLiensFinal: Contains tuples of nodes that are connected by a link.
  :param image: This is the original image
  :returns: Rerturns the NetworkX Graph that has been created.
  """
  graph = nx.Graph()
  graph.add_nodes_from(nodes)
  graph.add_edges_from(listeLiensFinal)
  return graph


def conversionBlobRecorder(img,graph,positions,echelle,img_name):
  """
  This function converts the NetworkX graph to the "Blob Recorder" JSON format and saves it into the JSON_results folder.

  :param img: This is the numpy array image.
  :param graph: This is the NetworkX graph.
  :param positions: Contains the coordinates of each node.
  :param echelle: This is a scaling variable, to fit the Blob Recorder image.
  :param img_name: This is the name of the image.
  :returns: Returns a JSON file, with the name of the image.
  """
  image_height = img.shape[0] # Hauteur de l'image pour inversion axe y
  blobRecorder = {          # Dictionnaire stockant données
      "type": "FeatureCollection",
      "features": []
  }
  # Ajout nœuds
  for node in graph.nodes:
      x, y = positions[node]    # On récupère les posisitions
      y_inverted = image_height - y  # Inverser axe y
      x_echelle = x * echelle # Mise à l'échelle
      y_echelle = y_inverted * echelle # Mise à l'échelle
      if(node.startswith('Np')): # Noeud Principal
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [x_echelle, y_echelle]
            },
            "properties": {
                "geometryNamex": "Noeud source",
                "_cost": 1,
                "_customName": node
            }
          }
      elif(node.startswith('Ns')):   # Noeud secondaire
              feature = {
                  "type": "Feature",
                  "geometry": {
                      "type": "Point",
                      "coordinates": [x_echelle, y_echelle]
                  },
                  "properties": {
                      "geometryNamex": "Noeud secondaire",
                      "_cost": 1,
                      "_customName": node
                  }
              }
      blobRecorder["features"].append(feature)
  # Ajout liens
  for edge in graph.edges:
      start_node, end_node = edge   # nodes connectés au lien
      start_x, start_y = positions[start_node]  # x,y premier noeud
      end_x, end_y = positions[end_node]   # x,y second noeud
      start_x_echelle = start_x * echelle # mise à l'échelle
      end_x_echelle = end_x * echelle # mise à l'échelle
      start_y_inverted = image_height - start_y  # Inversion axe y
      end_y_inverted = image_height - end_y
      start_y_echelle = start_y_inverted * echelle # mise à l'échelle
      end_y_echelle = end_y_inverted * echelle # mise à l'échelle
      feature = {
          "type": "Feature",
          "geometry": {
              "type": "LineString",
              "coordinates": [[start_x_echelle, start_y_echelle], [end_x_echelle, end_y_echelle]]
          },
          "properties": {
              "geometryNamex": "Veine primaire",
              "_cost": 1
          }
      }
      blobRecorder["features"].append(feature)

  file_name = img_name[:-3] + "json"    # Nom de l'image +json
  script_dir = os.path.dirname(__file__)  # Chemin ou se trouve le script
  json_dir = os.path.join(script_dir,"JSON_Results") # Chemin du dossier JSON_Results
  full_path = os.path.join(json_dir, file_name)  # Emplacement du fichier JSON
  with open(full_path, "w") as outfile: # Ouverture du fichier en écriture
      json.dump(blobRecorder, outfile)
  return full_path


def blobDetection(img_name, data_dir):
  """
  This function predicts objects from a physarum polycephalum image and put the information in a JSON file.

  :param data_dir: This is the path to the directory directory
  :param img_name: This is the name of the image.
  """
  
  echelle = 1.28 # Mise à l'échelle par rapport a l'image blob recorder

  outputs, image = prediction(img_name, data_dir)  # Prediction using TorchScript Model
  nodes, positions, linkPositions = extractionOutputs(outputs)  # Extraction of outputs data
  listeLiensFinal = detectionLiens(positions, linkPositions)  # Link detection between Nodes
  graph = drawGraph(nodes, positions, listeLiensFinal, image)  # Conversion to NetworkX and Drawing
  file_json = conversionBlobRecorder(image,graph,positions,echelle,img_name)  # Conversion NetworkX to Blob Recorder

blobDetection("boite_41_1.png", "C:\\Users\\Marin HURE\\Desktop\\physarum")

