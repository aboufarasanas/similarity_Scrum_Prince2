import json
import pandas as pd
from difflib import SequenceMatcher
import numpy as np

# Charger le contenu du fichier JSON
file_path = 'metamodel.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Fonction pour calculer la similarité de Monge-Elkan
def monge_elkan_similarity(str1, str2):
    words1 = str1.split()
    words2 = str2.split()
    
    max_similarities = []
    
    for word1 in words1:
        max_similarity = 0
        for word2 in words2:
            similarity = SequenceMatcher(None, word1, word2).ratio()
            max_similarity = max(max_similarity, similarity)
        max_similarities.append(max_similarity)
    
    return np.mean(max_similarities) if max_similarities else 0

# Exemple de comparaison de classes entre PRINCE2 et Scrum
prince2_classes = data['PRINCE2']['Classes']
scrum_classes = data['Scrum']['Classes']

# Initialiser un DataFrame pour stocker les résultats de similarité
monge_elkan_results = pd.DataFrame(index=prince2_classes.keys(), columns=scrum_classes.keys())

# Comparer les classes de PRINCE2 avec celles de Scrum
for prince2_class in prince2_classes.keys():
    for scrum_class in scrum_classes.keys():
        similarity_score = monge_elkan_similarity(prince2_class, scrum_class)
        monge_elkan_results.at[prince2_class, scrum_class] = similarity_score

# Affichage des résultats dans un tableau
print("Résultats de similarité de Monge-Elkan :")
print(monge_elkan_results.to_string(index=True))
