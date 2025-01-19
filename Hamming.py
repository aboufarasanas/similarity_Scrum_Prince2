import json
import pandas as pd
from scipy.spatial.distance import hamming

# Charger le contenu du fichier JSON
file_path = 'metamodel.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Fonction pour calculer la similarité Hamming
def hamming_similarity(str1, str2):
    # Normaliser les chaînes à la même longueur
    max_len = max(len(str1), len(str2))
    str1 = str1.ljust(max_len)
    str2 = str2.ljust(max_len)
    return 1 - hamming(list(str1), list(str2))

# Exemple de comparaison de classes entre PRINCE2 et Scrum
prince2_classes = data['PRINCE2']['Classes']
scrum_classes = data['Scrum']['Classes']

# Initialiser un DataFrame pour stocker les résultats de similarité
hamming_results = pd.DataFrame(index=prince2_classes.keys(), columns=scrum_classes.keys())

# Comparer les classes de PRINCE2 avec celles de Scrum
for prince2_class in prince2_classes.keys():
    for scrum_class in scrum_classes.keys():
        hamming_score = hamming_similarity(prince2_class, scrum_class)
        hamming_results.at[prince2_class, scrum_class] = hamming_score

# Affichage des résultats dans un tableau
print("Résultats de similarité Hamming :")
print(hamming_results)
