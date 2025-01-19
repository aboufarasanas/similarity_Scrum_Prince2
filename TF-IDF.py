import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Charger le contenu du fichier JSON
file_path = 'metamodel.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Fonction pour calculer la similarité TF-IDF
def tfidf_similarity(str1, str2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([str1, str2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# Exemple de comparaison de classes entre PRINCE2 et Scrum
prince2_classes = data['PRINCE2']['Classes']
scrum_classes = data['Scrum']['Classes']

# Initialiser un DataFrame pour stocker les résultats de similarité
tfidf_results = pd.DataFrame(index=prince2_classes.keys(), columns=scrum_classes.keys())

# Comparer les classes de PRINCE2 avec celles de Scrum
for prince2_class in prince2_classes.keys():
    for scrum_class in scrum_classes.keys():
        tfidf_score = tfidf_similarity(prince2_class, scrum_class)
        tfidf_results.at[prince2_class, scrum_class] = tfidf_score

# Affichage des résultats dans un tableau
print("Résultats de similarité TF-IDF :")
print(tfidf_results)
