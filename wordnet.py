import json
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

# Télécharger les ressources nécessaires de NLTK
nltk.download('punkt')
nltk.download('wordnet')

def get_wordnet_synsets(word):
    return wn.synsets(word)

def compute_similarity(word1, word2):
    synsets1 = get_wordnet_synsets(word1)
    synsets2 = get_wordnet_synsets(word2)
    max_similarity = 0
    for synset1 in synsets1:
        for synset2 in synsets2:
            similarity = synset1.wup_similarity(synset2)
            if similarity is not None and similarity > max_similarity:
                max_similarity = similarity
    return max_similarity

def calculate_similarity(prince2_elements, scrum_elements):
    similarity_matrix = {}
    for pr_key, pr_values in prince2_elements.items():
        similarity_matrix[pr_key] = {}
        for sc_key, sc_values in scrum_elements.items():
            avg_similarity = 0
            count = 0
            for pr_value in pr_values:
                for sc_value in sc_values:
                    similarity = compute_similarity(pr_value, sc_value)
                    if similarity is not None:
                        avg_similarity += similarity
                        count += 1
            if count > 0:
                avg_similarity /= count
            similarity_matrix[pr_key][sc_key] = avg_similarity
    return similarity_matrix

# Charger les éléments PRINCE2 et Scrum à partir des fichiers JSON
with open('prince2.json', 'r') as file:
    prince2_data = json.load(file)["PRINCE2"]

with open('scrum.json', 'r') as file:
    scrum_data = json.load(file)["Scrum"]

# Extraire les noms des éléments de PRINCE2 et Scrum
prince2_elements = {key: list(value.keys()) for key, value in prince2_data.items()}
scrum_elements = {key: list(value.keys()) for key, value in scrum_data.items()}

# Calculer la similarité
similarity_matrix = calculate_similarity(prince2_elements, scrum_elements)
print("Similarité entre PRINCE2 et Scrum :")
print("PRINCE2 \\ Scrum |", " | ".join(scrum_elements.keys()))
for pr_key, sc_similarities in similarity_matrix.items():
    print(f"{pr_key:<20} |", " | ".join(f"{sc_similarities.get(sc_key, 0):.2f}" for sc_key in scrum_elements.keys()))
