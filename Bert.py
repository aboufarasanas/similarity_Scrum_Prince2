import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

# Charger le contenu du fichier JSON
file_path = 'metamodel.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Fonction pour calculer la similarité BERT
def bert_similarity(str1, str2):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    inputs1 = tokenizer(str1, return_tensors='pt')
    inputs2 = tokenizer(str2, return_tensors='pt')

    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    # Obtenir les embeddings de la première token
    embedding1 = outputs1.last_hidden_state[:, 0, :].numpy()
    embedding2 = outputs2.last_hidden_state[:, 0, :].numpy()

    # Calculer la similarité cosinus
    cos_sim = cosine_similarity(embedding1, embedding2)
    return cos_sim[0][0]

# Exemple de comparaison de classes entre PRINCE2 et Scrum
prince2_classes = data['PRINCE2']['Classes']
scrum_classes = data['Scrum']['Classes']

# Initialiser un DataFrame pour stocker les résultats de similarité
bert_results = pd.DataFrame(index=prince2_classes.keys(), columns=scrum_classes.keys())

# Comparer les classes de PRINCE2 avec celles de Scrum
for prince2_class in prince2_classes.keys():
    for scrum_class in scrum_classes.keys():
        bert_score = bert_similarity(prince2_class, scrum_class)
        bert_results.at[prince2_class, scrum_class] = bert_score

# Affichage des résultats dans un tableau
print("Résultats de similarité BERT :")
print(bert_results)
