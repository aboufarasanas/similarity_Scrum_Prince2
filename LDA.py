from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric, strip_short
import pandas as pd

# Étape 1 : Définir les éléments des deux métamodèles
prince2_elements = ['BusinessCase', 'ProjectBoard',
                    'ProjectPlan', 'StagePlan', 'WorkPackage', 'EndStageReport']
scrum_elements = ['ProductBacklog', 'Sprint', 'ScrumTeam',
                  'SprintBacklog', 'Increment', 'DailyScrum']

# Étape 2 : Prétraitement des données


def preprocess_elements(elements):
    CUSTOM_FILTERS = [strip_punctuation, strip_numeric, strip_short]
    return [preprocess_string(element, CUSTOM_FILTERS) for element in elements]


prince2_processed = preprocess_elements(prince2_elements)
scrum_processed = preprocess_elements(scrum_elements)

# Étape 3 : Créer le dictionnaire et la matrice bag-of-words
all_elements = prince2_processed + scrum_processed
dictionary = Dictionary(all_elements)
corpus = [dictionary.doc2bow(text) for text in all_elements]

# Étape 4 : Appliquer LDA pour générer des vecteurs de sujets
lda_model = LdaModel(corpus=corpus, id2word=dictionary,
                     num_topics=6, random_state=42)
topic_distributions = [lda_model.get_document_topics(
    bow, minimum_probability=0.0) for bow in corpus]

# Convertir les distributions en vecteurs
num_topics = lda_model.num_topics
vectors = [[prob for _, prob in doc] for doc in topic_distributions]

# Séparer les vecteurs en PRINCE2 et Scrum
prince2_vectors = vectors[:len(prince2_elements)]
scrum_vectors = vectors[len(prince2_elements):]

# Étape 5 : Calculer les similarités cosinus
similarity_matrix = cosine_similarity(prince2_vectors, scrum_vectors)

# Étape 6 : Afficher la matrice de similarité
similarity_df = pd.DataFrame(
    similarity_matrix, index=prince2_elements, columns=scrum_elements)
print("Matrice de similarité :\n", similarity_df)

# Étape 7 : Tester les métriques de qualité pour tous les seuils
thresholds = [i / 10 for i in range(1, 10)]
real_matches = [
    ('BusinessCase', 'ProductBacklog'),
    ('ProjectBoard', 'ScrumTeam'),
    ('ProjectPlan', 'SprintBacklog'),
    ('StagePlan', 'Sprint'),
    ('WorkPackage', 'Increment'),
    ('EndStageReport', 'DailyScrum'),
]

metrics_results = []

for threshold in thresholds:
    matches = []
    for i, prince2_elem in enumerate(prince2_elements):
        for j, scrum_elem in enumerate(scrum_elements):
            if similarity_matrix[i][j] >= threshold:
                matches.append((prince2_elem, scrum_elem))

    # Calcul des métriques
    true_positives = sum(1 for match in matches if match in real_matches)
    false_positives = len(matches) - true_positives
    false_negatives = len(real_matches) - true_positives

    precision = true_positives / len(matches) if matches else 0
    recall = true_positives / len(real_matches)
    f1_score = 2 * (precision * recall) / (precision +
                                           recall) if (precision + recall) > 0 else 0
    overall_accuracy = true_positives / len(real_matches)

    metrics_results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "overall_accuracy": overall_accuracy,
    })

# Afficher les résultats des métriques pour tous les seuils
metrics_df = pd.DataFrame(metrics_results)
print("\nMétriques pour chaque seuil :")
print(metrics_df)

# Trouver le seuil optimal
best_threshold = metrics_df.loc[metrics_df['f1_score'].idxmax()]
print("\nMeilleur seuil basé sur le F1-score :")
print(best_threshold)
