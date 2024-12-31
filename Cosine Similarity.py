import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Étape 1 : Définition des éléments des métamodèles
scrum_elements = [
    "ProductBacklog", "Sprint", "ScrumTeam", "SprintBacklog", "Increment", "DailyScrum"
]

prince2_elements = [
    "BusinessCase", "ProjectBoard", "ProjectPlan", "StagePlan", "WorkPackage", "EndStageReport"
]

# Étape 2 : Vecteurs d'attributs
scrum_vectors = np.array([
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # ProductBacklog
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # Sprint
    [1, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # ScrumTeam
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # SprintBacklog
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # Increment
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 1],  # DailyScrum
])

prince2_vectors = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # BusinessCase
    [1, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # ProjectBoard
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # ProjectPlan
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # StagePlan
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # WorkPackage
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # EndStageReport
])

# Étape 3 : Calcul de la similarité cosinus
similarity_matrix = cosine_similarity(scrum_vectors, prince2_vectors)


# Étape 4 : Correspondances réelles pour l'évaluation
true_matches = {
    "ProductBacklog": "BusinessCase",
    "Sprint": "ProjectPlan",
    "ScrumTeam": "ProjectBoard",
    "SprintBacklog": "WorkPackage",
    "Increment": "EndStageReport",
    "DailyScrum": "StagePlan"
}

# Étape 5 : Fonction pour évaluer les seuils et calculer les métriques de qualité


def evaluate_threshold(threshold, similarity_matrix, true_matches, scrum_elements, prince2_elements):
    predicted_matches = {}
    for i, scrum_entity in enumerate(scrum_elements):
        for j, prince2_entity in enumerate(prince2_elements):
            if similarity_matrix[i][j] >= threshold:
                predicted_matches[scrum_entity] = prince2_entity
                break

    tp = sum(1 for k, v in predicted_matches.items()
             if true_matches.get(k) == v)
    total_predicted = len(predicted_matches)
    total_actual = len(true_matches)

    precision = tp / total_predicted if total_predicted > 0 else 0
    recall = tp / total_actual if total_actual > 0 else 0
    f_measure = 2 * (precision * recall) / (precision +
                                            recall) if (precision + recall) > 0 else 0

    return precision, recall, f_measure


# Étape 6 : Tester différents seuils
thresholds = np.linspace(0, 1, 100)
results = [evaluate_threshold(th, similarity_matrix, true_matches,
                              scrum_elements, prince2_elements) for th in thresholds]

# Étape 7 : Trouver le meilleur seuil
best_threshold_index = np.argmax([result[2] for result in results])
best_threshold = thresholds[best_threshold_index]
best_precision, best_recall, best_f_measure = results[best_threshold_index]

# Étape 8 : Calcul des valeurs maximales et moyennes des similarités
similarities = similarity_matrix.flatten()
max_similarity = np.max(similarities)
average_similarity = np.mean(similarities)

# Affichage des résultats
print("Tableau de similarité (cosinus) :")
print(similarity_matrix)
print(f"Meilleur seuil : {best_threshold:.3f}")
print(f"Précision : {best_precision:.3f}")
print(f"Rappel : {best_recall:.3f}")
print(f"F-mesure : {best_f_measure:.3f}")
print(f"Valeur maximale de similarité : {max_similarity:.3f}")
print(f"Valeur moyenne de similarité : {average_similarity:.3f}")
