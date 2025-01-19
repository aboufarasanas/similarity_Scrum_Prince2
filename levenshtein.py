def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def normalized_similarity(s1, s2):
    distance = levenshtein_distance(s1.lower(), s2.lower())
    max_length = max(len(s1), len(s2))
    return 1 - (distance / max_length)


# Éléments des métamodèles
prince2_elements = ['BusinessCase', 'ProjectBoard',
                    'ProjectPlan', 'StagePlan', 'WorkPackage', 'EndStageReport']
scrum_elements = ['ProductBacklog', 'Sprint', 'ScrumTeam',
                  'SprintBacklog', 'Increment', 'DailyScrum']

# Calcul de toutes les similarités
similarities = []
for p2_elem in prince2_elements:
    for scrum_elem in scrum_elements:
        sim = normalized_similarity(p2_elem, scrum_elem)
        similarities.append((p2_elem, scrum_elem, sim))

# Tri des similarités par score
similarities.sort(key=lambda x: x[2], reverse=True)

# Affichage des résultats
print("\nMatrice de similarité :")
print("\nPRINCE2 Element | Scrum Element | Similarité")
print("-" * 50)
for p2_elem, scrum_elem, sim in similarities:
    print(f"{p2_elem:15} | {scrum_elem:12} | {sim:.3f}")

# Correspondances attendues pour le calcul des métriques
expected_matches = {
    ('ProjectPlan', 'Sprint'),
    ('WorkPackage', 'SprintBacklog'),
    ('BusinessCase', 'ProductBacklog'),
    ('EndStageReport', 'Increment'),
    ('ProjectBoard', 'ScrumTeam')
}

# Test avec différents seuils
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print("\nÉvaluation des seuils:")
print("\nSeuil | Précision | Rappel | F-mesure")
print("-" * 45)

for threshold in thresholds:
    # Trouver les correspondances avec le seuil
    matches = {(p2_elem, scrum_elem)
               for p2_elem, scrum_elem, sim in similarities if sim >= threshold}

    # Calculer les métriques
    true_positives = len(matches.intersection(expected_matches))
    false_positives = len(matches - expected_matches)
    false_negatives = len(expected_matches - matches)

    precision = true_positives / \
        (true_positives + false_positives) if (true_positives +
                                               false_positives) > 0 else 0
    recall = true_positives / \
        (true_positives + false_negatives) if (true_positives +
                                               false_negatives) > 0 else 0
    f_measure = 2 * (precision * recall) / (precision +
                                            recall) if (precision + recall) > 0 else 0

    print(f"{threshold:.1f}   | {precision:.3f}    | {
          recall:.3f}  | {f_measure:.3f}")
