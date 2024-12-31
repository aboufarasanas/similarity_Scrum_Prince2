import pandas as pd

# Define the elements of each metamodel
scrum_elements = {
    "ProductBacklog": ["ID", "Name", "Description"],
    "Sprint": ["ID", "Name", "Goal", "StartDate", "EndDate"],
    "ScrumTeam": ["ID", "Name", "Members"],
    "SprintBacklog": ["ID", "Name", "Tasks"],
    "Increment": ["ID", "Description", "Version"],
    "DailyScrum": ["ID", "Date", "Notes"]
}
prince2_elements = {
    "BusinessCase": ["ID", "Title", "Description"],
    "ProjectBoard": ["ID", "Name", "Members"],
    "ProjectPlan": ["ID", "Name", "StartDate", "EndDate"],
    "StagePlan": ["ID", "Name", "StageObjective"],
    "WorkPackage": ["ID", "Name", "Tasks"],
    "EndStageReport": ["ID", "Name", "Summary"]
}
# Calculate Jaccard similarity


def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


# Create similarity matrix
similarity_matrix = {}
for scrum_element, scrum_attributes in scrum_elements.items():
    similarity_matrix[scrum_element] = {}
    for prince2_element, prince2_attributes in prince2_elements.items():
        similarity = jaccard_similarity(
            set(scrum_attributes), set(prince2_attributes))
        similarity_matrix[scrum_element][prince2_element] = similarity

# Convert similarity matrix to DataFrame
similarity_df = pd.DataFrame(similarity_matrix).T

# Define the actual matches for calculating metrics
actual_matches = [
    ("ProductBacklog", "BusinessCase"),
    ("Sprint", "StagePlan"),
    ("ScrumTeam", "ProjectBoard"),
    ("SprintBacklog", "WorkPackage"),
    ("Increment", "EndStageReport"),
    # No corresponding element for DailyScrum
]

# Function to calculate metrics


def calculate_metrics(matches, actual_matches):
    true_positives = len([match for match in matches if (
        match[0], match[1]) in actual_matches])
    false_positives = len([match for match in matches if (
        match[0], match[1]) not in actual_matches])
    false_negatives = len([match for match in actual_matches if (
        match[0], match[1]) not in [m[:2] for m in matches]])

    precision = true_positives / \
        (true_positives + false_positives) if (true_positives +
                                               false_positives) > 0 else 0
    recall = true_positives / \
        (true_positives + false_negatives) if (true_positives +
                                               false_negatives) > 0 else 0
    f_measure = (2 * precision * recall) / (precision +
                                            recall) if (precision + recall) > 0 else 0

    return precision, recall, f_measure


# Test multiple thresholds and find the best one
best_threshold = 0
best_f_measure = 0
metrics_results = []

for threshold in [i * 0.1 for i in range(1, 11)]:
    matches = []
    for scrum_element, similarities in similarity_matrix.items():
        for prince2_element, similarity in similarities.items():
            if similarity >= threshold:
                matches.append((scrum_element, prince2_element, similarity))

    precision, recall, f_measure = calculate_metrics(matches, actual_matches)
    metrics_results.append((threshold, precision, recall, f_measure))

    if f_measure > best_f_measure:
        best_threshold = threshold
        best_f_measure = f_measure

# Display best threshold and metrics
print(f"Best Threshold: {best_threshold}")
best_metrics = metrics_results[int(best_threshold * 10 - 1)]
print(f"Best Precision: {best_metrics[1]:.4f}")
print(f"Best Recall: {best_metrics[2]:.4f}")
print(f"Best F-measure: {best_metrics[3]:.4f}")

# Find matches with the best threshold
best_matches = []
for scrum_element, similarities in similarity_matrix.items():
    for prince2_element, similarity in similarities.items():
        if similarity >= best_threshold:
            best_matches.append((scrum_element, prince2_element, similarity))

# Display similarity matrix
print("\nSimilarity Matrix:")
print(similarity_df)

# Calculate max and mean similarities
max_similarity = similarity_df.max().max()
mean_similarity = similarity_df.mean().mean()

print(f"\nMaximum Similarity: {max_similarity:.4f}")
print(f"Mean Similarity: {mean_similarity:.4f}")

# Display matches with the best threshold
print("\nMatches (with threshold = {:.1f}):".format(best_threshold))
for match in best_matches:
    print(f"{match[0]} <-> {match[1]} : Similarity = {match[2]:.4f}")

# Display all metrics results
print("\nAll Metrics Results:")
for result in metrics_results:
    print(f"Threshold: {result[0]:.1f} - Precision: {result[1]
          :.4f}, Recall: {result[2]:.4f}, F-measure: {result[3]:.4f}")
