from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Initialize the Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example elements from Scrum and PRINCE2
scrum_elements = [
    "Product Backlog", "Sprint", "Sprint Backlog", "Daily Scrum", "Product Increment"
]

prince2_elements = [
    "Product Description", "Stage Plan", "Work Packages", "Daily Stand-up", "Project Deliverables"
]

# Step 1: Compute the SBERT similarity
similarities = np.zeros((len(scrum_elements), len(prince2_elements)))

for i, scrum_item in enumerate(scrum_elements):
    for j, prince2_item in enumerate(prince2_elements):
        scrum_embedding = model.encode(scrum_item)
        prince2_embedding = model.encode(prince2_item)
        similarity = np.dot(scrum_embedding, prince2_embedding) / (np.linalg.norm(scrum_embedding) * np.linalg.norm(prince2_embedding))
        similarities[i, j] = similarity

# Step 2: Define True Correspondences
true_correspondences = [
    (0, 0),  # "Product Backlog" <-> "Product Description"
    (1, 1),  # "Sprint" <-> "Stage Plan"
    (2, 2),  # "Sprint Backlog" <-> "Work Packages"
    (3, 3),  # "Daily Scrum" <-> "Daily Stand-up"
    (4, 4),  # "Product Increment" <-> "Project Deliverables"
]

# Step 3: Predict Correspondences based on a threshold for SBERT similarity
threshold = 0.7  # Define a similarity threshold for prediction
predicted_correspondences = []

for i in range(len(scrum_elements)):
    for j in range(len(prince2_elements)):
        if similarities[i, j] > threshold:
            predicted_correspondences.append((i, j))

# Step 4: Calculate Precision, Recall, and F1-score
# Flatten the true and predicted correspondences to compare them
true_labels = np.zeros((len(scrum_elements), len(prince2_elements)))
for true_corr in true_correspondences:
    true_labels[true_corr] = 1

predicted_labels = np.zeros_like(true_labels)
for pred_corr in predicted_correspondences:
    predicted_labels[pred_corr] = 1

# Flatten the matrices to compare them easily
true_labels_flat = true_labels.flatten()
predicted_labels_flat = predicted_labels.flatten()

# Calculate metrics
precision = precision_score(true_labels_flat, predicted_labels_flat)
recall = recall_score(true_labels_flat, predicted_labels_flat)
f1 = f1_score(true_labels_flat, predicted_labels_flat)

# Display results
print(f"SBERT Similarities: \n{similarities}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
