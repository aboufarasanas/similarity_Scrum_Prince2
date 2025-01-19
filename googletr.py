import json
from googletrans import Translator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Step 1: Load PRINCE2 and Scrum JSON data
prince2_data = {
    "PRINCE2": {
        "BusinessCase": {"ID": "String", "Title": "String", "Description": "String"},
        "ProjectBoard": {"ID": "String", "Name": "String", "Members": "List"},
        "ProjectPlan": {"ID": "String", "Name": "String", "StartDate": "Date", "EndDate": "Date"},
        "StagePlan": {"ID": "String", "Name": "String", "StageObjective": "String"},
        "WorkPackage": {"ID": "String", "Name": "String", "Tasks": "List"},
        "EndStageReport": {"ID": "String", "Name": "String", "Summary": "String"}
    }
}

scrum_data = {
    "Scrum": {
        "ProductBacklog": {"ID": "String", "Name": "String", "Description": "String"},
        "Sprint": {"ID": "String", "Name": "String", "Goal": "String", "StartDate": "Date", "EndDate": "Date"},
        "ScrumTeam": {"ID": "String", "Name": "String", "Members": "List"},
        "SprintBacklog": {"ID": "String", "Name": "String", "Tasks": "List"},
        "Increment": {"ID": "String", "Name": "String", "Description": "String", "Version": "String"},
        "DailyScrum": {"ID": "String", "Date": "Date", "Notes": "String"}
    }
}

# Step 2: Initialize Translator
translator = Translator()

# Function to concatenate the keys and descriptions
def translate_and_concatenate(data, target_language='en'):
    combined_texts = []
    for key, value in data.items():
        translated_key = translator.translate(key, dest=target_language).text
        if isinstance(value, dict):
            combined_text = translated_key + ' ' + ' '.join([str(translator.translate(str(v), dest=target_language).text) for v in value.values()])
            combined_texts.append(combined_text)
        else:
            combined_texts.append(translated_key)
    return combined_texts

# Translate and concatenate PRINCE2 and Scrum data
prince2_combined = translate_and_concatenate(prince2_data['PRINCE2'], 'en')
scrum_combined = translate_and_concatenate(scrum_data['Scrum'], 'en')

# Step 3: Vectorization and Cosine Similarity Calculation
vectorizer = TfidfVectorizer()

# Combine both sets of texts into one vector space for comparison
all_texts = prince2_combined + scrum_combined

# Fit TF-IDF on all texts and then compare
tfidf_matrix = vectorizer.fit_transform(all_texts)
cosine_sim = cosine_similarity(tfidf_matrix[:len(prince2_combined)], tfidf_matrix[len(prince2_combined):])

# Step 4: Generate a DataFrame for the similarity report
prince2_keys = list(prince2_data['PRINCE2'].keys())
scrum_keys = list(scrum_data['Scrum'].keys())

similarity_df = pd.DataFrame(cosine_sim, index=prince2_keys, columns=scrum_keys)

# Display the similarity results
print("Similarity Report:")
print(similarity_df)
