import networkx as nx
import numpy as np
import pandas as pd
import json

# Lire les fichiers JSON pour PRINCE2 et Scrum
with open('prince2.json', 'r') as f:
    prince2_data = json.load(f)

with open('scrum.json', 'r') as f:
    scrum_data = json.load(f)

# Définition des graphes PRINCE2 et Scrum
prince2_graph = nx.Graph()
scrum_graph = nx.Graph()

# Ajout des nœuds et des attributs pour PRINCE2
for element, attributes in prince2_data['PRINCE2'].items():
    prince2_graph.add_node(element, attributes=list(attributes.keys()))

# Ajout des relations fictives pour PRINCE2
prince2_edges = [
    ('BusinessCase', 'ProjectBoard'),
    ('ProjectBoard', 'ProjectPlan'),
    ('ProjectPlan', 'StagePlan'),
    ('StagePlan', 'WorkPackage'),
    ('WorkPackage', 'EndStageReport')
]
prince2_graph.add_edges_from(prince2_edges)

# Ajout des nœuds et des attributs pour Scrum
for element, attributes in scrum_data['Scrum'].items():
    scrum_graph.add_node(element, attributes=list(attributes.keys()))

# Ajout des relations fictives pour Scrum
scrum_edges = [
    ('ProductBacklog', 'Sprint'),
    ('Sprint', 'ScrumTeam'),
    ('ScrumTeam', 'SprintBacklog'),
    ('SprintBacklog', 'Increment'),
    ('Increment', 'DailyScrum')
]
scrum_graph.add_edges_from(scrum_edges)

# Fonction pour calculer la similarité des attributs des nœuds
def calculate_attribute_similarity(attr1, attr2):
    return len(set(attr1) & set(attr2)) / len(set(attr1) | set(attr2))

# Fonction pour calculer la similarité structurelle d'un nœud
def calculate_structural_similarity(graph1, graph2, node1, node2):
    # Récupérer les voisins des nœuds
    neighbors1 = set(graph1.neighbors(node1))
    neighbors2 = set(graph2.neighbors(node2))
    
    # Calculer la similarité des ensembles de voisins
    intersection_size = len(neighbors1 & neighbors2)
    union_size = len(neighbors1 | neighbors2)
    neighbor_similarity = intersection_size / union_size if union_size != 0 else 0
    
    # Calculer la similarité des attributs des nœuds
    attr_similarity = calculate_attribute_similarity(graph1.nodes[node1]['attributes'], graph2.nodes[node2]['attributes'])
    
    # Combiner les similarités (pondération à ajuster selon les besoins)
    return 0.5 * attr_similarity + 0.5 * neighbor_similarity

# Initialisation de la matrice de similarité (structurelle)
similarity_matrix = np.zeros((len(prince2_graph.nodes), len(scrum_graph.nodes)))

# Calcul des similarités structurelles des nœuds
prince2_nodes = list(prince2_graph.nodes())
scrum_nodes = list(scrum_graph.nodes())

for i, p_node in enumerate(prince2_nodes):
    for j, s_node in enumerate(scrum_nodes):
        similarity_matrix[i, j] = calculate_structural_similarity(prince2_graph, scrum_graph, p_node, s_node)

# Conversion de la matrice de similarité en DataFrame pour affichage
similarity_df = pd.DataFrame(similarity_matrix, index=prince2_nodes, columns=scrum_nodes)

# Affichage du tableau des correspondances structurelles
print(similarity_df)
