import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def extract_pubmed_data():
    # Load the PubMed dataset
    data = pd.read_json('MeSH/dev.json', lines=True)
    print(data.columns)

    # Create a graph based on the MeSH terms
    G = nx.Graph()
    for index, row in data.iterrows():
        #print(row)
        mesh_terms = row['label']
        print(mesh_terms)
        for i in range(len(mesh_terms)):
            if not G.has_node(mesh_terms[i]):
                G.add_node(mesh_terms[i])
            for j in range(i+1, len(mesh_terms)):
                if not G.has_node(mesh_terms[j]):
                    G.add_node(mesh_terms[j])
                if G.has_edge(mesh_terms[i], mesh_terms[j]):
                    G.edges[mesh_terms[i], mesh_terms[j]]['weight'] += 1
                else:
                    G.add_edge(mesh_terms[i], mesh_terms[j], weight=1)
    # Extract the edges and node features from the graph
    """
    nx.draw(G, with_labels=True)
    plt.savefig('plotgraph.png', dpi=300, bbox_inches='tight')
    plt.show()
    """
    edges = []
    node_features = []
    for i, node in enumerate(G.nodes()):
        node_features.append([0]*len(G.nodes()))
        node_features[i][i] = 1
        for neighbor in G.neighbors(node):
            if G.edges[node, neighbor]['weight'] > 0:
                edges.append([i, list(G.nodes()).index(neighbor)])

    return edges, node_features



#edges, node_features = extract_pubmed_data()
