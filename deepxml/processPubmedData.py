import pandas as pd
import networkx as nx
#import pubmed_parser as pp
from elasticsearch import Elasticsearch
#from elasticsearch import Elasticsearch, RequestsHttpConnection, serializer, compat, exceptions, helpers

def connect_elasticsearch():
    _es = None
    _es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    if _es.ping():
        print('Yay Connect')
    else:
        print('Awww it could not connect!')
    return _es


def extract_pubmed_data():
    # Load the PubMed dataset
    data = pd.read_json('MeSH/dev.json', lines=True)
    print(data.columns)
    return
    # Create a graph based on the MeSH terms
    G = nx.Graph()
    for index, row in data.iterrows():
        print(row)
        mesh_terms = row['MeSH Terms'].split('; ')
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
    edges = []
    node_features = []
    for i, node in enumerate(G.nodes()):
        node_features.append([0]*len(G.nodes()))
        node_features[i][i] = 1
        for neighbor in G.neighbors(node):
            if G.edges[node, neighbor]['weight'] > 0:
                edges.append([i, list(G.nodes()).index(neighbor)])

    return edges, node_features


connect_elasticsearch()

extract_pubmed_data()
# path = r'D:\UNT\PhD\Medline\Medline_data\all_gz'
# dicts_out = pp.parse_pubmed_references(path) # return list of dictionary

#dict_out = pp.parse_pubmed_xml(path)

#dicts_out = pp.parse_medline_xml('data/medline16n0902.xml.gz',
#                                 year_info_only=False,
#                                 nlm_category=False,
#                                 author_list=False,
#                                 reference_list=False) # return list of dictionary
