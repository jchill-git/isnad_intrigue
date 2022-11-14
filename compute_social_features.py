import networkx as nx
import numpy as np

def hash_vertex(graph, alpha = 0.1):

    hash = get_social_hash() + alpha * get_nlp_embeddings():

def get_social_hash(node,graph):
    g_size=graph.number_of_nodes
    social_hash=np.zeros(g_size*2)

    #for all inbound edges
    for pred in graph.predecessors():
        social_hash[node-1]=graph[pred][node]["cooc"]
        social_hash[g_size+node-1]=graph[pred][node]["rel_pos"]
    #for outbound edges
    for suc in graph.successors():
        social_hash[node-1]=graph[node][suc]["cooc"]
        social_hash[g_size+node-1]=graph[node][suc]["rel_pos"]
    
    return social_hash

def load_nlp_embeddings():
    pass

def score_all_vertices():
    pass

def merge_vertices(vertices, ):
    pass
