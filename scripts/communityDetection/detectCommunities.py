#given a collection of name embeddings and
# a graph representing a network of those embeddings, 
# this script will run a simple label propagation community detection
# algorithm to find communities in the embedding set.
#
#This will output a json file of node ids and their assigned communities
# to the provided path
#
#Author: Ryan Muther, February 2021

import argparse
import sys
import os
import json

import pandas as pd
import networkx as nx
import numpy as np
from scipy.sparse import load_npz
from cdlib import algorithms

#read in the graph adjacency matrix
def readGraph():
	adjMatrix = load_npz(args.graphPath).toarray()
	embeddingGraph = nx.convert_matrix.from_numpy_matrix(adjMatrix,create_using=nx.Graph)
	print("Read %d-node graph with %d edges"%(len(embeddingGraph.nodes()),len(embeddingGraph.edges())))
	return embeddingGraph

parser = argparse.ArgumentParser()
parser.add_argument("--algorithm",required="True",choices=["surfaceForm","goldStandard","single","lp","leiden","leidenSurfaceForm","spinglass"], \
					help="The algorithm to be used to detect communities. Options are goldStandard (assigns embeddings to the correct cluster by disambiguated individual) surfaceForm (ignores the graph, simply clusters identical names) and labelPropagation.")
parser.add_argument("embeddingDataset",help="The embeddings used to construct the graph")
parser.add_argument("graphPath")
parser.add_argument("outfile")
args = parser.parse_args()

#read in data
nameEmbeddings = pd.read_json(args.embeddingDataset,lines=True,encoding='utf8')

#filter out names that haven't been disambiguted and add the node
# ID for each mention. Hopefully, the order is the same
# because there's no way to recover the node-to-matrix-index mapping 
# if they're not
if "disambiguated" in nameEmbeddings.columns:
	nameEmbeddings = nameEmbeddings.query("disambiguated").reset_index().drop(columns="index")
nameEmbeddings["nodeID"] = list(range(len(nameEmbeddings)))

print("Finding communities in %s"%args.graphPath)
if args.algorithm == "goldStandard":
	groupedByName = nameEmbeddings.groupby("nameDisambiguated",as_index=False).agg(lambda x :list(x))
	#extract communities from the data
	communities = [r.nodeID for r in groupedByName.itertuples()]
elif args.algorithm == "surfaceForm":
	groupedByName = nameEmbeddings.groupby("name",as_index=False).agg(lambda x :list(x))
	#extract communities from the data
	communities = [r.nodeID for r in groupedByName.itertuples()]
elif args.algorithm == "single":
	communities = [[r.nodeID for r in nameEmbeddings.itertuples()]]
elif args.algorithm == "lp":
	#read the graph
	embeddingGraph = readGraph()
	
	#create communities via label propagation
	lp_coms = algorithms.label_propagation(embeddingGraph)
	communities = lp_coms.communities
elif args.algorithm == "leiden":
	embeddingGraph = readGraph()
	
	leiden_coms = algorithms.leiden(embeddingGraph,weights="weight")
	communities = leiden_coms.communities
elif args.algorithm == "leidenSurfaceForm":
	#leiden community detection, but with the added constraint that
	# all surface form identical names begin in the same cluster
	embeddingGraph = readGraph()

	initial = nameEmbeddings.groupby("name",as_index=False).agg(lambda x :list(x))
	initial["communityID"] = list(range(len(initial)))
	initial = initial.explode("nodeID")
	initial = [r.communityID for r in initial.itertuples()]

	leiden_coms = algorithms.leiden(embeddingGraph,initial_membership=initial,weights="weight")
	communities = leiden_coms.communities
elif args.algorithm == "spinglass":
	embeddingGraph = readGraph()

	spinglass_coms = algorithms.spinglass(embeddingGraph)
	communities = spinglass_coms.communities

#map node ids to the community they're in so we don't
# have to search the communities later
communityDict = {}
for i in range(len(communities)):
	community = communities[i]
	#assign each mentionID to the index of the community it occurs in
	for nodeIndex in community:
		#some datasets have explicit mention ids if
		# an ner system is used to find mentions, others just have an id
		# such as when the documents are constructed from the names
		#(less than ideal, I know)
		if "mentionID" in nameEmbeddings.columns:
			communityDict[nameEmbeddings.loc[nodeIndex,"mentionID"]] = i
		else:
			communityDict[nameEmbeddings.loc[nodeIndex,"id"]] = i

print("Found %d communities"%len(communities))

#assign nodes to communities and save the data
print("Writing output")
f = open(args.outfile,"w",encoding="utf8")
for i in range(len(nameEmbeddings)):
	if "mentionID" in nameEmbeddings.columns:
		nodeID = nameEmbeddings.loc[i,"mentionID"]
	else:
		nodeID = nameEmbeddings.loc[i,"id"]
	#if the graph contains fewer nodes than the embeddings contain names
	# this will give an error. Perhaps ignore because for large datasets
	# we may want to only detect communities in subsets of the "full" 
	# dataset
	community = communityDict[nodeID]
	f.write(json.dumps({"nodeIndex":i,"mentionID":nodeID,"community":community})+"\n")
f.close()
print("Done")