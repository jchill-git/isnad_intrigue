#given a collection of name embeddings and
# their original and disambiguated forms, this script
# will optimize the hyperparameters of a given model
# (whether k in the kNN models or the multiplier in the
# surface for heuristic models) so that the conll 2012 score
# is optimized
#
#Thoughts: perhaps just optimize b cubed? optimize p/r/f@k?
# Questions to be answered.
#
#Author: Ryan Muther, March 2021

import argparse
import re
import json
import sys
import os

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances,cosine_similarity
from sklearn.preprocessing import normalize
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix,lil_matrix
from scorch import scores
from cdlib import algorithms
import networkx as nx

from tqdm import tqdm

#finds the distance to the furthest surface for identical node
def findFurthestIdenticalSurfaceForm(nodeIndex,identicalNodes):
	#get the distances from the node we're working with to each 
	# other node with the same surface form
	otherNodes = embeddings.iloc[identicalNodes]
	nodeDists = euclidean_distances(np.array(embeddings.iloc[nodeIndex]).reshape(1, -1),otherNodes)[0]
	return np.max(nodeDists)

def makeGraph(nameEmbeddings,distTree,multiplier):	
	if args.kNN:
		print("Constructing adjacency matrix for k=%d"%multiplier)
	else:
		print("Constructing adjacency matrix for multiplier %f"%multiplier)

	#create a sparse matrix to store the adjacency data
	adjMatrix = lil_matrix((len(nameEmbeddings),len(nameEmbeddings)))
	#for every node, find all the nodes within (multiplier*radius)
	# and create an edge between that node and each other
	# returned node
	for i in tqdm(list(range(len(nameEmbeddings)))):
		idx = nameEmbeddings.iloc[i]["idx"]
		#get the embedding and normalize it so we're using
		# the normalized point for the query
		embedding = np.array(nameEmbeddings.iloc[i]["embedding"]).reshape(1, -1)
		
		#get the indices of nodes that also come from this document if we need them
		if args.noIntradocLinks:
			sameDocIndices = nameEmbeddings.iloc[i]["sameDocIndices"]
			sameDocCount = len(sameDocIndices)

		if args.normalize:
			embedding = normalize(embedding)

		#get the embeddings for nearby points
		if args.kNN:
			if args.noIntradocLinks:
				#add 1 to k to account for the fact that we'll
				# get the embedding itself back as a neighbor w\ dist 0
				#also hackily add a constant greater than the number of names in any document 
				# this really should be done by finding all the indices of embeddings in
				# the same document ahead of time, then discarding those
				nearbyPoints = [node for node in distTree.query(embedding,k=multiplier+1+sameDocCount)[1][0] if node != idx and node not in sameDocIndices][:multiplier]
				assert(len(nearbyPoints)==multiplier)
			else:
				#add 1 to k to account for the fact that we'll
				# get the embedding itself back as a neighbor w\ dist 0
				nearbyPoints = [node for node in distTree.query(embedding,k=multiplier+1)[1][0] if node != idx]
		else:
			radius = nameEmbeddings.iloc[i]["radius"]
			if args.noIntradocLinks:
				nearbyPoints = [node for node in distTree.query_ball_point(embedding,multiplier*radius)[0] if node != idx and node not in sameDocIndices]
			else:
				nearbyPoints = [node for node in distTree.query_ball_point(embedding,multiplier*radius)[0] if node != idx]
		if len(nearbyPoints)==0: 
			continue

		nearbyEmbeddings = embeddings.iloc[nearbyPoints]

		similarities = cosine_similarity(embedding,nearbyEmbeddings)[0]
		nearbyNames = [nameEmbeddings.iloc[j]["nameDisambiguated"] for j in nearbyPoints]
		#print(list(zip(nearbyNames,similarities))[:100])

		for j,otherIndex in enumerate(nearbyPoints):
			if args.unweighted:
				adjMatrix[idx,otherIndex] = 1.0
				adjMatrix[otherIndex,idx] = 1.0
			else:
				adjMatrix[idx,otherIndex] = similarities[j]
				adjMatrix[otherIndex,idx] = similarities[j]
	# if args.kNN:
	# 	print("Sparsity for k=%d: %f"%(multiplier,1-(adjMatrix.count_nonzero()/len(nameEmbeddings)**2)))
	# else:		
	# 	print("Sparsity for multiplier %f: %f"%(multiplier,1-(adjMatrix.count_nonzero()/len(nameEmbeddings)**2)))
	return adjMatrix.toarray()

parser = argparse.ArgumentParser()
parser.add_argument("embeddingDataset")
parser.add_argument("outfile")
parser.add_argument("--normalize",help="Compute distances in normalized embedding space",action="store_true")
parser.add_argument("--unweighted",help="Add edges with weight equal to the cosine similarity between nodes i and j instead of an unweighted edge",action="store_true")
parser.add_argument("--suffix",help="The identifier to add at the end of the output graph",default="")
parser.add_argument("--kNN",help="Add edges to the k nearest neighbors rather than the neighbors nearer than the distance to the furthest identical name",action="store_true")
parser.add_argument("--kMin",help="The starting number of neighbors to use for kNN network creation, default 5",default=5,type=int)
parser.add_argument("--kMax",help="The ending number of neighbors to use for kNN network creation, default 100",default=100,type=int)
parser.add_argument("--kStep",help="The increment in the number of neighbors to use for kNN network creation, default 5",default=5,type=int)
parser.add_argument("--noIntradocLinks",help="Constrains the netowork to not contain links from mentions in one document to mentions in the same document",action="store_true")
args = parser.parse_args()

#read in data
nameEmbeddings = pd.read_json(args.embeddingDataset,lines=True,encoding='utf8')

embeddingLen = len(nameEmbeddings["embedding"][0])

#filter out names that haven't been disambiguted
if "disambiguated" in nameEmbeddings.columns:
	nameEmbeddings = nameEmbeddings.query("disambiguated").reset_index().drop(columns="index")
nameEmbeddings["idx"] = list(range(len(nameEmbeddings)))
print("Constructing graph out of %d embeddings"%len(nameEmbeddings))

#get the gold standard communities
groupedByName = nameEmbeddings.drop(columns=["embedding"]).groupby("nameDisambiguated",as_index=False).agg(lambda x :list(x))
goldStandardCommunities = [set(r.mentionID) for r in groupedByName.itertuples()]

#expand embeddings into columns rather than single-column lists
embeddingCols = ["emb%d"%dim for dim in range(embeddingLen)]
embeddings = pd.DataFrame(nameEmbeddings["embedding"].to_list(), columns=embeddingCols)

#normalize the embeddings and compute euclidean distance.
# which is rank equivalent to cosine distance
if args.normalize:
	#print("Using normalized embeddings")
	embeddings = normalize(embeddings)

if not args.kNN:
	#compute the radius around each 
	#group the embeddings by the surface form of the mention to
	# facilitate computing the distance to the furthest identical name.
	#print("Computing %d node radii"%len(nameEmbeddings))
	embeddingsGrouped = nameEmbeddings.drop(columns=["embedding"]).groupby(by="name", as_index=False).agg(lambda x: list(x))
	embeddingsGrouped = embeddingsGrouped[["name","idx"]].rename(columns={"idx":"identicalIndices"})
	embeddingsGrouped["index"] = embeddingsGrouped["identicalIndices"]
	embeddingsGrouped["count"] = embeddingsGrouped.identicalIndices.apply(lambda x: len(x))
	embeddingsExploded = embeddingsGrouped.explode("index").drop(columns=["name"])
	#add the indices of the identical nodes to the name data
	nameEmbeddings = nameEmbeddings.merge(embeddingsExploded,left_on="idx",right_on="index")
	if "index_x" in nameEmbeddings.columns:
		nameEmbeddings = nameEmbeddings.drop(columns=["index_x","index_y"])
	#find the distance of each node to the most-distant
	# surface form identical mention. Apply() is way slower for some reason
	radii = []
	for data in nameEmbeddings.itertuples():
		radius = findFurthestIdenticalSurfaceForm(data.idx,data.identicalIndices)
		radii.append(radius)
	nameEmbeddings["radius"] = radii

	#problem: singleton names have no neighbors to use to get a distance
	# solution: find the average radius of low-but->1-count names 
	#(<5 is used here) and use that as an estimate
	rareNames = list(embeddingsGrouped.query("count>1 and count <5").name)
	rareEmbeddings = nameEmbeddings[nameEmbeddings['name'].isin(rareNames)]
	estimatedRadius = np.average(list(rareEmbeddings.radius))

	#subsitute zero-radius entries for the estimated average
	nameEmbeddings.loc[nameEmbeddings.radius == 0.0, ['radius']] = estimatedRadius
else:
	print("Constructing graph using nearest neighbors")

if args.noIntradocLinks:
	#group the embeddings by the document they're from to find the 
	# indices that each one shouldn't be linked to
	print("Finding same-document pairs")
	embeddingsGrouped = nameEmbeddings.drop(columns=["embedding"]).groupby(by="docID", as_index=False).agg(lambda x: list(x))
	embeddingsGrouped = embeddingsGrouped[["docID","idx"]].rename(columns={"idx":"sameDocIndices"})
	embeddingsGrouped["index"] = embeddingsGrouped["sameDocIndices"]
	embeddingsExploded = embeddingsGrouped.explode("index").drop(columns=["docID"])
	#add the indices of the same-document nodes to the name data
	nameEmbeddings = nameEmbeddings.merge(embeddingsExploded,left_on="idx",right_on="index")

#convert these distances into a graph
minMultiplier = 1
maxMultiplier = 2
step = .05
if args.kNN:
	minMultiplier = args.kMin
	maxMultiplier = args.kMax
	step = args.kStep
numPoints = int((maxMultiplier-minMultiplier)/step)+1

if args.normalize:
	distTree = cKDTree(embeddingsNormed)
else:
	distTree = cKDTree(embeddings)

outfile = open(args.outfile,"w",encoding="utf8")
outfile.write("algorithm,multiplier,score,communityCount\n")

communityAlgs = [("Leiden",algorithms.leiden),("Label Propagation",algorithms.label_propagation)]
#store the best score and hyperparameter for each algorithm
bestScores = dict([(a,-1) for a,f in communityAlgs])
bestMultipliers = dict([(a,-1) for a,f in communityAlgs])
for multiplier in np.linspace(minMultiplier,maxMultiplier,numPoints):
	#construct the graph for this value of the hyperparameter
	adjMatrix = makeGraph(nameEmbeddings,distTree,multiplier)
	#turn that into a networkx graph
	embeddingGraph = nx.convert_matrix.from_numpy_matrix(adjMatrix,create_using=nx.Graph)

	#for each of the algorithms we're working with, get their communities
	for algorithm,func in communityAlgs:
		#run the community detection on the graph
		coms = func(embeddingGraph).communities
		#convert node indices to mention ids for the corresponding nodes
		coms = [set([nameEmbeddings.loc[nodeIndex,"mentionID"] for nodeIndex in c]) for c in coms]
		comCount = len(coms)
		
		#score this algorithm's result
		score = scores.conll2012(goldStandardCommunities,coms)
		
		outfile.write("%s,%f,%f,%d\n"%(algorithm,multiplier,score,comCount))
		if score > bestScores[algorithm]:
			bestScores[algorithm] = score
			bestMultipliers[algorithm] = multiplier

for algorithm,func in communityAlgs:
	print("The optimal multiplier for %s is %f with score %f"%(algorithm,bestMultipliers[algorithm],bestScores[algorithm]))