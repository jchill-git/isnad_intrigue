#given a collection of name embeddings and
# their original and disambiguated forms, this script
# will create a distance matrix between all embeddings
# by first normalizing the embeddings than computing euclidean
# distance. 
#
# Then it will turn the distance matrix into a graph where each
# node n is connected by an edge to another node n' if n' is within
# some multiple of the distance between n and the most-distant
# surface-form-identical embedding.
#
#Author: Ryan Muther, January 2021

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
from scipy.sparse import save_npz,csr_matrix,lil_matrix,load_npz

from tqdm import tqdm

#finds the distance to the furthest surface for identical node
def findFurthestIdenticalSurfaceForm(nodeIndex,identicalNodes):
	#get the distances from the node we're working with to each 
	# other node with the same surface form
	otherNodes = embeddings.iloc[identicalNodes]
	nodeDists = euclidean_distances(np.array(embeddings.iloc[nodeIndex]).reshape(1, -1),otherNodes)[0]
	return np.max(nodeDists)

parser = argparse.ArgumentParser()
parser.add_argument("embeddingDataset")
parser.add_argument("outfile")
parser.add_argument("--multiplier",help="The multiplier to use for the furthest surface form heuristic",type=float,default=None)
parser.add_argument("--normalize",help="Compute distances in normalized embedding space",action="store_true")
parser.add_argument("--unweighted",help="Add edges with weight equal to the cosine similarity between nodes i and j instead of an unweighted edge",action="store_true")
parser.add_argument("--suffix",help="The identifier to add at the end of the output graph",default="")
parser.add_argument("--sample",help="A json dataset of mentions to embed, with identical ids to a subset of the mentions in the embedding dataset. Will also cause the program to write the sampled data to the same directory as the graph, for ease of later use. If not given, will use all the embeddings in the input.",default=None)
parser.add_argument("--kNN",help="Add edges to the k nearest neighbors rather than the neighbors nearer than the distance to the furthest identical name",action="store_true")
parser.add_argument("--kMin",help="The starting number of neighbors to use for kNN network creation, default 5",default=5,type=int)
parser.add_argument("--kMax",help="The ending number of neighbors to use for kNN network creation, default 100",default=100,type=int)
parser.add_argument("--kStep",help="The increment in the number of neighbors to use for kNN network creation, default 5",default=5,type=int)
parser.add_argument("--noIntradocLinks",help="Constrains the netowork to not contain links from mentions in one document to mentions in the same document",action="store_true")
args = parser.parse_args()

#read in data
nameEmbeddings = pd.read_json(args.embeddingDataset,lines=True,encoding='utf8')

embeddingLen = len(nameEmbeddings["embedding"][0])

#if we only want to embed some of the mentions in the embeddings, 
# filter the dataset to contain only those
if args.sample:
	print("Reading sample data from %s"%args.sample)
	if os.path.isfile(args.sample):
		sample = [json.loads(line) for line in open(args.sample,"r",encoding="utf8").readlines()]	
	else:
		sample = []
		for f in sorted(os.listdir(args.sample)):
			if f.endswith(".json"):
				sample += [json.loads(line) for line in open(os.path.join(args.sample,f),"r",encoding="utf8").readlines()]
	IDs = [s["id"] for s in sample if s["id"]!=""]
	print("Read %d samples"%len(IDs))

	nameEmbeddings = nameEmbeddings[nameEmbeddings.id.isin(IDs)].reset_index().drop(columns="index")
	assert(len(nameEmbeddings)==len(IDs))

	#write the dataset of mentions used in this graph to the output directory
	dataPath = open(os.path.join(args.outfile,args.suffix+"_data.json"),"w",encoding="utf8")
	nameEmbeddings.to_json(dataPath, force_ascii=False, orient='records', lines=True)
	# for data in nameEmbeddings.iterrows():
	# 	dataPath.write(json.dumps(dict(data),ensure_ascii=False)+"\n")
	print("Wrote sample data to %s"%os.path.join(args.outfile,args.suffix+"_data.json"))

#filter out names that haven't been disambiguted
if "disambiguated" in nameEmbeddings.columns:
	nameEmbeddings = nameEmbeddings.query("disambiguated").reset_index().drop(columns="index")
nameEmbeddings["idx"] = list(range(len(nameEmbeddings)))
print("Constructing graph out of %d embeddings"%len(nameEmbeddings))


#expand embeddings into columns rather than lists
embeddingCols = ["emb%d"%dim for dim in range(embeddingLen)]
embeddings = pd.DataFrame(nameEmbeddings["embedding"].to_list(), columns=embeddingCols)

#normalize the embeddings and compute euclidean distance.
# which is rank equivalent to cosine distance
if args.normalize:
	print("Using normalized embeddings")
	embeddings = normalize(embeddings)
else:
	print("Using unnormalized embeddings")

if not args.kNN:
	#group the embeddings by the surface form of the mention to
	# facilitate computing the distance to the furthest identical name.
	print("Computing %d node radii"%len(nameEmbeddings))
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
	for data in tqdm(nameEmbeddings.itertuples()):
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
#this might take a long time. Throw the supercomputer at it?
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html
print("Constructing KDTree")
if not args.multiplier:
	minMultiplier = 1
	maxMultiplier = 2
	step = 1
	if args.kNN:
		minMultiplier = args.kMin
		maxMultiplier = args.kMax
		step = args.kStep
	multipliers = range(minMultiplier,maxMultiplier+1,step)
else:
	multipliers = [args.multiplier] 

if args.normalize:
	distTree = cKDTree(embeddingsNormed)
else:
	distTree = cKDTree(embeddings)
for multiplier in multipliers:
	#construct the filename for the output file
	if args.kNN:
		print("Constructing adjacency matrix for k=%d"%multiplier)
		filename = "kNN%d"%multiplier
	else:
		print("Constructing adjacency matrix for multiplier %f"%multiplier)
		filename = "multiplier%d"%multiplier
	if args.suffix != "":
		filename += "_"+args.suffix
	 
	#skip this multiplier if we already computed its graph
	graphPath = os.path.join(args.outfile,filename)
	if os.path.exists(graphPath+".npz"):
		mat = load_npz(graphPath+".npz")
		sparsity = 1.0 - (np.count_nonzero(mat.toarray()) / float(mat.toarray().size))
		print("Found existing matrix with %f sparsity"%sparsity)
		continue

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
			#add a tiny extra amount to link the furthest node,
			# because its unclear to me how that's handled
			radius += .000001
			if args.noIntradocLinks:
				nearbyPoints = [node for node in distTree.query_ball_point(embedding,multiplier*radius)[0] if node != idx and node not in sameDocIndices]
			else:
				nearbyPoints = [node for node in distTree.query_ball_point(embedding,multiplier*radius)[0] if node != idx]
		#print(len(nearbyPoints))
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
	if args.kNN:
		print("Sparsity for k=%d: %f"%(multiplier,1-(adjMatrix.count_nonzero()/len(nameEmbeddings)**2)))
	else:		
		print("Sparsity for multiplier %f: %f"%(multiplier,1-(adjMatrix.count_nonzero()/len(nameEmbeddings)**2)))
	#convert the matrix to a sparse representation for saving
	with open(graphPath+".npz", 'wb') as f:
		save_npz(f,csr_matrix(adjMatrix))
	print("Wrote adjacency matrix for multiplier %f to %s"%(multiplier,graphPath+".mpz"))