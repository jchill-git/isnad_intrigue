#given a collection of name embeddings, antecedent information
# and a network of those embeddings, this script creates a set of 
# training data for a classifier to learn how to classify a pair
# of embeddings (x1,x2) to determine if x2 is an antecedent of x1
#
#We also include a feature representing if the mentions
# x1 and x2 have the same surface form. Each of these is separate fields
# and will need to be concatenated via torch.cat or similar
#
#Author: Ryan Muther, May 2021

import argparse
import re
import json
import sys
from tqdm import tqdm
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz
import numpy as np
import networkx as nx

def readGraph(path):
	adjMatrix = load_npz(path).toarray()
	embeddingGraph = nx.convert_matrix.from_numpy_matrix(adjMatrix,create_using=nx.Graph)
	print("Read %d-node graph with %d edges"%(len(embeddingGraph.nodes()),len(embeddingGraph.edges())))
	return embeddingGraph

#given the id of a mention, return its index in the dataframe (and the network)
def ID2Index(ID):
	return int(nameEmbeddings[nameEmbeddings.mentionID==ID].index.tolist()[0])

#TODO: Condense these
def getLocation(embeddings,index):
	return embeddings["tokenStart"][index]

def getDocID(embeddings,index):
	return embeddings["docID"][index]

def getID(embeddings,index):
	return embeddings["mentionID"][index]

def getEmbedding(embeddings,index):
	return embeddings["embedding"][index]

def getName(embeddings,index):
	return embeddings["name"][index]

#checks if a paper1 is earlier than paper2 
# Either the year is earlier or the year is the same
# and the id (which is the DOI+a number) is earlier
#
#NB for isnad data this will need to be reworked as isnads aren't
# necessarily ordered by id
def isEarlier(id1,id2,loc1,loc2):
	return ((loc1<loc2) or (loc1==loc2 and id1<id2))

parser = argparse.ArgumentParser()
parser.add_argument("--training",help="If given, add all possible antecedents as training examples, even if they're not neighbors in the network. Has no effect if kNearest is also provided",action="store_true")
parser.add_argument("--kNearest",help="If given, add only the k nearest positive and negative examples for each mention. If given, the network is ignored entirely",action="store_true")
parser.add_argument("--k",type=int,default=5)
parser.add_argument("embeddings")
parser.add_argument("antecedents")
parser.add_argument("network")
parser.add_argument("outfile")
args = parser.parse_args()

if args.kNearest:
	print("Selecting examples by cosine similarity, using top %d pos and neg examples"%args.k)
else:
	print("Selecting examples via network adjacency")

#read in data
nameEmbeddings = pd.read_json(args.embeddings,lines=True,encoding='utf8')
antecedents = pd.read_json(args.antecedents,lines=True,encoding='utf8')
network = readGraph(args.network)

#remove still ambiguous embeddings
if "disambiguated" in nameEmbeddings.columns:
	nameEmbeddings = nameEmbeddings.query("disambiguated").reset_index().drop(columns="index")

#for each embedding in the set of embeddings, construct
# classification examples
posExampleCount = 0
negExampleCount = 0
outfile = open(args.outfile,"w",encoding="utf8")
for dataIndex in tqdm(list(nameEmbeddings.index)):
	#get the ID of this mention
	ID = nameEmbeddings["mentionID"][dataIndex]
	loc = getLocation(nameEmbeddings,dataIndex)
	docID = getDocID(nameEmbeddings,dataIndex)
	embedding = getEmbedding(nameEmbeddings,dataIndex)
	name = getName(nameEmbeddings,dataIndex)
	#get the antecedent information for this mention
	antecedentData = antecedents[antecedents.id==ID].reset_index()

	#get the antecedents for this mention
	#remove antecedents that are from the same paper (i.e. whose ids
	# contain the document id of the current mention
	#TODO: Find cases where we miss all antecedents entirely
	# figure out how to deal with those
	correctAntecedents = [a for a in antecedentData["antecedents"][0] if docID not in a and a != "NIL"]
	correctAntecedentIndices = nameEmbeddings[nameEmbeddings.mentionID.isin(correctAntecedents)].index.tolist()

	#if we're using the network as a way of selecting the instances,
	# get the id of the 
	if not args.kNearest:
		#get the id of the node in the network for this mention
		# and the ids of its neighbors
		networkIndex = antecedentData["index"][0]
		neighbors = [n for n in network.neighbors(networkIndex) if getLocation(nameEmbeddings,n) != -1]
		neighbors = [n for n in neighbors if isEarlier(getID(nameEmbeddings,n),ID,getLocation(nameEmbeddings,n),loc)]
		neighborIDs = [getID(nameEmbeddings,n) for n in neighbors]

		if args.training:
			potentialAntecedents = list(set(correctAntecedentIndices).union(set(neighbors)))
		else:
			potentialAntecedents = neighbors
	#if we're using the k nearest selection method, do that
	# this can be sped up by caching the similarities ahead of time
	else:
		#find all the mentions that are earlier than the current one we're looking at
		earlierMentionIndices = [i for i in list(nameEmbeddings.index) if isEarlier(getID(nameEmbeddings,i),ID,getLocation(nameEmbeddings,i),loc)]

		#collect all the embeddings of earlier mentions
		earlierMentionEmbeddings = [getEmbedding(nameEmbeddings,i) for i in earlierMentionIndices]
		#deduplicate the embeddings
		uniqueEmbeddings = []
		uniqueEmbeddingIndices = []
		for i in range(len(earlierMentionEmbeddings)):
			if earlierMentionEmbeddings[i] not in uniqueEmbeddings:
				uniqueEmbeddings.append(earlierMentionEmbeddings[i])
				uniqueEmbeddingIndices.append(earlierMentionIndices[i])
		
		earlierMentionIndices = uniqueEmbeddingIndices
		earlierMentionEmbeddings = uniqueEmbeddings

		if len(earlierMentionEmbeddings) > 0:

			if len(earlierMentionEmbeddings) == 1:
				earlierMentionEmbeddings = np.array(earlierMentionEmbeddings).reshape(1, -1)
		
			#compute the similarity between this and all earlier mentions
			similarities = cosine_similarity(np.array(embedding).reshape(1, -1),earlierMentionEmbeddings)[0]
			#pair each similarity with the index of the corresponding mention and sort by similarity in descending order
			indicesBySimilarity = sorted(list(zip(earlierMentionIndices,similarities)),key=lambda x: x[1],reverse=True)
			
			#find the k nearest (distinct) positive and negative examples
			posExamples = []
			negExamples = []
			correctID = nameEmbeddings["nameDisambiguated"][dataIndex]
			for index,similarity in indicesBySimilarity:
				otherID = nameEmbeddings["nameDisambiguated"][index]
				if len(posExamples) < args.k and otherID == correctID:
					posExamples.append(index)
				elif len(negExamples) < args.k and otherID != correctID:
					negExamples.append(index)
				#break if we've finished finding k positive and negative examples
				if len(posExamples) == args.k and len(negExamples) == args.k:
					break

			potentialAntecedents = posExamples + negExamples
		else:
			potentialAntecedents = []
		networkIndex = dataIndex

	#create the instances
	for neighborIndex in potentialAntecedents:
		d={}
		#convert indices to 32 bit ints for json module.
		# numpy uses 64 bit and pandas uses numpy under the hood
		d["index1"] = int(networkIndex)
		d["index2"] = int(neighborIndex)
		d["id1"] = ID
		d["id2"] = getID(nameEmbeddings,neighborIndex)
		d["name1"] = name
		d["name2"] = getName(nameEmbeddings,neighborIndex)

		#add a feature which is 1 iff the names are equal and 0 otherwise
		if d["name1"] == d["name2"]:
			d["namesEqual"] = 1
		else:
			d["namesEqual"] = 0

		#add the embeddings
		d["embedding1"] = embedding
		d["embedding2"] = getEmbedding(nameEmbeddings,neighborIndex)

		#add the class
		if neighborIndex in correctAntecedentIndices:
			d["isAntecedent"] = 1
			posExampleCount += 1
		else:
			d["isAntecedent"] = 0
			negExampleCount += 1

		#write the instance to the file
		outfile.write(json.dumps(d,ensure_ascii=False)+"\n")
print("Wrote %d positive samples and %d negative examples"%(posExampleCount,negExampleCount))