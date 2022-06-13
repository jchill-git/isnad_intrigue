#Given a collection of embedded mention and a network connecting them
# this script will use the provided antecedent classification model
# to infer clusters among the mentions in a sequential,
# document-by-document manner.
#
#NB: Mentions with no dates are ignored.
#
#This is done as follows:
# Each document's mentions are clustered at once.
# For a given documents mentions, we use the antecedent classifier
# to decide which (if any) potential antecedents (as selected by the network)
# the mention should be linked to. In cases where two mention in a document
# are assigned to a conflicting mention, we resolve conflicts by enforcing
# a matching constraint at the document level (i.e. any pairing of more
# than one mention in the same document to another document must be a
# matching) using the Hungarian algorithm (Kuhn, 1955).
#
#Author: Ryan Muther, May 2021

import sys
import os
import json
import argparse
import random
import re

import numpy as np
import networkx as nx
import pandas as pd
from scipy.sparse import load_npz
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import torch

from util import Dataset,FFNN

torch.manual_seed(42)

#Utility functions

#read in the graph adjacency matrix
def readGraph(path,threshold=0):
	adjMatrix = load_npz(path).toarray()
	#apply the threshold to prune edges if necessary
	if threshold > 0:
		adjMatrix[adjMatrix<threshold] = 0.0
	graph = nx.convert_matrix.from_numpy_matrix(adjMatrix,create_using=nx.Graph)
	print("Read %d-node graph with %d edges"%(len(graph.nodes()),len(graph.edges())))
	return graph

#converts a dataframe into a list of dicts representing
# the rows of the dataframe. This function is a bit not great
# but pandas to_dict gives a really funky structure that's a bit
# tricky to work with
#
#Ideally don't run this on a huge dataframe
def makeDictList(df):
	output = [{} for i in range(len(df))]
	for column,contents in df.to_dict().items():
		if column in ["index","level_0"]:
			continue

		for index,data in contents.items():
			output[index][column] = data
	return output

def check_cuda(model):
	if torch.cuda.is_available():
		cuda_device = "cuda:0"
		model = model.cuda(cuda_device)
	else:
		cuda_device = "cpu"
	return model,cuda_device

#converts a pair of mention info dicts into a tensor we can give the model
def makeInstance(e1,e2):
	if e1["name"] == e2["name"]:
		nameEqual = 1.0
	else:
		nameEqual = 0.0
	return torch.tensor([e1["embedding"]+e2["embedding"]+[nameEqual]])

#actually doing the thing functions

#given a collection of mentions and their antecedents,
# returns a list of those whose most likely antecedents
# overlap with others in the list and those that do not
#
#Sidenote, maybe the unambiguous ones are those whose
# most likely antecedents aren't possible antecedents
# instead of just most likely antecedents. It may be that
# another conflict will be introduced after resolving one issue
def findConflicts(mentions,clusters,idField):
	#get the cluster ids of the most likely antecedents for each mention
	mostLikely = []
	for m,antes in mentions:
		mostLikely.append(clusters[antes[0][1][idField]])

	conflicting = []
	unambiguous = []
	#for each mention, check if its most likely antecedent is also
	# assigned to another mention. if so, there's a conflict
	for i,mention in enumerate(mentions):
		#if the most likely antecedent shows up more than once in the list
		# of most likely assignments, we have a conflict
		if mostLikely.count(mostLikely[i]) > 1:
			conflicting.append(mention)
		else:
			unambiguous.append(mention)

	return conflicting,unambiguous

#for a given set of mentions, find which, if any, already seen mentions 
# are antecedents for those mentions and assignns them to clusters
# based on those decisions
#
#Returns the clustering and cluster count information updated to
# include the new document's mentions
#
#TODO: Clean up this function
def assignClusters(embeddings,docID,clusters,antecedentModel,mentionNet,clusterCount):
	#get the mentions for the current document
	# convert it to an easier-to-manipulate structure
	docMentions = embeddings[embeddings["docID"]==docID]
	docMentions = makeDictList(docMentions.reset_index())

	#some datasets have mention ids separate from normal ids,
	# due to needing to map gold entities to raw NER output.
	#IDs aren't necessarily grounded in the gold standard
	# while mentionIDs are (if both exist)
	if "mentionID" in docMentions[0]:
		idField = "mentionID"
	else:
		idField = "id"

	#enforce one sense per discorse constraint here
	# (find the unique names, cluster them, rather than clustering
	# all names including repeats)
	# We do this by ignoring all instances of a name after the first when
	# clustering, and putting the duplicate names in the same cluster as
	# the first instance at the end. 

	#record which names we've seen and which ID
	# corresponds to the first instance of each name
	duplicateNames = {}
	nameMap = {}
	seenNames = []
	for d in docMentions:
		if d["name"] not in seenNames:
			#If we haven't seen this name, add it to the list of names
			# we've seen and note that its first instance has this ID
			seenNames.append(d["name"])
			nameMap[d["name"]] = d[idField]
		else:
			#If we have seen the name already, note that this ID 
			# needs to be assigned the same cluster as the first instance of that name
			duplicateNames[d[idField]] = nameMap[d["name"]]

	#remove dupelicate names from the set of document mentions we're clustering
	docMentions = [d for d in docMentions if d[idField] not in duplicateNames]
	
	if len(duplicateNames) > 0:
		print("Found duplicate names. Here are the duplicate and first-instance IDs")
		print(duplicateNames)

	#get the potential antecedent's indices for each mention
	# from the network
	neighborIndicesAll = [list(mentionNet.neighbors(n["networkIndex"])) for n in docMentions]

	#convert those indices into the actual data about those mentions
	neighborsAll = []
	for indices in neighborIndicesAll:
		#filter the embeddings
		neighbors = makeDictList(embeddings[embeddings.networkIndex.isin(indices)].reset_index())
		#remove nodes that have yet to be clusterd (i.e. are later)
		# we do this by keeping those that are already assigned a cluster
		neighbors = [n for n in neighbors if n[idField] in clusters]

		neighborsAll.append(neighbors)

	#actually assign clusters to each mention now
	# this will be done by first iterating through the 
	# mentions in this document and assigning those who have no 
	# possible antecedents to new clusters.
	#
	#Then, those mentions that do have one or more antecedents will
	# be assigned separately, so we can resolve any potential conflicts
	
	# This will store a list of (mention, possible antecedent list) tuples
	unassignedMentions = []
	for mention,neighbors in zip(list(docMentions),neighborsAll):
		#if there are no possible antecedents, our job is done
		# for this mention
		if len(neighbors) == 0:
			#just assign the mention to a new cluster on its own
			clusters[mention[idField]] = clusterCount
			clusterCount += 1
		else:
			antecedents = []
			#convert each neighbor into an instance to give
			# to the antecedent classifier
			modelInput = [makeInstance(mention,n) for n in neighbors]
			#get the logit from the final layer of the model and run it
			# through a sigmoid for each potential antecedent. 
			# (this is done because the model was trained using a
			#  loss function that has the sigmoid activation built in)
			# This gives us p(antecedent) for each possible antecedent
			#Collect the mentions classified as antecedents together
			predictions = [torch.sigmoid(antecedentModel(pair)).item() for pair in modelInput]
			for i,n in enumerate(neighbors):
				if predictions[i] >= .5:
					antecedents.append((predictions[i],n))
			#sort them in descending order by likelihood
			antecedents.sort(key=lambda x:x[0],reverse=True)

			#if no possible antecedets are selected as antecedents
			# by the model, we're done here as above
			if len(antecedents) == 0:
				clusters[mention[idField]] = clusterCount
				clusterCount += 1
			else:
				unassignedMentions.append((mention,antecedents))
	
	conflictingMentions = []
	#If any mentions have conflicting most likely assignments, we will resolve
	# resolve the conflicts using the hungarian algorithm to ensure matching
	# between documents
	if len(unassignedMentions) > 0:
		#find non-conflicting mentions, assign them to clusters
		conflictingMentions,unambiguousMentions = findConflicts(unassignedMentions,clusters,idField)

		#assign non-conflicting mentions to their most likely
		# antecedent's cluster
		usedClusters = []
		for m,antecedents in unambiguousMentions:
			clusters[m[idField]] = clusters[antecedents[0][1][idField]]
			usedClusters.append(clusters[m[idField]])

		#resolve conflicts if necessary
		if len(conflictingMentions)>0:
			print("Conflicting most likely antecedent assignment.")
			print(" Resolve by enforcing matching constraint at document level")
			for mention,antecedents in conflictingMentions:
				print(mention[idField],mention["name"],mention["nameDisambiguated"])
				for prob,a in antecedents:
					print(" %d %3f %s %s %s"%(clusters[a[idField]],prob,a[idField],a["name"],a["nameDisambiguated"]))

			#reassign most likely mentions by eliminating conflicting
			# assignments as decided by a maximum matching algorithm
			# (i.e. two mentions both assign to the same antecedent would be resolved
			# by assigning the higher likelihood pair, while the lower likelihood pair
			# would be pruned from the list of antecedents for that mention. Potentially,
			# it could be assigned to another antecedent in a different document)
			resolvedMentions = resolveConflicts(conflictingMentions,idField,clusters,usedClusters)
			print("RESOLVED MENTIONS")
			for mention,antecedents in resolvedMentions:
				print(mention[idField],mention["name"],mention["nameDisambiguated"])
				for prob,a in antecedents:
					print(" %d %3f %s %s %s"%(clusters[a[idField]],prob,a[idField],a["name"],a["nameDisambiguated"]))
				print("================")	
				if len(antecedents) == 0:
					clusters[mention[idField]] = clusterCount
					clusterCount += 1
				else:
					clusters[mention[idField]] = clusters[antecedents[0][1][idField]]

	#sanity check: did we assign each mention in this document to a unique cluster?
	assignedClusters = []
	for mention in docMentions:
		cluster = clusters[mention[idField]]
		#this will need reworking for the one sense per discourse thing
		assert(cluster not in assignedClusters)
		assignedClusters.append(cluster)

	#assign the duplicate names to the clusters of their first instances
	for dupeID,originalID in duplicateNames.items():
		clusters[dupeID] = clusters[originalID]

	return clusters,clusterCount

#given a collection of mentions and possible antecedents,
# this function will reassign mentions to ensure a
# maximum-weight matching at the document level. This will return
# the same mention data, with constraint-violating potential antecedents
# removed
def resolveConflicts(mentions,idField,clusters,usedClusters):
	#need these to index the cost matrix we get from 
	# the assignment algorithm
	antecedentClusters = []
	thisDocIDs = []
	probabilities = {}
	clusterToMention = {}
	for mention,antecedents in mentions:
		mentionID = mention[idField]
		thisDocIDs.append(mentionID)
		for prob,antecedent in antecedents:
			anteID = antecedent[idField]
			cluster = clusters[anteID]
			if cluster not in usedClusters:
				if cluster not in antecedentClusters:
					antecedentClusters.append(cluster)
				if cluster not in clusterToMention:
					clusterToMention[cluster] = []
				clusterToMention[cluster].append(antecedent)
				
			#if there are multiple antecedents in one cluster,
			# the cost of that cluster for that mention will be the average
			# of the antecedent's costs
			if (mentionID,cluster) not in probabilities:
				probabilities[(mentionID,cluster)] = []
			probabilities[(mentionID,cluster)].append(prob)
	#do the averaging of cluster likelihoods across documents
	for mentionID,cluster in probabilities:
		probs = probabilities[(mentionID,cluster)]
		avgP = sum(probs)/len(probs)
		probabilities[(mentionID,cluster)] = avgP

	# print(probabilities)

	#construct the matrix
	#problem: we need the ability to assign a mention to no antecedent
	# in this setting when, for instance, you have a 2:1 mapping.
	# We're going to add a dummy antecedent with p=.5 to the matrix
	# as a false extra column. This might not be ideal, but it might help situations 
	# like the following:
	#    [.938 .567]
	#    [.508 0   ]
	# Even if the top left assignment is correct, this algorithm will pick 
	# bottom left and top right (.567+.508 > .938+0). Adding this extra column
	# predisposes the algorithm to pick a single higher score assignment
	# over two lower scoring assignments
	probMatrix = np.zeros((len(thisDocIDs),len(clusterToMention)+1))
	dummyIndex = len(clusterToMention)
	for i,mentionID in enumerate(thisDocIDs):
		for j,cluster in enumerate(antecedentClusters):
			if (mentionID,cluster) in probabilities:
				probMatrix[i,j] = probabilities[(mentionID,cluster)]
		probMatrix[i,dummyIndex] = .5
	print(probMatrix)
	# print("=====")

	#compute the best mapping between the 
	# items in this document and the items in the other document
	# with conflicting antecedents
	row_ind,col_ind = linear_sum_assignment(probMatrix,maximize=True)
	matching = list(zip(row_ind,col_ind))

	#get the ids of the pairs. This requires us to take the
	# id of the assigned cluster and select a mention from its
	# assigned cluster to represent it
	#Note: this sometimes isn't assigning everything properly. look into it?
	# as an example, try inferring clusters using untuned isnad embeddings 
	# and one of the high-k kNN graphs
	selectedPairs = [(thisDocIDs[m[0]],clusterToMention[antecedentClusters[m[1]]][0][idField]) for m in matching if probMatrix[m] != 0 and m[1] != dummyIndex]

	#now that we've assigned the mentions in this document to the mentions
	# in the antecedent doc (or perhaps unassigned them), we need to
	# remove antecedents that weren't selected
	resolvedMentions = []
	for m,antecedents in mentions:
		mentionID = m[idField]
		#add all the antecedents we didn't prune away
		newAntecedents = []
		for (prob,antecedent) in antecedents:
			anteID = antecedent[idField]
			if (mentionID,anteID) in selectedPairs:
				newAntecedents.append((prob,antecedent))
		resolvedMentions.append((m,newAntecedents))

	mentions = resolvedMentions
	return mentions

#data loading and bookkeeping
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("embeddingPath")
	parser.add_argument("networkPath")
	parser.add_argument("modelPath")
	parser.add_argument("outputPath")
	args = parser.parse_args()

	print("Loading mention embeddings")
	embeddings = pd.read_json(args.embeddingPath,lines=True,encoding='utf8')
	#remove ambiguous mentions if they exist (generally only for isnads)
	if "disambiguated" in embeddings.columns:
		embeddings = embeddings.query("disambiguated").reset_index().drop(columns="index")
	#add network indices to the data, so we can still find which node
	# in the network represents a mention after reordering
	embeddings["networkIndex"] = list(range(len(embeddings)))

	#reorder the embeddings so that they're ordered by date
	embeddings = embeddings.sort_values(by=["tokenStart","id"])
	embeddings = embeddings[embeddings["tokenStart"]!=-1].reset_index()
	print("Read %d mentions with start points"%len(embeddings))
	#get the documents ordered by date so we can easily retrieve 
	# mentions from a given document
	docsByDate = []
	for docID in embeddings["docID"]:
		if docID not in docsByDate:
			docsByDate.append(docID)

	print("Loading mention graph")
	mentionNetwork = readGraph(args.networkPath)

	print("Loading pretrained antecedent pair classification model from %s"%args.modelPath)
	inputSize = len(embeddings["embedding"][0])*2+1
	antecedentModel = FFNN(input_size=inputSize)
	#load the model to the cpu if we don't have a gpu
	if not torch.cuda.is_available():
		print("Model on CPU")
		antecedentModel.load_state_dict(torch.load(args.modelPath,map_location=torch.device('cpu')))
	else:
		print("Model on GPU")
		antecedentModel.load_state_dict(torch.load(args.modelPath))

	#set up cuda if we've got it
	antecedentModel,cuda_device = check_cuda(antecedentModel)
	#set the model to eval mode to remove dropout
	antecedentModel.eval()

	#finally, we can begin to infer clusters
	if "mentionID" in embeddings.columns:
		idField = "mentionID"
	else:
		idField = "id"

	#map the mention ids to the id of the cluster they're in
	clusters = {}
	clusterCount = 0
	#iterate over the documents (where each document is represented by 
	# a list of the embedded mentions in that document)
	for docID in tqdm(docsByDate):
		#do assign the mentions in this document to clusters 
		clusters,clusterCount = assignClusters(embeddings,docID,clusters,antecedentModel,mentionNetwork,clusterCount)
	print("Clustered %d mentions into %d clusters"%(len(clusters),clusterCount))

	#create the output data and write it
	clusterData = []
	for row in embeddings.drop(columns=["embedding"]).iterrows():
		ID = row[1][idField]
		nodeIndex = row[1]["networkIndex"]
		clusterNum = clusters[ID]
		mentionData = {"nodeIndex":nodeIndex,"mentionID":ID,"community":clusterNum}
		clusterData.append(mentionData)
	clusterData.sort(key=lambda x:x["community"])
	
	outfile = open(args.outputPath,"w",encoding="utf8")
	for c in clusterData:
		outfile.write(json.dumps(c)+"\n")