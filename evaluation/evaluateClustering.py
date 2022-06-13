#given a collection of entity ids and cluster assignments in a json
# file, this script will run the evaluation functions from scorch
#
# https://pypi.org/project/scorch/
#
#Author: Ryan Muther, February 2021

import argparse
import sys
import os
import json
from scorch import scores

def createScorchClusters(entities):
	communities = {}
	for entity in entities:
		ID = entity["mentionID"]
		cluster = int(entity["community"])

		if cluster not in communities:
			communities[cluster] = []
		communities[cluster].append(ID)
	return [set(c) for c in list(communities.values())]

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--noSingletons",help="Remove singletons from the gold standard data and mentions with the corresponding ids from the model output",action="store_true")
	parser.add_argument("gold")
	parser.add_argument("model")
	args = parser.parse_args()

	#read in data
	goldEntities = [json.loads(l) for l in open(args.gold,"r")]
	modelEntities = [json.loads(l) for l in open(args.model,"r")]

	goldClusters = createScorchClusters(goldEntities)
	modelClusters = createScorchClusters(modelEntities)

	#remove singletons from the gold standard data
	# if required
	if args.noSingletons:
		#get all the non-singleton gold clusters
		goldClustersWithoutSingletons = [c for c in goldClusters if len(c) > 1]
		goldClustersSingletonIDs = [list(c)[0] for c in goldClusters if len(c) == 1]
		#problem: there are now a ton of mentions in the model
		# output that don't exist in the gold standard. We need to remove all
		# the singletons we just removed from the gold standard data
		modelClustersWithoutSingletons = []
		for c in modelClusters:
			newCluster = []
			for ID in c:
				if ID not in goldClustersSingletonIDs:
					newCluster.append(ID)
			if len(newCluster) > 0:
				modelClustersWithoutSingletons.append(set(newCluster))
		
		goldClusters = goldClustersWithoutSingletons
		modelClusters = modelClustersWithoutSingletons

	print("Read %d gold clusters"%len(goldClusters))
	print("Read %d model clusters"%len(modelClusters))

	metricFs = {}
	for metric,func in [("MUC",scores.muc),("B Cubed",scores.b_cubed),("CEAF_m",scores.ceaf_m),("CEAF_e",scores.ceaf_e),("BLANC",scores.blanc)]:
		scores = func(goldClusters,modelClusters)
		metricFs[metric] = scores[2]
		print("%s: P: %f R: %f F1: %f"%(metric,scores[1],scores[0],scores[2]))
	conllScore = (metricFs["MUC"]+metricFs["B Cubed"]+metricFs["CEAF_e"])/3
	print("CoNLL-2012 Score: %f"%conllScore)