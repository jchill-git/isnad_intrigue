#given a collection of entity ids and cluster assignments in a json
# file, this script will plot a histogram of cluster sizes.
#
#Author: Ryan Muther, June 2022

import argparse
import sys
import os
import json
import matplotlib.pyplot as plt


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("clusters")
	parser.add_argument("graphTitle")
	parser.add_argument("out")
	args = parser.parse_args()

	#read in data
	clusters = [json.loads(l) for l in open(args.clusters,"r")]

	clusterSizesByID = {}
	for entry in clusters:
		clusterID = entry["community"]
		if clusterID not in clusterSizesByID:
			clusterSizesByID[clusterID] = 0
		clusterSizesByID[clusterID] += 1

	clusterSizes = list(clusterSizesByID.values())

	plt.hist(clusterSizes)
	plt.xlabel("Cluster Size")
	plt.ylabel("Frequency")
	plt.title(args.graphTitle)
	plt.savefig(args.out)
	plt.show()