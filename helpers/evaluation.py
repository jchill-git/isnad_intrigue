from typing import Optional, List

import json
from scorch import scores

def create_communities_file(
    file_name: str,
    isnad_mention_ids: List[List[int]]
):
    f = open(file_name,"w",encoding="utf8")
    node_index=0
    for i in range(len(isnad_mention_ids)):
        for j in range(len(isnad_mention_ids[i])):
            nodeID="JK_000916_"+str(i)+"_"+str(j)
            community=isnad_mention_ids[i][j]
            f.write(json.dumps({"nodeIndex":node_index,"mentionID":nodeID,"community":community})+"\n")
            node_index+=1
    f.close()


def createScorchClusters(entities):
	communities = {}
	for entity in entities:
		ID = entity["mentionID"]
		cluster = int(entity["community"])

		if cluster not in communities:
			communities[cluster] = []
		communities[cluster].append(ID)
	return [set(c) for c in list(communities.values())]


def createClusters(path: str, max_samples: Optional[int] = None):
    modelEntities = [json.loads(l) for l in open(path,"r")]

    if max_samples is not None:
        modelEntities = modelEntities[:max_samples]

    return createScorchClusters(modelEntities)


def calc_conLL(goldClusters, modelClusters):
    print("Read %d gold clusters"%len(goldClusters))
    print("Read %d model clusters"%len(modelClusters))

    metricFs = {}
    for metric,func in [("MUC",scores.muc),("B_Cubed",scores.b_cubed),("CEAF_m",scores.ceaf_m),("CEAF_e",scores.ceaf_e),("BLANC",scores.blanc)]:
        score = func(goldClusters,modelClusters)
        metricFs[metric] = score[2]
        print("%s: P: %f R: %f F1: %f"%(metric,score[1],score[0],score[2]))
    conllScore = (metricFs["MUC"]+metricFs["B_Cubed"]+metricFs["CEAF_e"])/3
    print("CoNLL-2012 Score: %f"%conllScore)
