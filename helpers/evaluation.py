from typing import Optional, List

import json
from parse import parse
from scorch import scores

def create_communities_file(
    file_name: str,
    isnad_mention_ids: List[List[int]],
    test_mentions: Optional[List[List[bool]]],
):
    f = open(file_name,"w",encoding="utf8")
    node_index = 0
    for isnad_index, isnad in enumerate(isnad_mention_ids):
        for mention_index, community in enumerate(isnad):
            is_test = test_mentions[isnad_index][mention_index] if test_mentions is not None else True
            if is_test:
                mention_id = f"JK_000916_{isnad_index}_{mention_index}"
                f.write(
                    json.dumps({
                        "nodeIndex": node_index,
                        "mentionID": mention_id,
                        "community": int(community)
                    }) + "\n"
                )
                node_index += 1
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


def createClusters(path: str, test_mentions: Optional[List[List[bool]]] = None):
    model_entities = [json.loads(l) for l in open(path,"r")]

    # this seriously needs some cleaning up
    if test_mentions is not None:
        filtered_model_entities = []
        for model_entity in model_entities:
            isnad_index, mention_index = parse("JK_000916_{}_{}", model_entity["mentionID"])
            isnad_index = int(isnad_index)
            mention_index = int(mention_index)

            if isnad_index < len(test_mentions) and mention_index < len(test_mentions[isnad_index]):
                if test_mentions[isnad_index][mention_index]:
                    filtered_model_entities.append(model_entity)

        model_entities = filtered_model_entities

    return createScorchClusters(model_entities)


def calc_conLL(goldClusters, modelClusters):
    print("Read %d gold clusters" % len(goldClusters))
    print("Read %d model clusters" % len(modelClusters))

    metricFs = {}
    for metric,func in [("MUC",scores.muc),("B_Cubed",scores.b_cubed),("CEAF_m",scores.ceaf_m),("CEAF_e",scores.ceaf_e),("BLANC",scores.blanc)]:
        score = func(goldClusters,modelClusters)
        metricFs[metric] = score[2]
        print("%s: P: %f R: %f F1: %f"%(metric,score[1],score[0],score[2]))
    conllScore = (metricFs["MUC"]+metricFs["B_Cubed"]+metricFs["CEAF_e"])/3
    print("CoNLL-2012 Score: %f"%conllScore)
