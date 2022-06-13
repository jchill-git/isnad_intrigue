#Given isnads with names marked with @Name_Beg@ and @Name_End@,
# creates tag data for use with NER models
# performs orthographic normalization

import json
import argparse

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

from tqdm import tqdm

#replaces _ with - in tags, because conll and I disagree on formatting
def fixTags(tags):
	for i in range(len(tags)):
		tags[i] = tags[i].replace("_","-")
	return tags

parser = argparse.ArgumentParser()
parser.add_argument("infile")
args = parser.parse_args()

#read in the data
data = [json.loads(s) for s in open(args.infile,"r",encoding="utf8").readlines()]

trueVals = [fixTags(d["true"]) for d in data]
predVals = [fixTags(d["predicted"]) for d in data]

print(classification_report(trueVals,predVals))