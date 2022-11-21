#This script embeds names in a tagged corpus using a pretrained MLM
#
#Author; Ryan Muther, January 2021
import sys
import os
import json
import argparse
import random
import re

import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim

from transformers import BertTokenizerFast, BertModel
from transformers import logging

from sklearn.model_selection import KFold

from util import Dataset,extractTaggedSpans

torch.manual_seed(1)
logging.set_verbosity_error()

def check_cuda(model):
    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1
    return model,cuda_device

#embeds each name in a given data instance, returns a list of dicts
# containing information about each embedded name and where it came from
def processDocument(instance):
    tokens = instance["tokens"]
    tags = instance[args.tagField]

    #determine which token spans are names
    nameSpans = extractTaggedSpans(tags)

    # print(len(nameSpans))

    #find which tokenized indices refer to which token, since
    # multiple subword tokens may exist for each token, ignoring
    # the [CLS] special token and the [SEP] special token
    tokenizedText = tokenizer(tokens, is_split_into_words=True, return_offsets_mapping = True, max_length=512)
    tokenMap = [0,0]
    for tStart,tEnd in tokenizedText["offset_mapping"][2:]:
        if tStart == 0:
            newIndex = tokenMap[-1] + 1
        else:
            newIndex = tokenMap[-1]
        tokenMap.append(newIndex)

    # print(tokenMap)

    #embed the document's text using the model
    #print(tokens)
    embeddings = model(torch.tensor(tokenizer.encode(tokens, is_split_into_words=True, max_length=512)).unsqueeze(0))[0][0]
    #print(embeddings[0])
    # print(embeddings.shape)
    # print(tags)

    #extract the name's embedding sequences
    results = []
    for i,(nameStart,nameEnd) in enumerate(nameSpans):
        result = {}

        #get data from the original dataset to include with the output
        if "bookID" in instance:
            result["bookID"] = instance["bookID"]
        result["docID"] = instance["id"]
        result["id"] = instance["id"]+"_"+str(i)
        result["name"] = " ".join(tokens[nameStart:nameEnd+1])
        result["tStart"] = nameStart
        result["tEnd"] = nameEnd

        #get the embedding for that name
        # print(result["id"])
        # print(nameStart,nameEnd)
        # print(result["name"])

        #first get the subword token indices we're looking for
        # ignoring the start and end padding tokens
        tokenIndices = [i for i,tokIdx in enumerate(tokenMap) if (tokIdx >= nameStart and tokIdx <= nameEnd) and i > 0 and i < len(tokenMap)-1]

        #then get the embeddings at those indices
        #print(tokenIndices)
        nameEmbeddings = embeddings[min(tokenIndices):max(tokenIndices)+1]
        #print(nameEmbeddings)
        #exit(0)

        #average the embeddings for all the subwords in the name to get
        # the name's embedding
        result["embedding"] = torch.mean(nameEmbeddings.detach(),dim=0).numpy().tolist()

        results.append(result)

    return results

parser = argparse.ArgumentParser()

parser.add_argument("--tagField",default="tags",required=False,help="The name of the field to use for named entity tags (default: tags)")
parser.add_argument("infile")
parser.add_argument("modelPath",help="the path to the model to use for embedding. If 'base' is passed, uses the un-finetuned model rather than loading a trained one")
parser.add_argument("outfile")

args = parser.parse_args()

outfilePath = args.outfile

data = [json.loads(line) for line in open(args.infile,"r",encoding="utf8").readlines()]
data = [d for d in data if len(d[args.tagField])>0]

#to start, instantiate the tokenizer
tokenizer = BertTokenizerFast.from_pretrained("lanwuwei/GigaBERT-v3-Arabic-and-English", do_lower_case=True)

#encode all the texts in the format required by the model
texts_all = [d["tokens"] for d in data]
tags_all = [d[args.tagField] for d in data]

if args.modelPath == "base":
    model = BertModel.from_pretrained("lanwuwei/GigaBERT-v3-Arabic-and-English")
else:
    model = BertModel.from_pretrained(args.modelPath)

#Using the trained model, embed all the names
outfile = open(outfilePath,"w",encoding="utf8")
for index in tqdm(range(len(data))):
    #get the test instance and embed its names
    instance = data[index]
    embeddedNames = processDocument(instance)

    #write output to file
    for entry in embeddedNames:
        outfile.write(json.dumps(entry,ensure_ascii=False)+"\n")

    break

outfile.close()
print("Wrote %d documents worth of output rows to %s"%(len(data),outfilePath))
