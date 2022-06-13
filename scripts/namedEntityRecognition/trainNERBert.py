#This script will train an NER model on the given data
import sys
import os
import json
import argparse
import random
import re
import pickle

import numpy as np

import torch
import torch.optim as optim

from transformers import BertTokenizerFast, BertForTokenClassification
from transformers import logging
from transformers import Trainer, TrainingArguments,get_linear_schedule_with_warmup

from sklearn.model_selection import KFold
from sklearn_crfsuite import metrics

from nerUtil import TaggedTokenDataset,encode_tags

torch.manual_seed(1)
logging.set_verbosity_error()

#arabic text normalization per Maxim Romanov
def normalizeArabicLight(text):
	new_text = text
	new_text = re.sub("[إأٱآا]", "ا", new_text)
	new_text = re.sub("[يى]ء", "ئ", new_text)
	new_text = re.sub("ى", "ي", new_text)
	new_text = re.sub("(ؤ)", "ء", new_text)
	new_text = re.sub("(ئ)", "ء", new_text)
	return new_text

#creates train/test splits
def makeTrainTestSplits(allDocs):
	splits = []
	splitMembership = {} 
	#gives a mapping from index to fold index so the right estimator can be used

	#if we're doing 10fold CV, use sklearn's KFold splitter
	# and record which fold each document was used as a test in
	cv = KFold(n_splits=10,shuffle=True,random_state=0)
	splits = list(cv.split(list(range(len(allDocs)))))
	fold = 0
	for trainIndices,testIndices in splits:
		for index in testIndices:
			splitMembership[index] = fold
		fold += 1

	return splits,splitMembership

def check_cuda(model):
	if torch.cuda.is_available():
		cuda_device = 0
		model = model.cuda(cuda_device)
	else:
		cuda_device = -1
	return model,cuda_device

#given a collection of encodings and their labels, create a dataset
# uaing the data from the given indices
def makeDataset(texts,tags,indices,tokenizer,tag2id):
	#select the texts and tags for the examples
	# we want to convert to a dataset
	textsSelected = [texts[i] for i in indices]
	tagsSelected = [tags[i] for i in indices]

	#encode the texts
	encoded_texts = tokenizer(textsSelected, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
	
	#now we've run into a problem. Subword tokenization
	# may leave us with more tokens than we have tags
	# we're going to ignore tags for subword tokens after
	# the first subword of a token by setting their label to -1
	encoded_labels = encode_tags(tagsSelected,encoded_texts,tag2id)

	encoded_texts.pop("offset_mapping")

	return TaggedTokenDataset(encoded_texts,encoded_labels)

def containsNonO(tags):
	for t in tags:
		if t not in ["PAD_TAG","PAD_TAG_DOC","O"]:
			return True
	return False

parser = argparse.ArgumentParser()

parser.add_argument("--test",action='store_true')
parser.add_argument("infile")
parser.add_argument("modelPath")
parser.add_argument("outfile")

args = parser.parse_args()

outfilePath = args.outfile

random.seed(42)

#iterate over the folds

#write the results, one per instance, to a json file
#if the file already exists, skip a fold if it's already been evaluated
if not os.path.exists(outfilePath):
	outfile = open(outfilePath,"w",encoding="utf8")
	finishedSplits = []
else:
	print("Found existing %s"%outfilePath)
	outfile = open(outfilePath,"r",encoding="utf8")
	finishedSplits = [int(json.loads(s)["foldIndex"]) for s in outfile.readlines()]
	outfile.close()

	print("Finished fold indices: %s"%sorted(list(set(finishedSplits))))

data = [json.loads(line) for line in open(args.infile,"r",encoding="utf8").readlines()]
data = [d for d in data if len(d["tags"])>0]

allSeenTags = []
for d in data:
	for t in d["tags"]:
		if t not in allSeenTags and t != "O":
			allSeenTags.append(t)

#create a mapping from tags to ids and vice versa
# adding the O tag manually since we don't really count it
# for evaluation
tag2id = {tag: id for id, tag in enumerate(["O"]+allSeenTags)}
id2tag = {id: tag for tag, id in tag2id.items()}
#print(tag2id)

splits,splitMembership = makeTrainTestSplits(data)

#convert the data into a pytorch dataset
# following the fine tuning tutorial from transformers

#to start, instantiate the tokenizer
tokenizer = BertTokenizerFast.from_pretrained("lanwuwei/GigaBERT-v3-Arabic-and-English", do_lower_case=True)

#encode all the texts in the format required by the model
texts_all = [d["tokens"] for d in data]
tags_all = [d["tags"] for d in data]

#if we're running a test on one fold, discard later folds
if args.test:
	splits = splits[:1]

foldIndex = 0
for trainIndices,testIndices in splits:
	if foldIndex in finishedSplits:
		print("Skipping fold %d"%foldIndex)
		foldIndex += 1
		continue

	#create the path we will save this fold's model to
	foldModelPath = os.path.join(args.modelPath,str(foldIndex))

	random.seed(42)

	random.shuffle(trainIndices)
	
	#get validation data indices
	valIndices = trainIndices[:len(testIndices)]
	trainIndices = trainIndices[len(testIndices):]

	#create dataset objects for training, test, and validation sets
	# there's probably a better way to do this.
	dataset_test = makeDataset(texts_all,tags_all,testIndices,tokenizer,tag2id)
	dataset_val = makeDataset(texts_all,tags_all,valIndices,tokenizer,tag2id)

	if not os.path.exists(foldModelPath):
		dataset_train = makeDataset(texts_all,tags_all,trainIndices,tokenizer,tag2id)
		
		#instantiate the model
		model = BertForTokenClassification.from_pretrained("lanwuwei/GigaBERT-v3-Arabic-and-English",num_labels=len(allSeenTags)+1)
		
		#setup cuda and set the model to train mode
		model.train()
		model,cuda_device = check_cuda(model)

		training_args = TrainingArguments(
			output_dir='./results/%d'%foldIndex, # output directory
			num_train_epochs=3,                  # total number of training epochs
			per_device_train_batch_size=5,       # batch size per device during training
			per_device_eval_batch_size=64,       # batch size for evaluation
			warmup_steps=500,                    # number of warmup steps for learning rate scheduler
			logging_dir='./logs',                # directory for storing logs
			logging_steps=10
		)

		trainer = Trainer(
			model=model,                         # the instantiated 🤗 Transformers model to be trained
			args=training_args,                  # training arguments, defined above
			train_dataset=dataset_train,         # training dataset
			eval_dataset=dataset_val,            # evaluation dataset
			tokenizer=tokenizer
		)

		trainer.train()

		model.save_pretrained(foldModelPath)
	else:
		#if we've already trained this model, load the pretrained model
		model = BertForTokenClassification.from_pretrained(foldModelPath,num_labels=len(allSeenTags)+1)
		
		print("Loaded existing model from %s"%foldModelPath)
		
		trainer = Trainer(
			model=model,                        # the instantiated 🤗 Transformers model to be trained
			eval_dataset=dataset_val,           # evaluation dataset
			tokenizer=tokenizer
		)

	#get predictions on the test set
	predictions = trainer.predict(dataset_test)

	#NOTE: there might be problems when we try to extract tags from 
	# text we don't have labels for, since we won't know which
	# subword token's tags to ignore
	outfile = open(outfilePath,"w",encoding="utf8")
	for i,dataIndex in list(enumerate(testIndices)):
		result = {}

		#get true tags
		trueTags = data[dataIndex]["tags"]

		#get model predictions and convert them to tags
		modelPredictions = np.argmax(predictions.predictions[i],axis=1)
		#we need to find which tags to ignore (for subtokens and padding),
		# that can be done using the label_ids field, ignoring -100
		labelIndicesToKeep = [index for index,labelID in enumerate(predictions.label_ids[i]) if labelID != -100]
		tagPredictions = [id2tag[tagID] for idx,tagID in enumerate(modelPredictions) if idx in labelIndicesToKeep]

		#compute token level evaluation metrics. span level to follow
		result["precision"] = metrics.flat_precision_score([trueTags],[tagPredictions],average="micro",labels=allSeenTags)
		result["recall"] = metrics.flat_recall_score([trueTags],[tagPredictions],average="micro",labels=allSeenTags)
		result["f1"] = metrics.flat_f1_score([trueTags],[tagPredictions],average="micro",labels=allSeenTags)
		result["sequence_acc"] = metrics.sequence_accuracy_score([trueTags],[tagPredictions])

		#get data from the original dataset to include with the output
		result["foldIndex"] = foldIndex
		result["bookID"] = data[dataIndex]["bookID"]
		result["id"] = data[dataIndex]["id"]
		result["true"] = trueTags
		result["predicted"] = tagPredictions
		result["tokens"] = data[dataIndex]["tokens"]

		#write output to file
		outfile.write(json.dumps(result,ensure_ascii=False)+"\n")
	foldIndex+=1