#This script will train a MLM on the given name-tagged dataset
# by converting each training instance into N instances
# one where each name is masked out in each. Rather than
# cross-validating, this trains one model on all the data at once. 
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

from transformers import BertTokenizerFast, BertForMaskedLM
from transformers import logging
from transformers import Trainer, TrainingArguments,get_linear_schedule_with_warmup

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

#given a collection of encodings and their labels, create a dataset
# uaing the data from the given indices
def makeDataset(texts,tags,indices,tokenizer):
	#select the texts and tags for the examples
	# we want to convert to a dataset
	if len(texts) > len(indices):
		textsSelected = [texts[i] for i in indices]
		tagsSelected = [tags[i] for i in indices]
	else:
		textsSelected = texts
		tagsSelected = tags

	#convert the set of documents into a larger dataset where
	# each name in each document is masked out once (# instances = # of names)
	#print("-Masking names")
	masked_texts,true_texts,instanceMap = makeMaskedNameData(textsSelected,tagsSelected,tokenizer,indices)

	#encode the masked texts
	#print("-Encoding Masked Texts")
	#not super clear what this does if you give it both padding=True
	# and a max length. look into? might allow faster training with larger batch?
	masked_texts = tokenizer(masked_texts, is_split_into_words=True, padding=True, truncation=True, max_length=512)

	#get the encodings of the true texts
	# to use as the labels for MLM training
	#print("-Encoding Unmasked Texts")
	encoded_true = tokenizer(true_texts, is_split_into_words=True, padding=True, truncation=True, max_length=512)
	true_texts = encoded_true.input_ids

	return Dataset(masked_texts,true_texts),instanceMap

#converts a set of tokens,tags lists into
# encoded representations for MLM training
# by masking out each name in turn
# also returns a map from the masked
# instance index to the unmasked instance index so we can refer
# to document-level metadata as well as a name index
def makeMaskedNameData(texts,tags,tokenizer,indices):
	masked_texts = []
	true_texts = []
	
	#record which masked instances come from which unmasked instances
	# as well as which name in the document was masked
	dataMap = {}
	dataIndex = 0
	for idx,text,tags in zip(indices,texts,tags):
		#determine what tokens the names are located at
		nameSpans = extractTaggedSpans(tags)

		#How do we mask tokens that tokenize into multiple subwords?
		# One [MASK] per word or one [MASK] per subword?
		# (the latter)
		for nameIdx,(nameStart,nameEnd) in enumerate(nameSpans):
			#create the token level mask
			tokenMask = [1 if (i < nameStart or i > nameEnd) else 0 for i in range(len(text))]
			maskedTokens = []
			for i in range(len(text)):
				#if the token isn't masked, just add it
				if tokenMask[i]:
					maskedTokens += [text[i]]
				#if it is, we need to find out how many subtokens it splits into
				# then add that many [MASK]s
				else:
					maskedTokens += ["[MASK]"]*(len(tokenizer([text[i]],is_split_into_words=True).input_ids)-2)
			unmaskedTokens = text

			masked_texts.append(maskedTokens)
			true_texts.append(unmaskedTokens)

			dataMap[dataIndex] = (idx,nameIdx)
			dataIndex += 1

	return masked_texts,true_texts,dataMap

parser = argparse.ArgumentParser()

parser.add_argument("infile")
parser.add_argument("modelPath")
args = parser.parse_args()

print("Reading data")
if os.path.isfile(args.infile):
	data = [json.loads(line) for line in open(args.infile,"r",encoding="utf8").readlines()]	
else:
	data = []
	for f in os.listdir(args.infile):
		if f.endswith(".json"):
			data += [json.loads(line) for line in open(os.path.join(args.infile,f),"r",encoding="utf8").readlines()]

data = [d for d in data if len(d["tags"])>0]

#convert the data into a pytorch dataset
# following the fine tuning tutorial from transformers

#to start, instantiate the tokenizer
tokenizer = BertTokenizerFast.from_pretrained("lanwuwei/GigaBERT-v3-Arabic-and-English", do_lower_case=True)

#encode all the texts in the format required by the model
texts_all = [d["tokens"] for d in data]
tags_all = [d["tags"] for d in data]

trainIndices = list(range(len(data)))

random.seed(42)
random.shuffle(trainIndices)

#create dataset objects for the training set
# there's probably a better way to do this.
print("Making dataset")
dataset_train,_ = makeDataset(texts_all,tags_all,trainIndices,tokenizer)

print("Tuning Model")
# if there are no checkpoints to use, train from scratch
checkpointDir = './results/singleRun'
if not os.path.exists(checkpointDir):
	latest = None
	#instantiate the model
	model = BertForMaskedLM.from_pretrained("lanwuwei/GigaBERT-v3-Arabic-and-English")
	print("Trainging from scratch, saving checkpoints to %s"%checkpointDir)
else:
	#otherwise find the latest checkpoint
	checkpoints = [(int(chk.split("-")[-1]),chk) for chk in os.listdir(checkpointDir)]
	checkpoints.sort(key=lambda x:x[0])
	latest = os.path.join(checkpointDir,checkpoints[-1][1])
	#instantiate the model
	print("Loading checkpoint %s"%latest)
	model = BertForMaskedLM.from_pretrained(latest)

#set up cuda and set the model to train mode
model.train()
model,cuda_device = check_cuda(model)

batchSize = 5

training_args = TrainingArguments(
	output_dir=checkpointDir,             # output directory
	num_train_epochs=3,                   # total number of training epochs
	per_device_train_batch_size=batchSize,# batch size per device during training
	per_device_eval_batch_size=batchSize, # batch size for evaluation
	warmup_steps=500,                     # number of warmup steps for learning rate scheduler
	logging_dir='./logs',                 # directory for storing logs
	logging_steps=10
)

trainer = Trainer(
	model=model,                         # the instantiated 🤗 Transformers model to be trained
	args=training_args,                  # training arguments, defined above
	train_dataset=dataset_train,         # training dataset
	tokenizer=tokenizer
)

trainer.train(latest)

model.save_pretrained(args.modelPath)