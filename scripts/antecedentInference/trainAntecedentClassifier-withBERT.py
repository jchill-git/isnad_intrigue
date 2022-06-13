#This script will train a feedforward network to 
# decide, for a pair of mentions with embeddings (x1 and x2)
# are, if mention 2 is an antecedent of x2
#
#Unlike the standalone version of this model, this version will also
# fine tune the representations given to the model
#
#Given the input: x' = [x1,x2,phi(x)] where phi(x) is any ancillary
# features you want to use, this model will learn a nonlinear mapping
# to solve the binary classification problem posed above.
#
#In the input data, the target should be 0 if x2 is not an antecedent of x1
# and 1 if x2 is an antecedent of x1
#
#Author: Ryan Muther, October 2021

import sys
import os
import json
import argparse
import random
import re

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizerFast, BertModel

from sklearn.metrics import classification_report

from util import Dataset

torch.manual_seed(42)

#Implements a feedforward network roughly along the lines of 
# (Kenton 2019)'s end to end neural coref work on top of a BERT encoder.
# two hidden layers w/dropout and ReLU activation on top of
# bert and masking out irrelevant embeddings (those not in the
# entity in question)
class FFNNWithBERT(torch.nn.Module):
	def __init__(self,encoder,input_size=768*2+1,hidden_size=150):
		"""
		Here we instantiate the various layers of the net and dropout
		 This is probably the form of network I want?
		"""
		super(FFNNWithBERT, self).__init__()
		self.encoder = encoder
		self.linear1 = nn.Linear(input_size, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, 1)

		#define activations and dropout
		self.dropout = nn.Dropout(0.5)
		self.act = nn.ReLU()
		

	def forward(self,doc1,mask1,doc2,mask2,otheFeatures):
		"""
		This function takes in the (tokenized) documents to embed
		 as well as masks for each document to mask out all words
		 not of interest in each document as well as a vector
		 of other features (just name equality right now)

		Returns the logits (well, logit in the binary case) (the class likelihood)
		"""
		#embed the full documents
		embeddings1 = self.encoder(doc1)[0]
		embeddings2 = self.encoder(doc2)[0]
		# print(embeddings1)
		# print(embeddings1.shape)
		# print(mask1.shape)
		
		#mask the embeddings to only those in the words of interest
		# need to unsqueeze an extra dimension in to broadcast
		embeddings1 = torch.unsqueeze(mask1,dim=2)*embeddings1
		embeddings2 = torch.unsqueeze(mask2,dim=2)*embeddings2

		# print(embeddings1)
		# print(embeddings1.shape)

		#average nonzeros in the masked final hidden states
		# Autograd should be ok with this
		zeroMask1 = embeddings1!=0
		zeroMask2 = embeddings2!=0
		embeddings1 = (embeddings1*zeroMask1).sum(dim=1)/zeroMask1.sum(dim=1)
		embeddings2 = (embeddings2*zeroMask2).sum(dim=1)/zeroMask2.sum(dim=1)

		# print(embeddings1)
		# print(embeddings1.shape)

		#make the antecedent classifier input
		x = torch.cat((embeddings1,embeddings2,otheFeatures),dim=1)

		# Apply dropout
		x = self.dropout(x)
		#linear layer 1
		x = self.act(self.linear1(x))
		# Apply dropout
		x = self.dropout(x)
		#linear layer 2
		x = self.act(self.linear2(x))
		# Apply dropout
		x = self.dropout(x)
		#final linear layer
		logits = self.linear3(x)
		return logits.squeeze(1)

def check_cuda(model):
	if torch.cuda.is_available():
		cuda_device = "cuda:0"
		model = model.cuda(cuda_device)
	else:
		cuda_device = "cpu"
	return model,cuda_device

#returns the start and end index of a name in a given document's tokens
# NB: this will break if a name occurs multiple times. 
# Really only meant for isnads. Even those might be dicey.
def findNames(names,tokens):
	nameEndpoints = []
	startTok = 0
	for name in names:
		nameTokens = [t for t in name.split(" ") if len(t)>0]

		#find all possible starting points for the name (we check if
		# a token is contained rather than just equal to account for
		# removed alephs and such)
		possibleStarts = [i for (i,token) in enumerate(tokens) if nameTokens[0] in token and i>=startTok]

		found = False
		for start in possibleStarts:
			valid = True
			#if we found the name, stop looking
			if found:
				continue
			for i,token in enumerate(nameTokens[1:]):
				if valid and token not in tokens[start+i+1]:
					valid = False
			if valid:
				end = start + len(nameTokens) - 1
				nameEndpoints.append((start,end))
				startTok = end+1
				found = True
		if not found:
			nameEndpoints.append((-1,-1))
	return nameEndpoints

#given a full isnad and indices of start and end tokens,
# create a mask in which all subwords between the start and end are 
# true and everywhere else is false
def createMask(text,spanEndpoints):
	nameStart,nameEnd = spanEndpoints
	tokenMask = [0 if (i < nameStart or i > nameEnd) else 1 for i in range(len(text))]
	mask = []
	for i in range(len(text)):
		#if the token isn't masked, add the appropriate number of 0s
		if tokenMask[i]:
			mask += [1.0]*(len(tokenizer([text[i]],is_split_into_words=True).input_ids)-2)
		#if it is, we need to find out how many subtokens it splits into
		# then add that many 1s
		else:
			mask += [0.0]*(len(tokenizer([text[i]],is_split_into_words=True).input_ids)-2)
	#pad the mask to the maximum sequence length
	mask += (512-len(mask))*[0.0]
	return mask

#reads in a json file of instances and formats them for use
# in a dataset
# there is probably a better way to read this in I don't know if
# pytorch supports metadata being part of a dataset. doesn't seem to.
def readData(path,docs,tokenizer):
	data = [json.loads(line) for line in open(path,"r",encoding="utf8").readlines()]
	#add docs here to mention pairs here, determine locations of names in text
	for entry in data:
		if "embedding1" in entry:
			del entry["embedding1"]
			del entry["embedding2"]
		#get the id of the documents the names are taken from
		# by removing the name index from the mention id
		docID1 = "_".join(entry["id1"].split("_")[:-1])
		docID2 = "_".join(entry["id2"].split("_")[:-1])
		entry["docID1"] = docID1
		entry["docID2"] = docID2
		doc1 = docs[docID1]
		doc2 = docs[docID2]
		entry["doc1"] = doc1
		entry["doc2"] = doc2

		#find which tokens are in each doc are associated with which name
		# so we can construct the embedding masks
		entry["name1Indices"] = findNames([entry["name1"]],doc1)[0]
		entry["name2Indices"] = findNames([entry["name2"]],doc2)[0]

	inputs = {"doc1":[],"mask1":[],"doc2":[],"mask2":[],"otherFeatures":[]}
	true = []
	for d in tqdm(data):
		doc1 = d["doc1"]
		doc2 = d["doc2"]

		#A thought:
		#might be padding excessively? try to reduce this to one tokenizer
		# call so I can find the max length in the dataset more easily?
		#construct encoded texts, masks here
		encodedDoc1 = tokenizer.encode(doc1, is_split_into_words=True, max_length=512, truncation=True, padding="max_length")
		encodedDoc2 = tokenizer.encode(doc2, is_split_into_words=True, max_length=512, truncation=True, padding="max_length")

		mask1 = createMask(d["doc1"],d["name1Indices"])
		mask2 = createMask(d["doc2"],d["name2Indices"])

		inputs["doc1"].append(encodedDoc1)
		inputs["mask1"].append(mask1)
		inputs["doc2"].append(encodedDoc2)
		inputs["mask2"].append(mask2)
		inputs["otherFeatures"].append([d["namesEqual"]])
		true.append(np.array(d["isAntecedent"]).astype("float"))
	
	return data,inputs,true

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--earlyStop",help="Stop when validation accuracy begins to decrease",action="store_true")
	parser.add_argument("--patience",help="The number of epochs without improvement to stop after",default=5)
	parser.add_argument("docPath",help="The path to the full texts of the documents, needed for creating the contextual embeddings")
	parser.add_argument("trainPath",help="The path to the training dataset as a collection of mentions pairs. Embeddings may be included but will be ignored")
	parser.add_argument("encoderPath",help="The path to the bert model to start with for encoding mentions")
	parser.add_argument("validationPath",help="The path to the evaluation dataset")
	parser.add_argument("modelPath",help="The path to save the model to")
	parser.add_argument("evalPath",help="The path to write the evaluation output to")
	args = parser.parse_args()

	print("Instantiating tokenizer")
	tokenizer = BertTokenizerFast.from_pretrained("lanwuwei/GigaBERT-v3-Arabic-and-English", do_lower_case=True)
	print("Loading encoder")
	if args.encoderPath == "base":
		encoder = BertModel.from_pretrained("lanwuwei/GigaBERT-v3-Arabic-and-English")
	else:
		encoder = BertModel.from_pretrained(args.encoderPath)

	print("Reading documents")
	docs = [json.loads(line) for line in open(args.docPath,"r",encoding="utf8").readlines()]
	#convert docs to a dict of id,token list pairs
	docs = dict([(d["id"],d["tokens"]) for d in docs])

	#read in the training data
	# some examples have only one entry (those mentions whose
	# near neighbors are all later in time)
	print("Reading training data")
	data_train,input_train,true_train = readData(args.trainPath,docs,tokenizer)

	#create dataset object for the training set
	# there's probably a better way to do this.
	dataset_train = Dataset(input_train,true_train)
	loader_train = torch.utils.data.DataLoader(dataset_train,shuffle=True,batch_size=4)

	#as above but for validation data
	print("Reading validation data")
	data_val,input_val,true_val = readData(args.validationPath,docs,tokenizer)
	dataset_val = Dataset(input_val,true_val)
	loader_val = torch.utils.data.DataLoader(dataset_val,shuffle=False,batch_size=1)

	#make a checkpoint directory if we don't have one
	checkpointDir = args.modelPath.split(".pt")[0]+"_checkpoints"
	if not os.path.isdir(checkpointDir):
		os.mkdir(checkpointDir)

	if not os.path.exists(args.modelPath):
		#instantiate the model
		model = FFNNWithBERT(encoder)
		
		#set up cuda if we've got it
		model,cuda_device = check_cuda(model)
		#print(model) #yep, that's BERT

		#set up bookkeeping for early stopping
		#note: if checkpointed, early stopping will break
		# since the best validation and epochs without
		# improvement aren't saved
		startEpoch = 0
		bestValLoss = -1
		epochsWithoutImprovement = 0

		#create the loss and optimizer
		criterion = torch.nn.BCEWithLogitsLoss() # | || || |_
		optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

		#load the latest checkpoint if one exists
		checkpoints = [os.path.join(checkpointDir,f) for f in os.listdir(checkpointDir)]
		checkpoints = sorted(checkpoints,reverse=True,key=lambda x:int(x.split("_")[-1]))

		if len(checkpoints)>0:
			latest = checkpoints[0]

			#load the model and optimizer
			checkpoint = torch.load(latest)
			model.load_state_dict(checkpoint['model_state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

			#figure out what epoch we were on
			lastEpoch = checkpoint['epoch']
			trainLoss = checkpoint['train_loss']
			valLoss = checkpoint['val_loss']
			print("Loaded checkpoint from epoch %d with training loss %f"%(lastEpoch,trainLoss))
			
			#set the model in training mode and resume
			# training starting at the correct epoch
			startEpoch = lastEpoch + 1
			model.train()

		if startEpoch>0:
			print("Resuming from checkpoint %d"%startEpoch)
		else:
			print("No checkpoints found. Training Model from scratch")
		for t in range(startEpoch,500):
			#reseed the rng so we get consistent results when resuming from a checkpoint
			torch.manual_seed(42 + t)
			
			trainLoss = 0.0
			for batchNum,sample in enumerate(loader_train):
				#send things to the right device
				local_doc1 = sample["doc1"].to(cuda_device)
				local_mask1 = sample["mask1"].to(cuda_device)
				local_doc2 = sample["doc2"].to(cuda_device)
				local_mask2 = sample["mask2"].to(cuda_device)
				local_otherFeats = sample["otherFeatures"].type("torch.FloatTensor").to(cuda_device)
				local_labels = sample["labels"].to(cuda_device)
				
				# Forward pass: Compute predicted y by passing the inputs to the model
				#might be a way to unpack the dict into individual items
				# rather than giving a long parameter list?
				y_pred = model(local_doc1,local_mask1,local_doc2,local_mask2,local_otherFeats)

				# Compute loss
				loss = criterion(y_pred, local_labels)
				trainLoss += loss.item()

				# Zero gradients, perform a backward pass, and update the weights.
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				if batchNum%250==0:
					print("Done %d batches"%batchNum)

			# #run validation every epoch
			# with torch.no_grad():
			# 	#swap to eval mode to not use dropout
			# 	model.eval()
			# 	valLoss = 0.0
			# 	for batchNum,sample in enumerate(loader_val):
			# 		#send things to the right device
			# 		local_doc1 = sample["doc1"].to(cuda_device)
			# 		local_mask1 = sample["mask1"].to(cuda_device)
			# 		local_doc2 = sample["doc2"].to(cuda_device)
			# 		local_mask2 = sample["mask2"].to(cuda_device)
			# 		local_otherFeats = sample["otherFeatures"].type("torch.FloatTensor").to(cuda_device)
			# 		local_labels = sample["labels"].to(cuda_device)
			# 		local_labels = sample["labels"].to(cuda_device)
					
			# 		# Forward pass: Compute predicted y by passing x to the model
			# 		y_pred = model(local_doc1,local_mask1,local_doc2,local_mask2,local_otherFeats)

			# 		# Compute and print loss
			# 		loss = criterion(y_pred, local_labels)
			# 		valLoss += loss

			if args.earlyStop:
				if t==0:
					bestValLoss = valLoss
				elif bestValLoss > valLoss:
					bestValLoss = valLoss
					epochsWithoutImprovement = 0
					print("Improved loss")
				elif valLoss > bestValLoss:
					epochsWithoutImprovement += 1
					print("No improvement for %d epochs"%epochsWithoutImprovement)

				if epochsWithoutImprovement == args.patience:
					print("Validation loss increased for %d epochs. Stopping early"%args.patience)
					break

				model.train()
			print("Epoch %d training loss: %f"%(t,trainLoss))
			# print("Epoch %d validation loss: %f"%(t,valLoss))

			#save a checkpoint every 10 epochs
			if t>0 and t%10==0:
				checkpointPath = os.path.join(checkpointDir,"checkpoint_%d"%t)
				torch.save({
					'epoch': t,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'train_loss': trainLoss,
					'val_loss': valLoss
					}, checkpointPath)
				print("Saved checkpoint to %s"%checkpointPath)

		#save the trained model
		torch.save(model.state_dict(), args.modelPath)
		model.eval()
	else:
		print("Loading pretrained model from %s"%args.modelPath)
		inputSize = dataset_train[0]["input"].shape[0]
		model = FFNNWithBERT(encoder)
		model.load_state_dict(torch.load(args.modelPath))
		#set up cuda if we've got it
		model,cuda_device = check_cuda(model)
		model.eval()
		
	# model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	# params = sum([np.prod(p.size()) for p in model_parameters])
	# print(params)

	#evaluate the model using the dev set
	outfile = open(args.evalPath,"w",encoding="utf8")
	print("Evaluating on test set")
	count = 0
	trueVals = []
	predVals = []
	for i,instance in tqdm(enumerate(loader_val)):
		#get the input and output for this instance
		local_batch = instance["input"].to(cuda_device)
		local_labels = instance["labels"].to(cuda_device)
		#run the model on that instance
		y_pred = model(local_batch)

		#predict if the pair consists of a mention and its antecedent
		prob = torch.sigmoid(y_pred)
		y_pred = torch.round(prob)

		# print(local_labels)
		# print(y_pred)
		# print(local_labels.item())
		# print(y_pred.item())
		# print(torch.sigmoid(model(local_batch)))

		#assemble the output for the instance
		d = {}
		d["id1"] = data_val[i]["id1"]
		d["id2"] = data_val[i]["id2"]
		d["index1"] = data_val[i]["index1"]
		d["index2"] = data_val[i]["index2"]
		d["true"] = local_labels.item()
		d["pred"] = y_pred.item()
		d["prob"] = prob.item()
		correct = (local_labels == y_pred).item()
		d["correct"] = correct

		#record this instance for the classification report
		trueVals.append(int(d["true"]))
		predVals.append(int(d["pred"]))

		#write the instance
		outfile.write(json.dumps(d)+"\n")
		count += 1

	print("Wrote predictions for %d instances to %s"%(count,args.evalPath))
	print(classification_report(trueVals,predVals,target_names=["Not Antecedent","Antecedent"]))
